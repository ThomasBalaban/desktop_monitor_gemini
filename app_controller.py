import tkinter as tk
from datetime import datetime
import threading
import time
import os

from config_loader import ConfigLoader
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from app_gui import AppGUI
from websocket_server import WebSocketServer, WEBSOCKET_PORT

# OpenAI Realtime imports
from openai_realtime_client import OpenAIRealtimeClient
from transcriber_core.openai_streamer import SmartAudioTranscriber


class AppController:
    def __init__(self):
        self.config = ConfigLoader()
        print("Gemini Screen Watcher (Unified Vision+Audio) - Starting up...")
        
        # 1. Initialize Screen Capture
        self.screen_capture = ScreenCapture(
            self.config.image_quality, 
            video_index=self.config.video_device_index
        )
        
        # 2. Initialize Gemini Client (handles both vision AND audio interpretation)
        self.gemini_client = GeminiClient(
            self.config.api_key, 
            self.config.prompt, 
            self.config.safety_settings,
            self._on_gemini_response, 
            self._on_gemini_error, 
            self.config.max_output_tokens, 
            self.config.debug_mode,
            audio_sample_rate=self.config.audio_sample_rate
        )

        # 3. Initialize Streaming Manager (sends frames + buffered transcripts to Gemini)
        self.streaming_manager = StreamingManager(
            self.screen_capture, self.gemini_client, self.config.fps,
            restart_interval=1500, debug_mode=self.config.debug_mode
        )
        self.streaming_manager.set_restart_callback(self.on_stream_restart)
        self.streaming_manager.set_error_callback(self._on_streaming_error)
        self.streaming_manager.set_preview_callback(self.gui_update_wrapper)
        
        # 4. Initialize WebSocket Server
        self.websocket_server = WebSocketServer()
        self.current_response_buffer = ""
        
        # 5. Initialize GUI
        self.gui = AppGUI(self)
        
        # 6. Initialize OpenAI Realtime (Whisper for transcription only)
        if not self.config.is_openai_key_configured():
            print("⚠️ WARNING: OPENAI_API_KEY not configured. Audio transcription will not work.")
            self.gui.add_error("OPENAI_API_KEY missing in api_keys.py")
            self.openai_client = None
            self.smart_transcriber = None
        else:
            self.openai_client = OpenAIRealtimeClient(
                api_key=self.config.openai_api_key,
                on_transcript=self._handle_whisper_transcript,
                on_error=self._on_openai_error
            )
            self.smart_transcriber = SmartAudioTranscriber(
                self.openai_client, 
                device_id=self.config.audio_device_id
            )

        # Only init region if we are NOT using a camera
        if self.config.video_device_index is None:
            self._initialize_capture_region()
            
        # Delay stream start
        self.gui.root.after(2000, self._start_stream_on_init)

    def gui_update_wrapper(self, frame):
        if self.gui:
            self.gui.update_preview(frame)

    def run(self):
        # Start Audio Transcription (Whisper)
        if self.smart_transcriber:
            print(f"🎙️ Starting OpenAI Whisper on Device {self.config.audio_device_id}...")
            self.smart_transcriber.start()
        
        if not self.config.is_api_key_configured():
            self.gui.update_status("ERROR: GEMINI_API_KEY not configured", "red")
            self.gui.add_error("GEMINI_API_KEY not configured.")
        
        self.websocket_server.start()
        
        try:
            self.gui.run()
        finally:
            print("Shutting down services...")
            if self.smart_transcriber:
                self.smart_transcriber.stop()
            self.streaming_manager.stop_streaming()

    # --- Whisper Transcript Handling ---
    
    def _handle_whisper_transcript(self, transcript):
        """
        Receives raw Whisper transcription.
        Buffers it for Gemini to process with the next pulse.
        """
        print(f"🎤 [Whisper]: {transcript}")
        
        # Add to streaming manager's buffer (will be sent to Gemini on next pulse)
        self.streaming_manager.add_transcript(transcript)
        
        # Also broadcast raw transcript via WebSocket (in case downstream wants it)
        self.websocket_server.broadcast({
            "type": "raw_transcript",
            "source": "whisper",
            "text": transcript,
            "timestamp": time.time()
        })

    def _on_openai_error(self, error_msg):
        print(f"❌ OpenAI Error: {error_msg}")
        self.gui.add_error(f"OpenAI Error: {error_msg}")

    # --- Gemini Response Handling ---

    def _on_gemini_response(self, text_chunk):
        """Handle Gemini's unified vision+audio analysis response."""
        self.current_response_buffer += text_chunk
        
        # Wait for complete sentences
        if self.current_response_buffer.strip().endswith(('.', '!', '?', '"', '\n')):
            final_text = self.current_response_buffer.strip()
            
            # Display in GUI
            self.gui.add_response(final_text)
            print(f"🎭 [SCREEN]: {final_text}")
            
            # Broadcast unified analysis via WebSocket
            self.websocket_server.broadcast({
                "type": "screen_analysis",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            })
            
            self.current_response_buffer = ""

    def _on_gemini_error(self, error_message):
        self.gui.add_error(f"Gemini API Error: {error_message}")

    def _on_streaming_error(self, error_message):
        self.gui.add_error(f"Streaming Error: {error_message}")

    # --- Other Methods ---

    def request_analysis(self):
        print("Manual analysis triggered.")
        self.gui.update_status("Requesting Analysis...", "cyan")
        self.streaming_manager.trigger_manual_analysis(
            "Describe exactly what is happening on screen right now, including any audio/dialogue."
        )

    def _start_stream_on_init(self):
        if not self.screen_capture.is_ready():
            self.gui.update_status("Cannot start. No source configured.", "red")
            print("ERROR: No Camera Index AND No Screen Region set.")
            return
        
        def run_check_and_start():
            print("Checking Gemini API connection...")
            api_ok, message = self.gemini_client.test_connection()
            self.gui.root.after(0, self._finalize_start, api_ok, message)
        
        self.gui.update_status("Checking API connection...", "orange")
        threading.Thread(target=run_check_and_start, daemon=True).start()

    def _finalize_start(self, api_ok, message):
        if not api_ok:
            self.gui.add_error(f"API Connection Check Failed: {message}")
            self.gui.update_status("API Check Failed", "red")
            return

        print("API connection successful. Starting streaming...")
        self.streaming_manager.set_status_callback(self.gui.update_status)
        self.streaming_manager.start_streaming()
        self.gui.update_status("Connecting...", "orange")

    def update_websocket_gui_status(self):
        self.gui.update_websocket_status(f"Running at ws://localhost:{WEBSOCKET_PORT}", "#4CAF50")

    def on_stream_restart(self):
        self.gui.add_reset_separator()

    def _initialize_capture_region(self):
        if self.config.capture_region:
            self.screen_capture.set_capture_region(self.config.capture_region)
            print(f"Using capture region from config: {self.config.get_region_description()}")
        else:
            print("No capture region set (will rely on GUI if not using camera)")

    def get_prompt(self):
        return self.config.prompt

    def get_timestamp(self):
        return datetime.now().strftime("%I:%M:%S %p")