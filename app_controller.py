import tkinter as tk
from datetime import datetime
import threading
import time
import os
from queue import Empty

from config_loader import ConfigLoader
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from app_gui import AppGUI
from websocket_server import WebSocketServer, WEBSOCKET_PORT
from transcriber_core.microphone import MicrophoneTranscriber

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
        
        # 2. Initialize Gemini Client
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
        
        # 3. Initialize Parakeet Microphone Transcriber
        self.mic_transcriber = MicrophoneTranscriber(keep_files=False)
        self.mic_polling_active = True 

        # 4. Initialize Streaming Manager
        self.streaming_manager = StreamingManager(
            self.screen_capture, self.gemini_client, self.config.fps,
            restart_interval=1500, debug_mode=self.config.debug_mode
        )
        self.streaming_manager.set_restart_callback(self.on_stream_restart)
        self.streaming_manager.set_error_callback(self._on_streaming_error)
        self.streaming_manager.set_preview_callback(self.gui_update_wrapper)
        
        # 5. Initialize WebSocket Server
        self.websocket_server = WebSocketServer()
        self.current_response_buffer = ""
        
        # 6. Initialize GUI
        self.gui = AppGUI(self)
        
        # 7. Initialize OpenAI Realtime
        if not self.config.is_openai_key_configured():
            print("⚠️ WARNING: OPENAI_API_KEY not configured. Desktop audio transcription will not work.")
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

        if self.config.video_device_index is None:
            self._initialize_capture_region()
            
        self.gui.root.after(2000, self._start_stream_on_init)

    def gui_update_wrapper(self, frame):
        if self.gui:
            self.gui.update_preview(frame)

    def run(self):
        print(f"🎙️ Starting Parakeet MLX Microphone Transcriber...")
        threading.Thread(target=self.mic_transcriber.run, daemon=True).start()
        threading.Thread(target=self._poll_mic_transcripts, daemon=True).start()

        if self.smart_transcriber:
            print(f"🔊 Starting OpenAI Whisper for Desktop Audio on Device {self.config.audio_device_id}...")
            self.smart_transcriber.start()
        
        if not self.config.is_api_key_configured():
            self.gui.update_status("ERROR: GEMINI_API_KEY not configured", "red")
            self.gui.add_error("GEMINI_API_KEY not configured.")
        
        self.websocket_server.start()
        
        try:
            self.gui.run()
        finally:
            self.stop() # Ensure all services stop if gui.run exits

    def stop(self):
        """Unified shutdown logic to stop all threads and clean up resources."""
        print("🛑 Shutting down services...")
        
        # Stop polling loops
        self.mic_polling_active = False
        
        # Stop background managers/services
        self.streaming_manager.stop_streaming()
        
        if hasattr(self, 'mic_transcriber'):
            # Signal the actual audio capture loop to stop
            self.mic_transcriber.stop_event.set() 
            
        if self.smart_transcriber:
            self.smart_transcriber.stop()
            
        self.websocket_server.stop()
        
        # Kill GUI if still alive
        try:
            if self.gui.root.winfo_exists():
                self.gui.root.quit()
                self.gui.root.destroy()
        except:
            pass

    def _poll_mic_transcripts(self):
        print("🎤 Microphone transcript polling started...")
        while self.mic_polling_active:
            try:
                text, filename, source, confidence = self.mic_transcriber.result_queue.get(timeout=0.1)
                
                if text and len(text.strip()) > 0:
                    print(f"🎙️ [Mic/Parakeet]: {text}")
                    self.streaming_manager.add_transcript(f"[USER]: {text}")
                    # UPDATED: Changed 'raw_transcript' to 'transcript' for Director Engine
                    self.websocket_server.broadcast({
                        "type": "transcript",
                        "source": "microphone",
                        "text": text,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Mic polling error: {e}")
                time.sleep(0.1)
        print("🎤 Microphone transcript polling stopped.")

    def _handle_whisper_transcript(self, transcript):
        print(f"🔊 [Desktop/Whisper]: {transcript}")
        self.streaming_manager.add_transcript(f"[AUDIO]: {transcript}")
        # UPDATED: Changed 'raw_transcript' to 'transcript' and source to 'desktop'
        self.websocket_server.broadcast({
            "type": "transcript",
            "source": "desktop",
            "text": transcript,
            "timestamp": time.time()
        })

    def _on_openai_error(self, error_msg):
        print(f"❌ OpenAI Error: {error_msg}")
        self.gui.add_error(f"OpenAI Error: {error_msg}")

    def _on_gemini_response(self, text_chunk):
        self.current_response_buffer += text_chunk
        if self.current_response_buffer.strip().endswith(('.', '!', '?', '"', '\n')):
            final_text = self.current_response_buffer.strip()
            self.gui.add_response(final_text)
            print(f"🎭 [SCREEN]: {final_text}")
            # UPDATED: Changed 'screen_analysis' to 'text_update' and field 'content'
            self.websocket_server.broadcast({
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            })
            self.current_response_buffer = ""

    def _on_gemini_error(self, error_message):
        self.gui.add_error(f"Gemini API Error: {error_message}")

    def _on_streaming_error(self, error_message):
        self.gui.add_error(f"Streaming Error: {error_message}")

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

    def get_prompt(self):
        return self.config.prompt

    def get_timestamp(self):
        return datetime.now().strftime("%I:%M:%S %p")