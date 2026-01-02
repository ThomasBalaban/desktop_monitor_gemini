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

# Transcript Enricher (GPT-4o speaker diarization)
from transcript_enricher import TranscriptEnricher


class AppController:
    def __init__(self):
        self.config = ConfigLoader()
        print("Gemini Screen Watcher (Unified Vision+Audio) - Starting up...")
        
        # Track if we're shutting down to prevent duplicate cleanup
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        
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
        
        # 7. Initialize OpenAI Realtime + Transcript Enricher
        self.smart_transcriber = None
        self.transcript_enricher = None
        self.last_gemini_context = ""  # Store latest Gemini visual analysis
        
        if not self.config.is_openai_key_configured():
            print("⚠️ WARNING: OPENAI_API_KEY not configured. Desktop audio transcription will not work.")
            self.gui.add_error("OPENAI_API_KEY missing in api_keys.py")
            self.openai_client = None
        else:
            # OpenAI Realtime for transcription
            self.openai_client = OpenAIRealtimeClient(
                api_key=self.config.openai_api_key,
                on_transcript=self._handle_whisper_transcript,
                on_error=self._on_openai_error
            )
            self.smart_transcriber = SmartAudioTranscriber(
                self.openai_client, 
                device_id=self.config.audio_device_id
            )
            
            # GPT-4o Transcript Enricher for speaker diarization
            self.transcript_enricher = TranscriptEnricher(
                api_key=self.config.openai_api_key,
                on_enriched_transcript=self._on_enriched_transcript
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
        
        # Start the transcript enricher
        if self.transcript_enricher:
            print(f"🎭 Starting GPT-4o Transcript Enricher (Speaker Diarization)...")
            self.transcript_enricher.start()
        
        if not self.config.is_api_key_configured():
            self.gui.update_status("ERROR: GEMINI_API_KEY not configured", "red")
            self.gui.add_error("GEMINI_API_KEY not configured.")
        
        self.websocket_server.start()
        
        try:
            self.gui.run()
        finally:
            self.stop()

    def stop(self):
        """Unified shutdown logic to stop all threads and clean up resources."""
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        
        print("\n" + "="*50)
        print("🛑 VISION APP SHUTDOWN INITIATED")
        print("="*50)
        
        # 1. Stop polling loops first
        print("  [1/7] Stopping polling loops...")
        self.mic_polling_active = False
        
        # 2. Stop streaming manager (stops sending frames to Gemini)
        print("  [2/7] Stopping Gemini streaming...")
        try:
            self.streaming_manager.stop_streaming()
        except Exception as e:
            print(f"    ⚠️ Streaming manager error: {e}")
        
        # 3. Stop microphone transcriber (Parakeet)
        print("  [3/7] Stopping Parakeet microphone transcriber...")
        try:
            if hasattr(self, 'mic_transcriber'):
                self.mic_transcriber.stop_event.set()
                time.sleep(0.2)  # Give it a moment to stop
        except Exception as e:
            print(f"    ⚠️ Mic transcriber error: {e}")
            
        # 4. Stop OpenAI Realtime / Smart Transcriber (Whisper)
        print("  [4/7] Stopping OpenAI Whisper transcriber...")
        try:
            if self.smart_transcriber:
                self.smart_transcriber.stop()
        except Exception as e:
            print(f"    ⚠️ Smart transcriber error: {e}")
        
        # 5. Stop Transcript Enricher
        print("  [5/7] Stopping GPT-4o Transcript Enricher...")
        try:
            if self.transcript_enricher:
                self.transcript_enricher.stop()
        except Exception as e:
            print(f"    ⚠️ Transcript enricher error: {e}")
            
        # 6. Stop WebSocket server
        print("  [6/7] Stopping WebSocket server...")
        try:
            self.websocket_server.stop()
        except Exception as e:
            print(f"    ⚠️ WebSocket server error: {e}")
        
        # 7. Release screen capture resources
        print("  [7/7] Releasing screen capture...")
        try:
            if hasattr(self, 'screen_capture'):
                self.screen_capture.release()
        except Exception as e:
            print(f"    ⚠️ Screen capture error: {e}")
        
        # Kill GUI if still alive
        print("  Closing GUI...")
        try:
            if self.gui and self.gui.root.winfo_exists():
                self.gui.root.quit()
                self.gui.root.destroy()
        except Exception as e:
            pass  # GUI might already be gone
        
        print("="*50)
        print("✅ VISION APP SHUTDOWN COMPLETE")
        print("="*50 + "\n")

    def _poll_mic_transcripts(self):
        print("🎤 Microphone transcript polling started...")
        while self.mic_polling_active:
            try:
                text, filename, source, confidence = self.mic_transcriber.result_queue.get(timeout=0.1)
                
                if text and len(text.strip()) > 0:
                    print(f"🎙️ [Mic/User]: {text}")
                    self.streaming_manager.add_transcript(f"[USER]: {text}")
                    
                    # Microphone is always the user - no enrichment needed
                    self.websocket_server.broadcast({
                        "type": "transcript",
                        "source": "microphone",
                        "speaker": "User",
                        "text": text,
                        "enriched": False,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })
            except Empty:
                continue
            except Exception as e:
                if self.mic_polling_active:  # Only log if we're not shutting down
                    print(f"❌ Mic polling error: {e}")
                time.sleep(0.1)
        print("🎤 Microphone transcript polling stopped.")

    def _handle_whisper_transcript(self, transcript):
        """
        Handle desktop audio transcripts from OpenAI Whisper.
        Send to enricher for speaker identification via GPT-4o.
        """
        print(f"🔊 [Desktop/Raw]: {transcript}")
        
        # Add raw transcript to streaming manager for Gemini context
        self.streaming_manager.add_transcript(f"[AUDIO]: {transcript}")
        
        # Send to enricher for speaker diarization
        if self.transcript_enricher:
            self.transcript_enricher.enrich(transcript)
        else:
            # Fallback: broadcast raw transcript if enricher not available
            self.websocket_server.broadcast({
                "type": "transcript",
                "source": "desktop",
                "speaker": "Unknown",
                "text": transcript,
                "enriched": False,
                "timestamp": time.time()
            })

    def _on_enriched_transcript(self, enriched_text):
        """
        Callback when GPT-4o returns an enriched transcript with speaker labels.
        Example: "[0:45] Charlie: (singing hopefully) "Inside of every demon is a rainbow!""
        """
        print(f"🎭 [Enriched]: {enriched_text}")
        
        # Parse the enriched text to extract speaker (basic parsing)
        speaker = "Unknown"
        try:
            # Format is typically: [timestamp] Speaker: (tone) "text"
            # Try to extract speaker name
            import re
            match = re.search(r'\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:', enriched_text)
            if match:
                speaker = match.group(1).strip()
        except:
            pass
        
        # Broadcast enriched transcript via WebSocket
        self.websocket_server.broadcast({
            "type": "transcript",
            "source": "desktop",
            "speaker": speaker,
            "text": enriched_text,
            "enriched": True,
            "timestamp": time.time()
        })

    def _on_openai_error(self, error_msg):
        print(f"❌ OpenAI Error: {error_msg}")
        if hasattr(self, 'gui') and self.gui:
            self.gui.add_error(f"OpenAI Error: {error_msg}")

    def _on_gemini_response(self, text_chunk):
        self.current_response_buffer += text_chunk
        if self.current_response_buffer.strip().endswith(('.', '!', '?', '"', '\n')):
            final_text = self.current_response_buffer.strip()
            
            # Store the visual context for the enricher
            self.last_gemini_context = final_text
            if self.transcript_enricher:
                self.transcript_enricher.update_visual_context(final_text)
            
            if hasattr(self, 'gui') and self.gui:
                self.gui.add_response(final_text)
            print(f"👁️ [Gemini Vision]: {final_text}")
            
            self.websocket_server.broadcast({
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            })
            self.current_response_buffer = ""

    def _on_gemini_error(self, error_message):
        if hasattr(self, 'gui') and self.gui:
            self.gui.add_error(f"Gemini API Error: {error_message}")

    def _on_streaming_error(self, error_message):
        if hasattr(self, 'gui') and self.gui:
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