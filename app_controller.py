import tkinter as tk
from datetime import datetime
import threading
import time
import uuid
import sounddevice as sd
from queue import Empty

from config_loader import ConfigLoader
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from app_gui import AppGUI
from websocket_server import WebSocketServer, WEBSOCKET_PORT
from transcriber_core.microphone import MicrophoneTranscriber
from config import MICROPHONE_DEVICE_ID, DESKTOP_AUDIO_DEVICE_ID

from openai_realtime_client import OpenAIRealtimeClient
from transcriber_core.openai_streamer import SmartAudioTranscriber
from transcript_enricher import TranscriptEnricher

class AppController:
    def __init__(self):
        self.config = ConfigLoader()
        print("Gemini Screen Watcher (Unified Vision+Audio) - Starting up...")
        
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        
        # Audio Device State
        self.current_mic_id = MICROPHONE_DEVICE_ID
        self.current_desktop_id = DESKTOP_AUDIO_DEVICE_ID
        
        # ---------------------------------------------------------
        # SETTING: BROADCAST RAW STREAM
        # True  = Send "Raw" text immediately, then "Enriched" text later (good for replacing text in UI).
        # False = Wait for "Enriched" text only (good for logs/simple clients).
        self.BROADCAST_RAW = False 
        # ---------------------------------------------------------
        
        self.screen_capture = ScreenCapture(
            self.config.image_quality, 
            video_index=self.config.video_device_index
        )
        
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
        
        self.mic_transcriber = MicrophoneTranscriber(keep_files=False, device_id=self.current_mic_id)
        self.mic_transcriber.set_volume_callback(lambda level: self.gui.set_volume_meter("mic", level))
        self.mic_polling_active = True 
        
        self.streaming_manager = StreamingManager(
            self.screen_capture, self.gemini_client, self.config.fps,
            restart_interval=1500, debug_mode=self.config.debug_mode
        )
        self.streaming_manager.set_restart_callback(self.on_stream_restart)
        self.streaming_manager.set_error_callback(self._on_streaming_error)
        self.streaming_manager.set_preview_callback(self.gui_update_wrapper)
        
        self.websocket_server = WebSocketServer()
        self.current_response_buffer = ""
        
        self.gui = AppGUI(self)
        
        self.smart_transcriber = None
        self.transcript_enricher = None
        self.last_gemini_context = "" 
        
        if not self.config.is_openai_key_configured():
            print("⚠️ WARNING: OPENAI_API_KEY missing.")
            self.gui.add_error("OPENAI_API_KEY missing.")
            self.openai_client = None
        else:
            self.openai_client = OpenAIRealtimeClient(
                api_key=self.config.openai_api_key,
                on_transcript=self._handle_whisper_transcript,
                on_error=self._on_openai_error
            )
            self.smart_transcriber = SmartAudioTranscriber(
                self.openai_client, 
                device_id=self.current_desktop_id
            )
            self.smart_transcriber.set_volume_callback(lambda level: self.gui.set_volume_meter("desktop", level))
            
            self.transcript_enricher = TranscriptEnricher(
                api_key=self.config.openai_api_key,
                on_enriched_transcript=self._on_enriched_transcript
            )

        if self.config.video_device_index is None:
            self._initialize_capture_region()
            
        self.refresh_audio_devices()
        self.gui.root.after(2000, self._start_stream_on_init)

    def refresh_audio_devices(self):
        try:
            devices = sd.query_devices()
            mic_list = []
            desktop_list = []
            
            for i, dev in enumerate(devices):
                name = f"{i}: {dev['name']}"
                inputs = dev['max_input_channels']
                if inputs > 0:
                    mic_list.append(name)
                    desktop_list.append(name)
            
            self.gui.set_device_lists(mic_list, desktop_list, self.current_mic_id, self.current_desktop_id)
        except Exception as e:
            print(f"Error querying devices: {e}")

    def on_mic_changed(self, event):
        selection = self.gui.combo_mic.get()
        if not selection: return
        try:
            new_id = int(selection.split(":")[0])
            if new_id == self.current_mic_id: return
            self.current_mic_id = new_id
            self._restart_mic_transcriber()
        except: pass

    def on_desktop_changed(self, event):
        selection = self.gui.combo_desktop.get()
        if not selection: return
        try:
            new_id = int(selection.split(":")[0])
            if new_id == self.current_desktop_id: return
            self.current_desktop_id = new_id
            self._restart_desktop_transcriber()
        except: pass

    def _restart_mic_transcriber(self):
        if self.mic_transcriber: self.mic_transcriber.stop()
        self.mic_transcriber = MicrophoneTranscriber(keep_files=False, device_id=self.current_mic_id)
        self.mic_transcriber.set_volume_callback(lambda level: self.gui.set_volume_meter("mic", level))
        threading.Thread(target=self.mic_transcriber.run, daemon=True).start()

    def _restart_desktop_transcriber(self):
        if not self.smart_transcriber: return
        self.smart_transcriber.stop()
        self.smart_transcriber = SmartAudioTranscriber(self.openai_client, device_id=self.current_desktop_id)
        self.smart_transcriber.set_volume_callback(lambda level: self.gui.set_volume_meter("desktop", level))
        self.smart_transcriber.start()

    def gui_update_wrapper(self, frame):
        if self.gui: self.gui.update_preview(frame)

    def run(self):
        threading.Thread(target=self.mic_transcriber.run, daemon=True).start()
        threading.Thread(target=self._poll_mic_transcripts, daemon=True).start()

        if self.smart_transcriber: self.smart_transcriber.start()
        if self.transcript_enricher: self.transcript_enricher.start()
        
        self.websocket_server.start()
        try: self.gui.run()
        finally: self.stop()
            
    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down: return
            self._shutting_down = True
        
        self.mic_polling_active = False
        try: self.streaming_manager.stop_streaming()
        except: pass
        try: self.mic_transcriber.stop()
        except: pass
        try: 
            if self.smart_transcriber: self.smart_transcriber.stop()
        except: pass
        try: 
            if self.transcript_enricher: self.transcript_enricher.stop()
        except: pass
        try: self.websocket_server.stop()
        except: pass
        try: self.screen_capture.release()
        except: pass
        try:
            if self.gui and self.gui.root.winfo_exists():
                self.gui.root.quit()
                self.gui.root.destroy()
        except: pass

    def _poll_mic_transcripts(self):
        while self.mic_polling_active:
            try:
                text, filename, source, confidence = self.mic_transcriber.result_queue.get(timeout=0.1)
                if text and len(text.strip()) > 0:
                    self.streaming_manager.add_transcript(f"[USER]: {text}")
                    self.websocket_server.broadcast({
                        "type": "transcript",
                        "source": "microphone",
                        "speaker": "User",
                        "text": text,
                        "enriched": False,
                        "confidence": confidence,
                        "timestamp": time.time(),
                        "id": str(uuid.uuid4())
                    })
            except Empty: continue
            except Exception: time.sleep(0.1)

    def _handle_whisper_transcript(self, transcript):
        self.streaming_manager.add_transcript(f"[AUDIO]: {transcript}")
        
        event_id = str(uuid.uuid4())
        
        # 1. IMMEDIATE BROADCAST (Raw)
        # SKIPPED if BROADCAST_RAW is False
        if self.BROADCAST_RAW:
            self.websocket_server.broadcast({
                "type": "transcript",
                "source": "desktop",
                "speaker": "Unknown",
                "text": transcript,
                "enriched": False,
                "timestamp": time.time(),
                "status": "raw",
                "id": event_id 
            })

        # 2. ENRICH IN BACKGROUND (Pass ID)
        if self.transcript_enricher:
            self.transcript_enricher.enrich(transcript, transcript_id=event_id)

    def _on_enriched_transcript(self, enriched_text, transcript_id=None):
        speaker = "Unknown"
        try:
            import re
            match = re.search(r'\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:', enriched_text)
            if match: speaker = match.group(1).strip()
        except: pass
        
        # Log to console
        print(f"{enriched_text}")

        # Broadcast to websocket
        self.websocket_server.broadcast({
            "type": "transcript",
            "source": "desktop",
            "speaker": speaker,
            "text": enriched_text,
            "enriched": True,
            "timestamp": time.time(),
            "id": transcript_id 
        })

    def _on_openai_error(self, error_msg):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"OpenAI Error: {error_msg}")

    def _on_gemini_response(self, text_chunk):
        self.current_response_buffer += text_chunk
        if self.current_response_buffer.strip().endswith(('.', '!', '?', '"', '\n')):
            final_text = self.current_response_buffer.strip()
            self.last_gemini_context = final_text
            if self.transcript_enricher:
                self.transcript_enricher.update_visual_context(final_text)
            if hasattr(self, 'gui') and self.gui:
                self.gui.add_response(final_text)
            self.websocket_server.broadcast({
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            })
            self.current_response_buffer = ""

    def _on_gemini_error(self, error_message):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"Gemini API Error: {error_message}")

    def _on_streaming_error(self, error_message):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"Streaming Error: {error_message}")

    def _start_stream_on_init(self):
        if not self.screen_capture.is_ready():
            self.gui.update_status("Cannot start. No source configured.", "red")
            return
        
        def run_check_and_start():
            api_ok, message = self.gemini_client.test_connection()
            self.gui.root.after(0, self._finalize_start, api_ok, message)
        
        self.gui.update_status("Checking API connection...", "orange")
        threading.Thread(target=run_check_and_start, daemon=True).start()

    def _finalize_start(self, api_ok, message):
        if not api_ok:
            self.gui.add_error(f"API Connection Check Failed: {message}")
            self.gui.update_status("API Check Failed", "red")
            return
        self.streaming_manager.set_status_callback(self.gui.update_status)
        self.streaming_manager.start_streaming()
        self.gui.update_status("Streaming", "#4CAF50")

    def update_websocket_gui_status(self):
        self.gui.update_websocket_status(f"Running at ws://localhost:{WEBSOCKET_PORT}", "#4CAF50")

    def on_stream_restart(self):
        self.gui.add_reset_separator()

    def _initialize_capture_region(self):
        if self.config.capture_region:
            self.screen_capture.set_capture_region(self.config.capture_region)

    def get_prompt(self): return self.config.prompt
    def get_timestamp(self): return datetime.now().strftime("%I:%M:%S %p")