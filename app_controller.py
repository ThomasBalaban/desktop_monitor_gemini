import tkinter as tk
from datetime import datetime
import threading
import time
import uuid
import sounddevice as sd
import socketio 
import asyncio 
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
        
        # --- HUB CLIENT SETUP ---
        # Initializes Socket.IO client to connect to Central Hub on port 8002
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_url = "http://localhost:8002" 
        
        self.BROADCAST_RAW = False
        
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

    async def _hub_connection_loop(self):
        """Maintains the async connection to the Central Hub."""
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    await self.sio.connect(self.hub_url)
                    print(f"✅ [Monitor] Hub connected: {self.hub_url}")
                except Exception as e:
                    print(f"⚠️ [Monitor] Hub connection failed: {e}. Retrying...")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event, data):
        """Thread-safe helper to send data to Hub."""
        if self.sio.connected:
            try:
                # Dispatches the emit to the running background event loop
                asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            except Exception as e:
                print(f"❌ [Monitor] Hub emit error: {e}")

    def run(self):
        """Main execution entry point."""
        # Start Hub Client event loop in a background thread
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(target=self._start_hub_loop, args=(self.hub_loop,), daemon=True).start()

        threading.Thread(target=self.mic_transcriber.run, daemon=True).start()
        threading.Thread(target=self._poll_mic_transcripts, daemon=True).start()

        if self.smart_transcriber: self.smart_transcriber.start()
        if self.transcript_enricher: self.transcript_enricher.start()
        
        self.websocket_server.start()
        try: 
            self.gui.run()
        finally: 
            self.stop()

    def _start_hub_loop(self, loop):
        """Initializes and runs the Hub client's asyncio loop."""
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    def _on_gemini_response(self, text_chunk):
        """Handles visual analysis chunks from Gemini."""
        self.current_response_buffer += text_chunk
        if self.current_response_buffer.strip().endswith(('.', '!', '?', '"', '\n')):
            final_text = self.current_response_buffer.strip()
            self.last_gemini_context = final_text
            
            if self.transcript_enricher:
                self.transcript_enricher.update_visual_context(final_text)

            # --- HUB BROADCAST ---
            # VisionContext model expects 'context' key for the UI log
            self._emit_to_hub('vision_context', {"context": final_text})
            # text_update ensures the live feed in the UI is updated
            self._emit_to_hub('text_update', {"type": "text_update", "content": final_text})

            if hasattr(self, 'gui') and self.gui:
                self.gui.add_response(final_text)
            
            # Local legacy broadcast
            self.websocket_server.broadcast({
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            })
            self.current_response_buffer = ""

    def _poll_mic_transcripts(self):
        """Polls transcribed text from the microphone queue."""
        while self.mic_polling_active:
            try:
                text, filename, source, confidence = self.mic_transcriber.result_queue.get(timeout=0.1)
                if text and len(text.strip()) > 0:
                    # --- HUB BROADCAST ---
                    # spoken_word_context populates the 'Spoken' column in the UI
                    self._emit_to_hub('spoken_word_context', {"context": text})
                    
                    # audio_context is routed to the Director Engine and general logs
                    self._emit_to_hub('audio_context', {
                        "context": text,
                        "is_partial": False,
                        "metadata": {"source": "microphone", "confidence": confidence}
                    })

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

    def _on_enriched_transcript(self, enriched_text, transcript_id=None):
        """Handles desktop audio transcripts enriched by GPT-4o."""
        speaker = "Unknown"
        try:
            import re
            match = re.search(r'\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:', enriched_text)
            if match: speaker = match.group(1).strip()
        except: pass
        
        # --- HUB BROADCAST ---
        # populates the 'Audio' column in the UI
        self._emit_to_hub('audio_context', {
            "context": enriched_text,
            "is_partial": False,
            "metadata": {"source": "desktop", "speaker": speaker, "id": transcript_id}
        })

        self.websocket_server.broadcast({
            "type": "transcript",
            "source": "desktop",
            "speaker": speaker,
            "text": enriched_text,
            "enriched": True,
            "timestamp": time.time(),
            "id": transcript_id 
        })

    def stop(self):
        """Cleanly shuts down all monitor components."""
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
        
        # Shutdown Hub loop thread-safely
        if hasattr(self, 'hub_loop'):
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)

        try:
            if self.gui and self.gui.root.winfo_exists():
                self.gui.root.quit()
                self.gui.root.destroy()
        except: pass

    # --- GUI CALLBACKS & HELPERS ---

    def update_websocket_gui_status(self):
        """Updates the Hub connection status in the GUI footer."""
        status = "Hub Connected" if self.sio.connected else "Hub Disconnected"
        color = "#4CAF50" if self.sio.connected else "orange"
        self.gui.update_websocket_status(status, color)

    def refresh_audio_devices(self):
        """Queries and updates available audio devices."""
        try:
            devices = sd.query_devices()
            mic_list = []
            desktop_list = []
            for i, dev in enumerate(devices):
                name = f"{i}: {dev['name']}"
                if dev['max_input_channels'] > 0:
                    mic_list.append(name)
                    desktop_list.append(name)
            self.gui.set_device_lists(mic_list, desktop_list, self.current_mic_id, self.current_desktop_id)
        except Exception as e: print(f"Error refreshing devices: {e}")

    def on_mic_changed(self, event):
        """Triggered when user selects a different microphone."""
        selection = self.gui.combo_mic.get()
        if not selection: return
        try:
            nid = int(selection.split(":")[0])
            if nid == self.current_mic_id: return
            self.current_mic_id = nid
            self._restart_mic_transcriber()
        except: pass

    def on_desktop_changed(self, event):
        """Triggered when user selects a different desktop audio source."""
        selection = self.gui.combo_desktop.get()
        if not selection: return
        try:
            nid = int(selection.split(":")[0])
            if nid == self.current_desktop_id: return
            self.current_desktop_id = nid
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

    def _handle_whisper_transcript(self, transcript):
        self.streaming_manager.add_transcript(f"[AUDIO]: {transcript}")
        if self.transcript_enricher:
            self.transcript_enricher.enrich(transcript)

    def _on_openai_error(self, error_msg):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"OpenAI Error: {error_msg}")

    def _on_gemini_error(self, error_message):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"Gemini API Error: {error_message}")

    def _on_streaming_error(self, error_message):
        if hasattr(self, 'gui') and self.gui: self.gui.add_error(f"Streaming Error: {error_message}")

    def _start_stream_on_init(self):
        if not self.screen_capture.is_ready():
            self.gui.update_status("Cannot start. No source.", "red")
            return
        def run_check():
            ok, msg = self.gemini_client.test_connection()
            self.gui.root.after(0, self._finalize_start, ok, msg)
        self.gui.update_status("Checking API...", "orange")
        threading.Thread(target=run_check, daemon=True).start()

    def _finalize_start(self, ok, msg):
        if not ok:
            self.gui.add_error(f"API Failed: {msg}")
            self.gui.update_status("API Failed", "red")
            return
        self.streaming_manager.set_status_callback(self.gui.update_status)
        self.streaming_manager.start_streaming()
        self.gui.update_status("Streaming", "#4CAF50")

    def on_stream_restart(self): self.gui.add_reset_separator()

    def _initialize_capture_region(self):
        if self.config.capture_region: self.screen_capture.set_capture_region(self.config.capture_region)

    def get_prompt(self): return self.config.prompt
    def get_timestamp(self): return datetime.now().strftime("%I:%M:%S %p")