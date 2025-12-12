import asyncio
import time
import threading
from enum import Enum
from audio_capture import AudioCapture
from config import DESKTOP_AUDIO_DEVICE_ID, AUDIO_SAMPLE_RATE

class StreamingState(Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RESTARTING = "restarting"
    ERROR = "error"

class StreamingManager:
    def __init__(self, screen_capture, gemini_client, fps=2, restart_interval=1500, debug_mode=False):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.fps = fps
        self.debug_mode = debug_mode
        self.state = StreamingState.STOPPED
        
        # Audio Setup
        self.audio_capture = AudioCapture(DESKTOP_AUDIO_DEVICE_ID, AUDIO_SAMPLE_RATE)
        
        # Session Management
        self.restart_interval = restart_interval # 1500s = 25 minutes
        self.session_start_time = None
        self.current_loop = None
        
        # Pulse Logic (The Anti-Spam Fix)
        self.last_pulse_time = 0
        self.pulse_interval = 10.0 # Only speak every 10s if nothing is happening
        self.silence_threshold = 0.01

        # Manual Trigger Logic
        self.manual_trigger_text = None
        self.trigger_lock = threading.Lock()
        
        # Transcript Injection (New)
        self.transcript_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Tasks
        self.streaming_task = None
        self.listener_task = None
        
        # Callbacks
        self.status_callback = None
        self.restart_callback = None
        self.error_callback = None
        self.preview_callback = None 

    def set_status_callback(self, callback): self.status_callback = callback
    def set_restart_callback(self, callback): self.restart_callback = callback
    def set_error_callback(self, callback): self.error_callback = callback
    def set_preview_callback(self, callback): self.preview_callback = callback

    def _update_status(self, status, color="black"):
        if self.status_callback:
            self.status_callback(status, color)

    def _report_error(self, message):
        if self.error_callback:
            self.error_callback(message)

    def trigger_manual_analysis(self, text):
        """Queues a text prompt to be sent with the NEXT frame."""
        with self.trigger_lock:
            self.manual_trigger_text = text
        self.info_print(f"Manual analysis queued: {text}")

    def inject_transcript(self, text, source):
        """Called by AppController when local STT hears something."""
        with self.buffer_lock:
            # Format: "[Microphone]: Hello Nami"
            self.transcript_buffer.append(f"[{source.capitalize()}]: {text}")

    def start_streaming(self):
        if self.state != StreamingState.STOPPED:
            return False
        self.state = StreamingState.CONNECTING
        self.session_start_time = time.time()
        
        # Start Ears
        self.audio_capture.start()
        
        def run_streaming():
            self.current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.current_loop)
            try:
                self.current_loop.run_until_complete(self._run_streaming_session())
            except Exception as e:
                self.info_print(f"Streaming thread error: {e}")
                self.state = StreamingState.ERROR
                self._update_status("Error occurred", "red")
                self._report_error(f"Streaming thread error: {e}")
            finally:
                if self.current_loop and not self.current_loop.is_closed():
                    self.current_loop.close()
                self.current_loop = None
                if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                    self.state = StreamingState.STOPPED
        
        threading.Thread(target=run_streaming, daemon=True).start()
        return True

    def stop_streaming(self):
        self.info_print("Stopping streaming...")
        self.state = StreamingState.STOPPED
        self.session_start_time = None
        self.audio_capture.stop()
        self._cleanup_async_tasks()
        self._update_status("Stopped", "red")
        return True

    def restart_streaming_session(self):
        if self.state == StreamingState.RESTARTING or self.state == StreamingState.STOPPED:
            return
        self.info_print("Restarting streaming session (Context Refresh)...")
        self.state = StreamingState.RESTARTING
        if self.restart_callback:
            self.restart_callback()
            
        self._cleanup_async_tasks()
        time.sleep(1.5)
        
        if self.state != StreamingState.RESTARTING:
            return
            
        self.state = StreamingState.STOPPED
        if not self.start_streaming():
            self.info_print("Failed to restart streaming session")
            self.state = StreamingState.ERROR
            self._update_status("Restart failed", "red")

    def _cleanup_async_tasks(self):
        if self.current_loop and not self.current_loop.is_closed():
            tasks = [self.streaming_task, self.listener_task]
            for task in tasks:
                if task and not task.done():
                    self.current_loop.call_soon_threadsafe(task.cancel)
            
            future = asyncio.run_coroutine_threadsafe(self.gemini_client.disconnect(), self.current_loop)
            try:
                future.result(timeout=3.0)
            except Exception:
                pass

    async def _run_streaming_session(self):
        try:
            self._update_status("Connecting...", "orange")
            if await self.gemini_client.connect():
                if self.state == StreamingState.RESTARTING: return
                
                self.state = StreamingState.STREAMING
                self._update_status("Watching (Silent Mode)", "green")
                
                await asyncio.sleep(1.0) 
                
                self.listener_task = asyncio.create_task(self.gemini_client.listen_for_responses())
                self.streaming_task = asyncio.create_task(self._streaming_loop())
                
                await asyncio.gather(self.streaming_task, self.listener_task, return_exceptions=True)
            else:
                self.state = StreamingState.ERROR
                self._update_status("Connection failed", "red")
                self._report_error("Failed to connect to Gemini")
        except Exception as e:
            self.info_print(f"Streaming session error: {e}")
            self.state = StreamingState.ERROR
            self._update_status("Error occurred", "red")
        finally:
            await self.gemini_client.disconnect()
            if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                self.state = StreamingState.STOPPED
                self._update_status("Stopped", "red")

    async def _streaming_loop(self):
        frame_interval = 1.0 / self.fps
        while self.state == StreamingState.STREAMING:
            start_time = time.time()
            try:
                # 1. Capture & Send Screen (Always)
                frame = self.screen_capture.capture_frame()
                base64_image = None
                if frame:
                    if self.preview_callback: 
                        self.preview_callback(frame)
                    base64_image = self.screen_capture.image_to_base64(frame)
                
                # 2. Capture Audio
                audio_bytes, is_loud = self.audio_capture.get_recent_audio()
                
                # 3. Logic: Should we trigger a response (Turn Complete)?
                turn_complete = False 
                manual_text = None
                
                # Check Manual Trigger First
                with self.trigger_lock:
                    if self.manual_trigger_text:
                        manual_text = self.manual_trigger_text
                        turn_complete = True
                        self.manual_trigger_text = None # consume it
                        self.last_pulse_time = time.time() # Reset timer
                
                # --- NEW: Check for Local Transcripts ---
                injected_text = None
                with self.buffer_lock:
                    if self.transcript_buffer:
                        # Combine all recent utterances into one context block
                        combined_speech = " ".join(self.transcript_buffer)
                        injected_text = f"CONTEXT UPDATE (What you just heard locally):\n{combined_speech}"
                        self.transcript_buffer.clear() # Clear after reading
                        
                        # If we have speech, we force a turn complete so Gemini answers immediately
                        turn_complete = True 
                        self.last_pulse_time = time.time()
                        if self.debug_mode: print(f"Trigger: Local Speech - {combined_speech}")

                # Combine Manual and Injected text if both exist
                final_text_payload = manual_text 
                if injected_text:
                    if final_text_payload:
                        final_text_payload += f"\n\n{injected_text}"
                    else:
                        final_text_payload = injected_text

                # Check Audio/Heartbeat triggers (only if no explicit text trigger)
                if not turn_complete:
                    current_time = time.time()
                    time_since_pulse = current_time - self.last_pulse_time

                    if is_loud:
                        turn_complete = True
                        self.last_pulse_time = current_time
                        if self.debug_mode: print("Trigger: Audio Detected")
                        
                    elif time_since_pulse > self.pulse_interval:
                        turn_complete = True
                        self.last_pulse_time = current_time
                        if self.debug_mode: print("Trigger: Heartbeat Pulse")

                # 4. Send Data using the Hybrid Method
                await self.gemini_client.send_multimodal_frame(
                    base64_image, 
                    audio_bytes, 
                    turn_complete, 
                    text=final_text_payload
                )
                
                # 5. Session Time Limit Check
                if self.session_start_time and (time.time() - self.session_start_time) >= self.restart_interval:
                    self.info_print("Session time limit reached. Initiating restart.")
                    threading.Thread(target=self.restart_streaming_session, daemon=True).start()
                    break

                # Sleep to maintain FPS
                elapsed = time.time() - start_time
                delay = max(0, frame_interval - elapsed)
                await asyncio.sleep(delay)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.info_print(f"Error in streaming loop: {e}")
                self._report_error(f"Streaming loop error: {e}")
                self.state = StreamingState.ERROR
                break

    def debug_print(self, message):
        if self.debug_mode:
            print(f"[DEBUG] {message}")

    def info_print(self, message):
        print(message)