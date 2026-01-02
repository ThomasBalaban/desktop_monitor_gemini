import threading
import time
import base64
import cv2
import traceback
from datetime import datetime

class StreamingManager:
    def __init__(self, screen_capture, gemini_client, target_fps=1.0, restart_interval=1500, debug_mode=False):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.target_fps = target_fps
        self.restart_interval = restart_interval  # Restart stream every N frames (if applicable)
        self.debug_mode = debug_mode

        self.streaming_active = False
        self.frame_count = 0
        self.stop_event = threading.Event()
        self.stream_thread = None
        
        # Callbacks
        self.status_callback = None
        self.error_callback = None
        self.restart_callback = None
        self.preview_callback = None  # New: For GUI preview updates
        
        # Audio Context Buffer
        self.transcript_buffer = []  # Stores recent transcripts to send with next frame
        self.buffer_lock = threading.Lock()

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_error_callback(self, callback):
        self.error_callback = callback

    def set_restart_callback(self, callback):
        self.restart_callback = callback
        
    def set_preview_callback(self, callback):
        """Callback to update the GUI preview image."""
        self.preview_callback = callback

    def start_streaming(self):
        if self.streaming_active:
            return

        print("StreamingManager: Starting stream...")
        self.streaming_active = True
        self.stop_event.clear()
        self.frame_count = 0
        
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()

    def stop_streaming(self):
        if not self.streaming_active:
            return

        print("StreamingManager: Stopping stream...")
        self.streaming_active = False
        self.stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None

    def add_transcript(self, text):
        """
        Received from AppController (Mic or Desktop audio).
        Buffers the text to be sent alongside the next video frame.
        """
        with self.buffer_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Format: [10:05:00] [USER]: Hello world
            entry = f"[{timestamp}] {text}"
            self.transcript_buffer.append(entry)
            if self.debug_mode:
                print(f"StreamingManager buffered: {entry}")

    def trigger_manual_analysis(self, prompt_override=None):
        """
        Manually captures a frame and sends it, regardless of the loop timer.
        Useful for 'Ask AI' buttons.
        """
        print("StreamingManager: Manual analysis triggered.")
        threading.Thread(target=self._process_single_frame, args=(prompt_override,), daemon=True).start()

    def _process_single_frame(self, prompt_override=None):
        """Helper to capture and send one frame."""
        frame = self.screen_capture.capture_frame()
        if frame:
            if self.preview_callback:
                self.preview_callback(frame)
            self._send_frame_to_gemini(frame, prompt_suffix=prompt_override)

    def _stream_loop(self):
        delay = 1.0 / self.target_fps
        
        while self.streaming_active and not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                # 1. Capture Frame
                frame = self.screen_capture.capture_frame()
                
                if frame:
                    self.frame_count += 1
                    
                    # 2. Update GUI Preview
                    if self.preview_callback:
                        self.preview_callback(frame)

                    # 3. Send to Gemini
                    self._send_frame_to_gemini(frame)
                    
                    # 4. Periodic Restart Logic (to manage context window or token limits if needed)
                    if self.restart_interval and self.frame_count % self.restart_interval == 0:
                        if self.restart_callback:
                            self.restart_callback()
                        # Optional: Reset session or clear history if needed
                        pass
                        
                else:
                    if self.error_callback:
                        self.error_callback("Failed to capture frame")
            
            except Exception as e:
                traceback.print_exc()
                if self.error_callback:
                    self.error_callback(f"Stream Loop Error: {e}")

            # Maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            time.sleep(sleep_time)

    def _send_frame_to_gemini(self, frame_data, prompt_suffix=None):
        """
        Encodes the frame and sends it along with any buffered audio transcripts.
        """
        try:
            # 1. Get Buffered Transcripts
            current_context = ""
            with self.buffer_lock:
                if self.transcript_buffer:
                    # Join all recent logs
                    joined_logs = "\n".join(self.transcript_buffer)
                    current_context = f"\n\nRECENT AUDIO LOGS:\n{joined_logs}\n"
                    # Clear buffer after consuming
                    self.transcript_buffer.clear()
            
            # 2. Construct the message
            # If we have a manual prompt override, use it. Otherwise rely on system prompt + context.
            text_part = current_context
            if prompt_suffix:
                text_part += f"\nUser Instruction: {prompt_suffix}"
            
            # If there is no audio context and no specific instruction, 
            # we send just the image (Gemini uses the system prompt configured in client).
            # However, the client.send_message usually expects (image, text) or just image.
            
            # Note: We pass the text_part to the client. The client handles how to format it for the API.
            self.gemini_client.send_message(frame_data, text_prompt=text_part if text_part.strip() else None)

        except Exception as e:
            print(f"StreamingManager Send Error: {e}")
            if self.error_callback:
                self.error_callback(str(e))