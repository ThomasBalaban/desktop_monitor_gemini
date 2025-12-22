# thomasbalaban/desktop_monitor_gemini/transcriber_core/desktop_transcriber.py

import os
import time
import re
import sys
import uuid
from threading import Thread, Event, Lock
from queue import Queue
import sounddevice as sd  # type: ignore
import numpy as np  # type: ignore
import parakeet_mlx  # type: ignore
import mlx.core as mx  # type: ignore
from .desktop_speech_music_classifier import SpeechMusicClassifier
from .config import FS, SAVE_DIR, DESKTOP_DEVICE_ID

# Define stop_event at the module level
stop_event = Event()

class SpeechMusicTranscriber:
    def __init__(self, keep_files=False, auto_detect=True, transcript_manager=None):
        self.FS = FS
        self.SAVE_DIR = SAVE_DIR
        self.DESKTOP_DEVICE_ID = DESKTOP_DEVICE_ID 
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print(f"🎙️ Initializing Parakeet for Desktop Audio (Streaming Mode)...")
            
        try:
            # Fixing memory issues by ensuring the model is loaded properly
            self.model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
            print("✅ Parakeet model for Desktop is ready.")
        except Exception as e:
            print(f"❌ Error loading Parakeet model: {e}")
            raise

        self.result_queue = Queue()
        self.stop_event = stop_event
        self.saved_files = []
        self.keep_files = keep_files
        self.active_threads = 0
        self.last_processed = time.time()

        self.classifier = SpeechMusicClassifier()
        self.auto_detect = auto_detect
        self.transcript_manager = transcript_manager
        
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = Lock()
        self.transcriber_stream = None
        
        # Singular item tracking
        self.current_session_id = str(uuid.uuid4())
        
        # --- TUNING PARAMETERS ---
        self.VOLUME_BOOST = 5.0       
        self.SILENCE_THRESHOLD = 0.005 
        # -------------------------

        self.name_variations = {
            r'\bnaomi\b': 'Nami', r'\bnow may\b': 'Nami', r'\bnomi\b': 'Nami', 
            r'\bnamy\b': 'Nami', r'\bnot me\b': 'Nami', r'\bnah me\b': 'Nami', 
            r'\bnonny\b': 'Nami', r'\bnonni\b': 'Nami', r'\bmamie\b': 'Nami', 
            r'\bgnomey\b': 'Nami', r'\barmy\b': 'Nami', r'\bpeepingnaomi\b': 'PeepingNami', 
            r'\bpeepingnomi\b': 'PeepingNami'
        }

    def _apply_name_correction(self, text):
        corrected_text = text
        for variation, name in self.name_variations.items():
            corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)
        return corrected_text

    def audio_callback(self, indata, frames, timestamp, status):
        """Buffers audio from the input stream."""
        if status:
            pass 
        if self.stop_event.is_set():
            return

        if len(indata.shape) > 1 and indata.shape[1] > 1:
            new_audio = np.mean(indata, axis=1).astype(np.float32)
        else:
            new_audio = indata.flatten().astype(np.float32)
        
        with self.buffer_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])

    def streaming_worker(self):
        """Dedicated thread to feed audio chunks to the Parakeet stream."""
        CHUNK_SIZE = int(self.FS * 0.8) 
        
        # Initialize variables before the loop to avoid UnboundLocalError
        last_text = ""
        
        try:
            with self.model.transcribe_stream() as self.transcriber_stream:
                print("🎧 Parakeet Streaming Worker started.")
                
                while not self.stop_event.is_set():
                    audio_to_process = None
                    
                    with self.buffer_lock:
                        if len(self.audio_buffer) >= CHUNK_SIZE:
                            audio_to_process = self.audio_buffer[:CHUNK_SIZE].copy()
                            self.audio_buffer = self.audio_buffer[CHUNK_SIZE:]
                    
                    if audio_to_process is not None:
                        peak_amp = np.max(np.abs(audio_to_process))
                        
                        # --- FIX FOR DOUBLE FREE ---
                        # Explicitly evaluate the array before passing to the stream
                        input_mx = mx.array(audio_to_process * self.VOLUME_BOOST)
                        mx.eval(input_mx)
                        self.transcriber_stream.add_audio(input_mx)
                        
                        # Process results if we have actual sound
                        if peak_amp > self.SILENCE_THRESHOLD:
                            current_result = self.transcriber_stream.result
                            
                            if current_result and hasattr(current_result, 'text'):
                                full_text = current_result.text.strip()
                                
                                # Only emit if text has grown or changed
                                if full_text != last_text:
                                    # Visual console feedback
                                    display_text = full_text[-80:].replace('\n', ' ')
                                    sys.stdout.write(f"\r🦜 {display_text}")
                                    sys.stdout.flush()

                                    corrected = self._apply_name_correction(full_text)
                                    # Passing session_id as the 'filename' parameter so transcription_service can find it
                                    payload = (corrected, self.current_session_id, "desktop_partial", 0.9)
                                    self.result_queue.put(payload)
                                    
                                    last_text = full_text
                                    self.last_processed = time.time()
                        else:
                            # If silence has lasted long enough, reset the session to start a new "singular item"
                            if time.time() - self.last_processed > 3.0 and last_text != "":
                                self.current_session_id = str(uuid.uuid4())
                                last_text = ""
                                print("\n🍃 [Desktop] Silence detected, session reset.")
                        
                    time.sleep(0.05)
                    
        except Exception as e:
            print(f"[Desktop-Parakeet-ERROR] Streaming Worker failed: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Starts the audio stream and the Parakeet streaming worker thread."""
        Thread(target=self.streaming_worker, daemon=True, name="ParakeetDesktopWorker").start()
        
        try:
            device_info = sd.query_devices(self.DESKTOP_DEVICE_ID)
            channels = min(device_info['max_input_channels'], 2)
            
            print(f"🎧 Opening audio stream for Parakeet:")
            print(f"   Device ID: {self.DESKTOP_DEVICE_ID}, Channels: {channels}, Target Rate: {self.FS} Hz")
            
            stream_kwargs = {
                'device': self.DESKTOP_DEVICE_ID,
                'samplerate': self.FS,
                'channels': channels,
                'callback': self.audio_callback,
                'blocksize': self.FS // 10,
                'dtype': 'float32'
            }
            
            with sd.InputStream(**stream_kwargs):
                print(f"🎧 Listening to desktop audio (device {self.DESKTOP_DEVICE_ID}) with live streaming...")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        except Exception as e:
            print(f"\n❌ Error starting desktop audio stream: {e}")
        finally:
            self.stop_event.set()
            print("\nShutting down desktop transcriber...")
            print("🖥️ Desktop transcription stopped.")