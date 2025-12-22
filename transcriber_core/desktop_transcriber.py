# thomasbalaban/desktop_monitor_gemini/transcriber_core/desktop_transcriber.py

import os
import time
import re
import sys
import uuid
from threading import Thread, Event, Lock
from queue import Queue
from difflib import SequenceMatcher
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
        
        # Session tracking
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

    def cleanup(self):
        """Explicitly cleanup Parakeet resources."""
        print("🧹 [Desktop] Cleaning up Parakeet resources...")
        
        # Clear the transcriber stream reference
        self.transcriber_stream = None
        
        # Clear the model reference
        if hasattr(self, 'model'):
            self.model = None
        
        # Force MLX to clear its memory
        try:
            mx.metal.clear_cache()
            print("✅ [Desktop] MLX metal cache cleared.")
        except Exception as e:
            print(f"⚠️ [Desktop] Could not clear MLX cache: {e}")
        
        # Clear audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)

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
        # Use 1.5 second chunks like the reference example
        CHUNK_SIZE = int(self.FS * 1.5) 
        
        # Sentence splitting pattern: . ! ? ... ....
        # Match sentences ending with these punctuation marks
        sentence_pattern = re.compile(r'[^.!?]*(?:\.{3,4}|[.!?])')
        
        # Track what we've already emitted
        emitted_sentences = []  # List of sentences we've already emitted (for fuzzy matching)
        last_full_text = ""
        last_emit_text = ""  # The exact text of the last emission (to prevent duplicates)
        
        # Fuzzy matching threshold (0.0 to 1.0)
        SIMILARITY_THRESHOLD = 0.80
        
        def is_sentence_emitted(sentence, emitted_list):
            """Check if sentence is similar enough to any already-emitted sentence."""
            s_normalized = sentence.lower().strip()
            for emitted in emitted_list:
                ratio = SequenceMatcher(None, s_normalized, emitted).ratio()
                if ratio >= SIMILARITY_THRESHOLD:
                    return True
            return False
        
        try:
            with self.model.transcribe_stream() as self.transcriber_stream:
                print("🎧 Parakeet Streaming Worker started (3-sentence buffer mode with fuzzy matching).")
                print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
                
                while not self.stop_event.is_set():
                    audio_to_process = None
                    
                    with self.buffer_lock:
                        if len(self.audio_buffer) >= CHUNK_SIZE:
                            audio_to_process = self.audio_buffer[:CHUNK_SIZE].copy()
                            remaining = self.audio_buffer[CHUNK_SIZE:]
                            self.audio_buffer = remaining if len(remaining) > 0 else np.array([], dtype=np.float32)
                    
                    if audio_to_process is not None:
                        peak_amp = np.max(np.abs(audio_to_process))
                        
                        # Add audio to transcriber (with volume boost)
                        input_mx = mx.array(audio_to_process * self.VOLUME_BOOST)
                        self.transcriber_stream.add_audio(input_mx)
                        
                        # Get result
                        result = self.transcriber_stream.result
                        
                        if result and hasattr(result, 'text'):
                            current_full_text = result.text.strip()
                            
                            # Only process if text has changed
                            if current_full_text != last_full_text:
                                last_full_text = current_full_text
                                
                                # Apply name corrections to full text
                                corrected_text = self._apply_name_correction(current_full_text)
                                
                                # Split into ALL sentences
                                all_sentences = sentence_pattern.findall(corrected_text)
                                all_sentences = [s.strip() for s in all_sentences if s.strip()]
                                
                                # Filter out sentences we've already emitted (using fuzzy matching)
                                new_sentences = []
                                for s in all_sentences:
                                    if not is_sentence_emitted(s, emitted_sentences):
                                        new_sentences.append(s)
                                
                                # Check for partial sentence at the end
                                partial_buffer = ""
                                if all_sentences:
                                    last_sentence_end = corrected_text.rfind(all_sentences[-1]) + len(all_sentences[-1])
                                    partial_buffer = corrected_text[last_sentence_end:].strip()
                                else:
                                    partial_buffer = corrected_text
                                
                                # Visual feedback
                                display_sentences = f"[{len(new_sentences)} new sentences]"
                                if new_sentences:
                                    last_sent = new_sentences[-1][-40:] if len(new_sentences[-1]) > 40 else new_sentences[-1]
                                    display_sentences += f" Last: {last_sent}"
                                if partial_buffer:
                                    display_sentences += f" | Partial: {partial_buffer[-30:]}"
                                sys.stdout.write(f"\r🦜 {display_sentences}")
                                sys.stdout.flush()
                                
                                # Check if we have 3+ NEW sentences - emit first 2
                                if len(new_sentences) >= 3:
                                    to_emit = new_sentences[:2]
                                    emit_text = ' '.join(to_emit)
                                    
                                    # Only emit if different from last emission
                                    if emit_text != last_emit_text:
                                        print(f"\n✅ [EMIT] {emit_text}")
                                        
                                        payload = (emit_text, self.current_session_id, "desktop", 0.9)
                                        self.result_queue.put(payload)
                                        
                                        last_emit_text = emit_text
                                        
                                        # Mark these sentences as emitted (store normalized for fuzzy matching)
                                        for s in to_emit:
                                            emitted_sentences.append(s.lower().strip())
                                    
                                self.last_processed = time.time()
                        
                        # Check for silence
                        if peak_amp <= self.SILENCE_THRESHOLD:
                            if time.time() - self.last_processed > 3.0:
                                # Get any remaining new sentences
                                if last_full_text:
                                    corrected_text = self._apply_name_correction(last_full_text)
                                    all_sentences = sentence_pattern.findall(corrected_text)
                                    all_sentences = [s.strip() for s in all_sentences if s.strip()]
                                    
                                    # Get un-emitted sentences (fuzzy match)
                                    remaining_sentences = [s for s in all_sentences if not is_sentence_emitted(s, emitted_sentences)]
                                    
                                    # Also get any partial at the end
                                    partial = ""
                                    if all_sentences:
                                        last_end = corrected_text.rfind(all_sentences[-1]) + len(all_sentences[-1])
                                        partial = corrected_text[last_end:].strip()
                                    
                                    remaining_text = ' '.join(remaining_sentences)
                                    if partial:
                                        remaining_text = (remaining_text + ' ' + partial).strip()
                                    
                                    if remaining_text and remaining_text != last_emit_text:
                                        print(f"\n🍃 [SILENCE FLUSH] {remaining_text}")
                                        
                                        payload = (remaining_text, self.current_session_id, "desktop", 0.9)
                                        self.result_queue.put(payload)
                                
                                # Reset for new session
                                self.current_session_id = str(uuid.uuid4())
                                last_full_text = ""
                                last_emit_text = ""
                                emitted_sentences.clear()
                                print("\n🍃 [Desktop] Session reset.")
                        
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
                print(f"🎧 Listening to desktop audio (device {self.DESKTOP_DEVICE_ID}) - RAW FINALIZED MODE")
                print(f"   Emitting new finalized tokens as they arrive...")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        except Exception as e:
            print(f"\n❌ Error starting desktop audio stream: {e}")
        finally:
            self.stop_event.set()
            print("\nShutting down desktop transcriber...")
            self.cleanup()
            print("🖥️ Desktop transcription stopped.")