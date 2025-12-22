# transcriber_core/desktop_transcriber.py
"""
Desktop Audio Transcriber using faster-whisper with fixed-interval chunking.
Optimized for continuous audio like gameplay dialogue, cutscenes, and media.

Strategy: Process audio in 3-second windows with 1.5s overlap buffer to avoid
missing words at chunk boundaries. Uses previous transcriptions as context
for improved accuracy.
"""

import os
import time
import re
import sys
import uuid
from threading import Thread, Event, Lock
from queue import Queue, Empty
from collections import deque
import sounddevice as sd
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from .desktop_speech_music_classifier import SpeechMusicClassifier
from .config import (
    FS, SAVE_DIR, DESKTOP_DEVICE_ID,
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
)

# Define stop_event at the module level
stop_event = Event()

# Chunking Parameters - Tuned for live streaming with context
CHUNK_DURATION = 3.0      # Process every 3 seconds for lower latency
OVERLAP_DURATION = 1.5    # 1.5 second overlap to catch boundary words
MIN_AUDIO_ENERGY = 0.002  # Skip chunks that are essentially silent
CONTEXT_TRANSCRIPTS = 2   # Number of previous transcriptions to use as context


class SpeechMusicTranscriber:
    """
    Desktop audio transcriber using faster-whisper with fixed-interval chunking.
    Uses overlapping windows and previous transcription context for accuracy.
    """
    
    def __init__(self, keep_files=False, auto_detect=True, transcript_manager=None):
        self.FS = FS
        self.SAVE_DIR = SAVE_DIR
        self.DESKTOP_DEVICE_ID = DESKTOP_DEVICE_ID
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print(f"🎧 Initializing faster-whisper for Desktop Audio (Context-Aware Mode)...")
        print(f"   Chunk: {CHUNK_DURATION}s | Overlap: {OVERLAP_DURATION}s | Context: last {CONTEXT_TRANSCRIPTS} transcripts")
        
        try:
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            print(f"✅ faster-whisper model '{WHISPER_MODEL_SIZE}' loaded on {WHISPER_DEVICE}.")
        except Exception as e:
            print(f"❌ Error loading faster-whisper model: {e}")
            raise

        self.result_queue = Queue()
        self.stop_event = stop_event
        self.saved_files = []
        self.keep_files = keep_files
        
        self.classifier = SpeechMusicClassifier()
        self.auto_detect = auto_detect
        self.transcript_manager = transcript_manager
        
        # Audio buffer - use a Queue instead of locked array for thread safety
        self.audio_queue = Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Timing
        self.chunk_samples = int(CHUNK_DURATION * FS)
        self.overlap_samples = int(OVERLAP_DURATION * FS)
        self.window_samples = self.chunk_samples + self.overlap_samples
        
        # Session tracking
        self.current_session_id = str(uuid.uuid4())
        self.last_processed = time.time()
        
        # Context - store recent transcriptions for conditioning (thread-safe deque)
        self.transcript_history = deque(maxlen=CONTEXT_TRANSCRIPTS)
        
        # Deduplication
        self.recent_texts = deque(maxlen=5)
        
        # Volume boost for quiet desktop audio
        self.VOLUME_BOOST = 3.0

        # Name correction dictionary
        self.name_variations = {
            r'\bnaomi\b': 'Nami', r'\bnow may\b': 'Nami', r'\bnomi\b': 'Nami',
            r'\bnamy\b': 'Nami', r'\bnot me\b': 'Nami', r'\bnah me\b': 'Nami',
            r'\bnonny\b': 'Nami', r'\bnonni\b': 'Nami', r'\bmamie\b': 'Nami',
            r'\bgnomey\b': 'Nami', r'\barmy\b': 'Nami', r'\bpeepingnaomi\b': 'PeepingNami',
            r'\bpeepingnomi\b': 'PeepingNami'
        }

    def cleanup(self):
        """Cleanup resources."""
        print("🧹 [Desktop] Cleaning up faster-whisper resources...")
        self.model = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_history.clear()
        print("✅ [Desktop] Cleanup complete.")

    def _apply_name_correction(self, text):
        """Apply name corrections to transcribed text."""
        corrected_text = text
        for variation, name in self.name_variations.items():
            corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)
        return corrected_text

    def _get_context_prompt(self):
        """Build context prompt from recent transcriptions."""
        try:
            if not self.transcript_history:
                return None
            context = " ".join(self.transcript_history)
            if len(context) > 500:
                context = context[-500:]
            return context
        except:
            return None

    def _add_to_history(self, text):
        """Add transcription to history for context."""
        try:
            self.transcript_history.append(text)
        except:
            pass

    def _is_duplicate(self, text):
        """Check if text is too similar to recent transcriptions."""
        text_lower = text.lower().strip()
        
        if len(text_lower) < 5:
            return True
        
        try:
            for recent in list(self.recent_texts):
                if text_lower == recent:
                    return True
                if text_lower in recent:
                    return True
                words_new = set(text_lower.split())
                words_old = set(recent.split())
                if words_new and words_old:
                    overlap = len(words_new & words_old) / max(len(words_new), len(words_old))
                    if overlap > 0.7:
                        return True
        except:
            pass
        return False

    def _add_to_recent(self, text):
        """Add text to recent history for deduplication."""
        try:
            self.recent_texts.append(text.lower().strip())
        except:
            pass

    def audio_callback(self, indata, frames, timestamp, status):
        """
        Buffers incoming audio. Uses queue for lock-free thread safety.
        """
        if status and status.input_overflow:
            pass  # Silently handle overflow

        if self.stop_event.is_set():
            return

        # Convert to mono if stereo
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            new_audio = np.mean(indata, axis=1).astype(np.float32)
        else:
            new_audio = indata.flatten().astype(np.float32)

        # Apply volume boost and queue
        self.audio_queue.put(new_audio * self.VOLUME_BOOST)

    def _processing_loop(self):
        """
        Single-threaded processing loop - no thread pool, just sequential processing.
        This avoids thread contention and the freezing issue.
        """
        print("🔄 [Desktop] Processing loop started (single-threaded).")
        
        while not self.stop_event.is_set():
            try:
                # Drain audio queue into buffer
                chunks_added = 0
                while True:
                    try:
                        audio_chunk = self.audio_queue.get_nowait()
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                        chunks_added += 1
                    except Empty:
                        break
                
                # Limit buffer size (max 30 seconds)
                max_samples = int(30 * self.FS)
                if len(self.audio_buffer) > max_samples:
                    self.audio_buffer = self.audio_buffer[-max_samples:]
                
                # Check if we have enough audio for a full window
                if len(self.audio_buffer) >= self.window_samples:
                    # Extract window
                    window = self.audio_buffer[:self.window_samples].copy()
                    
                    # Advance buffer by chunk size (keep overlap)
                    self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                    
                    # Check energy
                    rms = np.sqrt(np.mean(window**2))
                    if rms < MIN_AUDIO_ENERGY:
                        continue
                    
                    # Get context and transcribe (blocking, single-threaded)
                    context = self._get_context_prompt()
                    self._transcribe_chunk(window, self.current_session_id, context)
                else:
                    # Not enough audio yet, small sleep
                    time.sleep(0.05)
                
            except Exception as e:
                print(f"[DESKTOP-ERROR] Processing loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        print("🛑 [Desktop] Processing loop stopped.")

    def _transcribe_chunk(self, audio_window, session_id, context_prompt):
        """
        Transcribe an audio window using faster-whisper.
        Runs in the processing thread (single-threaded, no pool).
        """
        start_time = time.time()
        
        try:
            # Optional: Classify audio type
            if self.auto_detect:
                audio_type, confidence = self.classifier.classify(audio_window)
                if audio_type == "music" and confidence > 0.75:
                    return
            
            # Build transcription parameters
            transcribe_params = {
                'beam_size': 1,
                'language': "en",
                'vad_filter': True,
                'vad_parameters': dict(
                    min_silence_duration_ms=250,
                    speech_pad_ms=150
                ),
                'condition_on_previous_text': True,
            }
            
            # Add context from previous transcriptions
            if context_prompt:
                transcribe_params['initial_prompt'] = context_prompt
            
            # Transcribe
            segments, info = self.model.transcribe(audio_window, **transcribe_params)
            
            # Collect all segment text
            text_parts = []
            for segment in segments:
                seg_text = segment.text.strip()
                if seg_text:
                    text_parts.append(seg_text)
            
            text = " ".join(text_parts).strip()
            
            # Clean up artifacts
            text = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', text)
            text = re.sub(r'[♪♫🎵🎶]+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            elapsed = time.time() - start_time
            
            if text and len(text) >= 3:
                # Check for duplicates
                if self._is_duplicate(text):
                    return
                
                # Apply name corrections
                corrected_text = self._apply_name_correction(text)
                
                # Add to history
                self._add_to_history(corrected_text)
                self._add_to_recent(corrected_text)
                
                duration = len(audio_window) / self.FS
                print(f"\n✅ [Desktop] ({duration:.1f}s → {elapsed:.2f}s): {corrected_text}")
                
                # Queue result
                payload = (corrected_text, session_id, "desktop", 0.85)
                self.result_queue.put(payload)
                self.last_processed = time.time()
                    
        except Exception as e:
            print(f"[DESKTOP-ERROR] Transcription failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def run(self):
        """Start the audio stream and processing loop."""
        try:
            device_info = sd.query_devices(self.DESKTOP_DEVICE_ID)
            channels = min(device_info['max_input_channels'], 2)
            
            print(f"\n🎧 Desktop Audio Configuration:")
            print(f"   Device ID: {self.DESKTOP_DEVICE_ID}")
            print(f"   Device: {device_info['name']}")
            print(f"   Sample Rate: {self.FS} Hz")
            print(f"   Model: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE}")
            print(f"   Window: {CHUNK_DURATION + OVERLAP_DURATION}s ({CHUNK_DURATION}s chunk + {OVERLAP_DURATION}s overlap)")
            print(f"   Context: Last {CONTEXT_TRANSCRIPTS} transcriptions as prompt")
            print(f"   Expected Latency: ~{CHUNK_DURATION + 0.5:.1f}s")
            
            # Start processing thread (single thread, no pool)
            processing_thread = Thread(target=self._processing_loop, daemon=True)
            processing_thread.start()
            
            stream_kwargs = {
                'device': self.DESKTOP_DEVICE_ID,
                'samplerate': self.FS,
                'channels': channels,
                'callback': self.audio_callback,
                'blocksize': self.FS // 10,
                'dtype': 'float32'
            }
            
            with sd.InputStream(**stream_kwargs):
                print(f"\n🎧 Listening to desktop audio...")
                print(f"   Context-aware transcription with overlapping windows\n")
                
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        except Exception as e:
            print(f"\n❌ Error starting desktop audio stream: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_event.set()
            print("\nShutting down desktop transcriber...")
            self.cleanup()
            print("🖥️ Desktop transcription stopped.")