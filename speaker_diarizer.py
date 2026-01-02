"""
Real-time Speaker Diarization using Resemblyzer embeddings.
Lightweight and fast - prioritizes speed over accuracy.
"""

import numpy as np
from collections import deque
import threading

class SpeakerDiarizer:
    """
    Identifies speakers in real-time using voice embeddings.
    Assigns consistent labels like "Speaker 1", "Speaker 2", etc.
    """
    
    def __init__(self, similarity_threshold=0.75, max_speakers=10):
        """
        Args:
            similarity_threshold: How similar an embedding must be to match (0.0-1.0)
                                 Lower = more likely to match existing speaker
                                 Higher = more likely to create new speaker
            max_speakers: Maximum number of unique speakers to track
        """
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        
        # Speaker profiles: speaker_id -> deque of recent embeddings
        self.speaker_profiles = {}
        self.speaker_count = 0
        self.max_embeddings_per_speaker = 20  # Rolling window of embeddings
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Lazy-load the encoder (it's ~50MB)
        self._encoder = None
        self._encoder_lock = threading.Lock()
        
        print("🎭 Speaker Diarizer initialized (Resemblyzer)")
    
    @property
    def encoder(self):
        """Lazy-load the voice encoder on first use."""
        if self._encoder is None:
            with self._encoder_lock:
                if self._encoder is None:  # Double-check after acquiring lock
                    print("🔄 Loading Resemblyzer voice encoder...")
                    from resemblyzer import VoiceEncoder
                    self._encoder = VoiceEncoder()
                    print("✅ Voice encoder ready")
        return self._encoder
    
    def identify_speaker(self, audio_data, sample_rate=16000):
        """
        Identify which speaker produced this audio segment.
        
        Args:
            audio_data: numpy array of audio samples (float32, mono)
            sample_rate: sample rate of the audio (default 16000)
            
        Returns:
            tuple: (speaker_label, confidence)
                   speaker_label: "Speaker 1", "Speaker 2", etc.
                   confidence: similarity score (0.0-1.0)
        """
        try:
            # Ensure audio is the right format
            if len(audio_data) < sample_rate * 0.3:  # Need at least 0.3s of audio
                return "Unknown", 0.0
            
            audio_float = self._prepare_audio(audio_data, sample_rate)
            
            if audio_float is None or len(audio_float) < 4000:
                return "Unknown", 0.0
            
            # Get embedding for this audio segment
            # Resemblyzer expects 16kHz audio
            if sample_rate != 16000:
                audio_float = self._resample(audio_float, sample_rate, 16000)
            
            embedding = self.encoder.embed_utterance(audio_float)
            
            # Find best matching speaker
            with self.lock:
                best_speaker, best_similarity = self._find_best_match(embedding)
                
                if best_speaker is None:
                    # Create new speaker
                    if self.speaker_count < self.max_speakers:
                        self.speaker_count += 1
                        best_speaker = f"Speaker {self.speaker_count}"
                        self.speaker_profiles[best_speaker] = deque(maxlen=self.max_embeddings_per_speaker)
                        best_similarity = 1.0  # Perfect match with self
                        print(f"🆕 New speaker detected: {best_speaker}")
                    else:
                        # At max speakers, assign to closest match anyway
                        best_speaker, best_similarity = self._find_closest_match(embedding)
                
                # Update speaker profile with this embedding
                if best_speaker in self.speaker_profiles:
                    self.speaker_profiles[best_speaker].append(embedding)
                
                return best_speaker, float(best_similarity)
                
        except Exception as e:
            print(f"⚠️ Diarization error: {e}")
            return "Unknown", 0.0
    
    def _prepare_audio(self, audio_data, sample_rate):
        """Prepare audio data for embedding extraction."""
        # Convert to numpy if needed
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.float32:
            audio_float = audio_data
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(audio_float))
        if max_val > 0.001:
            audio_float = audio_float / max_val
        else:
            return None  # Too quiet
        
        return audio_float
    
    def _resample(self, audio, orig_sr, target_sr):
        """Simple resampling using scipy."""
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples).astype(np.float32)
    
    def _find_best_match(self, embedding):
        """Find the best matching speaker above threshold."""
        best_speaker = None
        best_similarity = self.similarity_threshold
        
        for speaker_id, embeddings in self.speaker_profiles.items():
            if len(embeddings) == 0:
                continue
            
            # Compare against centroid of recent embeddings
            centroid = np.mean(list(embeddings), axis=0)
            similarity = np.dot(embedding, centroid)
            
            if similarity > best_similarity:
                best_speaker = speaker_id
                best_similarity = similarity
        
        return best_speaker, best_similarity
    
    def _find_closest_match(self, embedding):
        """Find closest speaker regardless of threshold (used when at max speakers)."""
        best_speaker = "Speaker 1"  # Default fallback
        best_similarity = 0.0
        
        for speaker_id, embeddings in self.speaker_profiles.items():
            if len(embeddings) == 0:
                continue
            
            centroid = np.mean(list(embeddings), axis=0)
            similarity = np.dot(embedding, centroid)
            
            if similarity > best_similarity:
                best_speaker = speaker_id
                best_similarity = similarity
        
        return best_speaker, best_similarity
    
    def reset(self):
        """Reset all speaker profiles (e.g., when content changes significantly)."""
        with self.lock:
            self.speaker_profiles.clear()
            self.speaker_count = 0
            print("🔄 Speaker profiles reset")
    
    def get_speaker_count(self):
        """Return current number of identified speakers."""
        return self.speaker_count
    
    def get_stats(self):
        """Return statistics about speaker identification."""
        with self.lock:
            stats = {
                "total_speakers": self.speaker_count,
                "speakers": {}
            }
            for speaker_id, embeddings in self.speaker_profiles.items():
                stats["speakers"][speaker_id] = {
                    "embedding_count": len(embeddings)
                }
            return stats


class BufferedDiarizer:
    """
    Buffers audio samples and performs diarization when enough audio is collected.
    Useful for streaming scenarios where audio comes in small chunks.
    """
    
    def __init__(self, diarizer, buffer_duration=1.0, sample_rate=16000):
        """
        Args:
            diarizer: SpeakerDiarizer instance
            buffer_duration: Seconds of audio to buffer before diarization
            sample_rate: Expected sample rate of incoming audio
        """
        self.diarizer = diarizer
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        
        # Track last identified speaker for continuity
        self.last_speaker = None
        self.last_confidence = 0.0
    
    def add_audio(self, audio_chunk):
        """
        Add audio to buffer. Returns speaker info if buffer is full.
        
        Args:
            audio_chunk: numpy array of audio samples
            
        Returns:
            tuple or None: (speaker_label, confidence) if identification performed, else None
        """
        with self.lock:
            # Convert chunk to float32 if needed
            if audio_chunk.dtype == np.int16:
                chunk_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                chunk_float = audio_chunk.astype(np.float32)
            
            self.buffer = np.concatenate([self.buffer, chunk_float.flatten()])
            
            # Check if we have enough audio
            target_samples = int(self.buffer_duration * self.sample_rate)
            
            if len(self.buffer) >= target_samples:
                # Extract buffer for processing
                audio_to_process = self.buffer[:target_samples].copy()
                
                # Keep some overlap for continuity
                overlap = int(0.2 * self.sample_rate)  # 200ms overlap
                self.buffer = self.buffer[target_samples - overlap:]
                
                # Perform diarization
                speaker, confidence = self.diarizer.identify_speaker(
                    audio_to_process, 
                    self.sample_rate
                )
                
                self.last_speaker = speaker
                self.last_confidence = confidence
                
                return speaker, confidence
        
        return None
    
    def get_current_speaker(self):
        """Get the most recently identified speaker."""
        return self.last_speaker, self.last_confidence
    
    def clear(self):
        """Clear the audio buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)


# Singleton instance for easy access across modules
_global_diarizer = None

def get_diarizer():
    """Get or create the global diarizer instance."""
    global _global_diarizer
    if _global_diarizer is None:
        _global_diarizer = SpeakerDiarizer(
            similarity_threshold=0.72,  # Slightly lower for more speaker continuity
            max_speakers=8
        )
    return _global_diarizer