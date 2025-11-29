import sounddevice as sd
import numpy as np
import threading
import queue
import time

class AudioCapture:
    def __init__(self, device_id, sample_rate=16000, channels=1):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None
        self.lock = threading.Lock()
        
        # Audio threshold for "Hearing something" (RMS)
        # Adjust this if it triggers too easily or not enough
        self.silence_threshold = 0.01

    def _callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio chunks."""
        if status:
            print(f"[AudioCapture] Status: {status}")
        self.audio_queue.put(indata.copy())

    def start(self):
        if self.is_running:
            return
        
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._callback,
                dtype='int16'  # Gemini prefers PCM 16-bit
            )
            self.stream.start()
            self.is_running = True
            print(f"AudioCapture started on device {self.device_id}")
        except Exception as e:
            print(f"Failed to start AudioCapture: {e}")

    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            self.stream = None

    def get_recent_audio(self):
        """
        Retrieves all audio accumulated since the last call.
        Returns: (bytes, is_loud)
        """
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())
        
        if not frames:
            return None, False

        # Concatenate all chunks
        audio_data = np.concatenate(frames, axis=0)
        
        # Calculate Volume (RMS) to decide if we should trigger a "Turn"
        # Normalize int16 to float -1..1 for calculation
        audio_float = audio_data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        is_loud = rms > self.silence_threshold

        return audio_data.tobytes(), is_loud