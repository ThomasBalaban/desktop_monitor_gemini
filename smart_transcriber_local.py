import numpy as np
import sounddevice as sd
import threading
import queue
import time
from scipy import signal

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


# ============================
# CONFIG
# ============================

TARGET_SAMPLE_RATE = 16000
SILENCE_DB_THRESHOLD = -40
SILENCE_DURATION_SEC = 1.2
MIN_UTTERANCE_SEC = 0.6

GAIN = 2.0
REMOVE_DC = True


# ============================
# TRANSCRIPTION + DIARIZATION
# ============================

class LocalSpeechEngine:
    def __init__(self):
        print("🧠 Loading Whisper model...")
        self.whisper = WhisperModel(
            "medium.en",
            device="cpu",            # M4 CPU is excellent
            compute_type="int8"      # Fast + memory efficient
        )

        print("🧠 Loading speaker diarization model...")
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="YOUR_HF_TOKEN_HERE"
        )

    def process_utterance(self, audio: np.ndarray):
        """
        audio: float32 mono, 16kHz
        """
        print("📝 Processing utterance...")

        diarization = self.diarization(
            {"waveform": audio[None, :], "sample_rate": TARGET_SAMPLE_RATE}
        )

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start = int(segment.start * TARGET_SAMPLE_RATE)
            end = int(segment.end * TARGET_SAMPLE_RATE)
            chunk = audio[start:end]

            if len(chunk) < TARGET_SAMPLE_RATE * 0.3:
                continue

            segments, _ = self.whisper.transcribe(chunk)

            for s in segments:
                text = s.text.strip()
                if text:
                    print(f"🗣 {speaker}: {text}")


# ============================
# AUDIO CAPTURE
# ============================

class SmartAudioTranscriber:
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.queue = queue.Queue(maxsize=500)
        self.running = False

        self.engine = LocalSpeechEngine()

        self.input_rate = 48000
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_voice_time = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()

    def stop(self):
        print("🛑 Stopping transcriber...")
        self.running = False
        self.thread.join(timeout=2)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return

        try:
            self.queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _calculate_db(self, audio):
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms) if rms > 0 else -100

    def _resample(self, audio):
        if self.input_rate == TARGET_SAMPLE_RATE:
            return audio
        samples = int(len(audio) * TARGET_SAMPLE_RATE / self.input_rate)
        return signal.resample(audio, samples)

    def _audio_loop(self):
        dev = sd.query_devices(self.device_id, "input")
        self.input_rate = int(dev["default_samplerate"])

        blocksize = int(self.input_rate * 0.1)

        print(f"🎧 Using device: {dev['name']}")
        print(f"   {self.input_rate}Hz → {TARGET_SAMPLE_RATE}Hz")

        with sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.input_rate,
            callback=self._audio_callback,
            blocksize=blocksize,
            dtype="int16"
        ):
            while self.running:
                while not self.queue.empty():
                    data = self.queue.get()
                    audio = data.flatten().astype(np.float32) / 32768.0

                    if REMOVE_DC:
                        audio -= np.mean(audio)

                    audio = np.clip(audio * GAIN, -1.0, 1.0)

                    db = self._calculate_db(audio)

                    if db >= SILENCE_DB_THRESHOLD:
                        self.last_voice_time = time.time()

                    self.audio_buffer = np.concatenate([self.audio_buffer, audio])

                now = time.time()

                if self.last_voice_time:
                    silence = now - self.last_voice_time
                    utterance_len = len(self.audio_buffer) / self.input_rate

                    if silence >= SILENCE_DURATION_SEC and utterance_len >= MIN_UTTERANCE_SEC:
                        print("🔕 Silence detected, flushing utterance")

                        audio = self._resample(self.audio_buffer)
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.last_voice_time = None

                        self.engine.process_utterance(audio)

                # Prevent runaway memory
                max_len = self.input_rate * 10
                if len(self.audio_buffer) > max_len:
                    self.audio_buffer = self.audio_buffer[-max_len:]

                time.sleep(0.02)


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    import sounddevice as sd

    print(sd.query_devices())
    DEVICE_ID = int(input("Enter input device ID: "))

    t = SmartAudioTranscriber(DEVICE_ID)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        t.stop()
