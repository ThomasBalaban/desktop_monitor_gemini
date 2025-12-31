import numpy as np
import sounddevice as sd
import asyncio
import threading
from scipy import signal
import queue
import time

class SmartAudioTranscriber:
    def __init__(self, client, device_id):
        self.client = client
        self.device_id = device_id
        self.input_rate = 16000     
        self.target_rate = 24000
        self.queue = queue.Queue()
        self.running = False
        
        # Threads
        self.process_thread = None
        self.network_thread = None
        self.loop = None
        
        # Audio Settings
        self.gain = 5.0
        self.remove_dc = True

    def start(self):
        self.running = True
        
        # 1. Create a dedicated Event Loop for Network I/O
        self.loop = asyncio.new_event_loop()
        
        # 2. Start Network Thread (Runs the Async Loop)
        self.network_thread = threading.Thread(target=self._network_worker, args=(self.loop,), daemon=True)
        self.network_thread.start()
        
        # 3. Start Audio Processing Thread (Blocking DSP)
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()

    def stop(self):
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.process_thread:
            self.process_thread.join()
        if self.network_thread:
            self.network_thread.join()

    def _network_worker(self, loop):
        """Runs the asyncio loop forever in a background thread."""
        asyncio.set_event_loop(loop)
        # Connect and keep the loop running
        loop.run_until_complete(self.client.connect())
        loop.run_forever()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio Status]: {status}")
        self.queue.put(indata.copy())

    def _resample(self, audio_data, orig_sr, target_sr):
        if orig_sr == target_sr: return audio_data
        num_samples = int(len(audio_data) * target_sr / orig_sr)
        return signal.resample(audio_data, num_samples)

    def _process_worker(self):
        """Main Audio Processing Loop (DSP & Queue Consumption)."""
        try:
            dev_info = sd.query_devices(self.device_id, 'input')
            self.input_rate = int(dev_info['default_samplerate'])
        except Exception as e:
            print(f"⚠️ Could not query device {self.device_id}: {e}")
            self.input_rate = 48000
            
        print(f"🎧 Capturing device {self.device_id} at {self.input_rate}Hz -> Resampling to {self.target_rate}Hz")
        print(f"🎚️  Gain: {self.gain}x | DC Removal: {self.remove_dc}")

        # Wait briefly for network to be ready
        time.sleep(1)

        try:
            with sd.InputStream(device=self.device_id, channels=1, 
                                samplerate=self.input_rate, callback=self._audio_callback,
                                dtype='int16'):
                
                print(f"✅ Audio Stream Active. Speak/Play Music now!")
                last_print = time.time()
                
                while self.running:
                    try:
                        # 1. Collect Audio Chunk (~100ms)
                        frames_needed = int(self.input_rate * 0.1)
                        audio_buffer = []
                        collected_frames = 0
                        
                        while collected_frames < frames_needed and self.running:
                            try:
                                data = self.queue.get(timeout=0.1)
                                audio_buffer.append(data)
                                collected_frames += len(data)
                            except queue.Empty:
                                break
                        
                        if not audio_buffer: continue

                        full_chunk = np.concatenate(audio_buffer)
                        
                        # --- DSP PIPELINE ---
                        float_audio = full_chunk.astype(np.float32) / 32768.0
                        
                        if self.remove_dc:
                            float_audio = float_audio - np.mean(float_audio)
                        
                        float_audio = float_audio * self.gain
                        float_audio = np.clip(float_audio, -1.0, 1.0)
                        
                        # --- VISUALIZER ---
                        if time.time() - last_print > 0.2:
                            rms = np.sqrt(np.mean(float_audio**2))
                            db = 20 * np.log10(rms + 1e-9)
                            bar_len = int(max(0, (db + 60) / 2))
                            bar = "█" * bar_len
                            # Pad to prevent jitter
                            bar = bar.ljust(30)
                            # Use \r to overwrite line
                            print(f"\r[Vol]: {db:.1f}dB |{bar}|", end="", flush=True)
                            last_print = time.time()

                        # --- RESAMPLE & SEND ---
                        resampled = self._resample(float_audio, self.input_rate, self.target_rate)
                        pcm_bytes = (resampled * 32767).astype(np.int16).tobytes()
                        
                        # Thread-safe send to the running network loop
                        if self.loop and self.loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.client.send_audio_chunk(pcm_bytes), self.loop
                            )
                        
                    except Exception as e:
                        print(f"\nStreaming error: {e}")
                        
        except Exception as e:
            print(f"\n❌ Critical Audio Error on Device {self.device_id}: {e}")