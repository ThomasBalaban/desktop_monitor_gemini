import numpy as np
import sounddevice as sd
import asyncio
import threading
from scipy import signal
import queue
import time

class SmartAudioTranscriber:
    """Streams desktop audio to OpenAI with volume monitoring and chunked sending."""
    def __init__(self, client, device_id):
        self.client = client
        self.device_id = device_id
        self.input_rate = 16000     
        self.target_rate = 24000
        self.queue = queue.Queue(maxsize=500)
        self.running = False
        
        # New: volume monitoring callback
        self.volume_callback = None
        
        # Threads
        self.process_thread = None
        self.network_thread = None
        self.loop = None
        
        # Audio Settings
        self.gain = 5.0
        self.remove_dc = True
        
        # Streaming settings
        self.chunk_duration_ms = 100 
        
        # ===== CHUNKED SENDING SETTINGS =====
        # FIX: Increased to 15s to stay under OpenAI transcription rate limits
        self.send_interval_seconds = 15.0  
        self.db_threshold = -35 
        self.overlap_seconds = 0.5 

    def set_volume_callback(self, callback):
        """Register callback for GUI volume updates"""
        self.volume_callback = callback

    def start(self):
        self.running = True
        self.loop = asyncio.new_event_loop()
        self.network_thread = threading.Thread(target=self._network_worker, args=(self.loop,), daemon=True)
        self.network_thread.start()
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()

    def stop(self):
        """Gracefully stop all threads and connections."""
        print("    Stopping SmartAudioTranscriber...")
        self.running = False
        
        # 1. Disconnect the OpenAI client
        if self.client and self.loop and self.loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.client.disconnect(), 
                    self.loop
                )
                future.result(timeout=2.0)
            except Exception as e:
                print(f"      Error disconnecting client: {e}")
        
        # 2. Stop the event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # 3. Wait for threads to finish
        if self.network_thread and self.network_thread.is_alive():
            self.network_thread.join(timeout=2.0)
            
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        # 4. Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
                
        print("    SmartAudioTranscriber stopped.")

    def _network_worker(self, loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.client.connect())
        except Exception as e:
            if self.running:
                print(f"Network worker error: {e}")
        finally:
            try:
                loop.run_forever()
            except:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback with volume reporting."""
        if not self.running:
            return
            
        # Volume Calculation for GUI
        float_data = indata.flatten().astype(np.float32) / 32768.0
        rms_val = np.sqrt(np.mean(float_data**2))
        
        if self.volume_callback:
            self.volume_callback(min(1.0, rms_val * 10))
            
        try:
            self.queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(indata.copy())
            except:
                pass

    def _resample(self, audio_data, orig_sr, target_sr):
        if orig_sr == target_sr: 
            return audio_data
        num_samples = int(len(audio_data) * target_sr / orig_sr)
        return signal.resample(audio_data, num_samples)

    def _calculate_db(self, audio_float):
        """Calculate dB level of audio chunk."""
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -100 
        return db

    def _process_worker(self):
        try:
            dev_info = sd.query_devices(self.device_id, 'input')
            self.input_rate = int(dev_info['default_samplerate'])
            device_name = dev_info['name']
        except Exception as e:
            print(f"⚠️ Could not query device {self.device_id}: {e}")
            self.input_rate = 48000
            device_name = f"Device {self.device_id}"
            
        print(f"🎧 OpenAI Audio: {device_name}")
        print(f"   Rate: {self.input_rate}Hz → {self.target_rate}Hz | Interval: {self.send_interval_seconds}s")

        time.sleep(2.0)
        samples_per_chunk = int(self.input_rate * self.chunk_duration_ms / 1000)
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                self._run_audio_stream(samples_per_chunk)
                break
            except Exception as e:
                if not self.running: break
                retry_count += 1
                print(f"⚠️ Audio error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(2.0)

    def _run_audio_stream(self, samples_per_chunk):
        with sd.InputStream(
            device=self.device_id, channels=1, samplerate=self.input_rate, 
            callback=self._audio_callback, blocksize=samples_per_chunk,
            dtype='int16', latency='low'
        ) as stream:
            print(f"✅ OpenAI Audio Stream Active")
            
            audio_buffer = np.array([], dtype=np.float32)
            overlap_buffer = np.array([], dtype=np.float32)
            
            last_send_time = time.time()
            samples_per_interval = int(self.input_rate * self.send_interval_seconds)
            samples_per_overlap = int(self.input_rate * self.overlap_seconds)
            
            while self.running:
                while not self.queue.empty():
                    try:
                        data = self.queue.get_nowait()
                        float_chunk = data.flatten().astype(np.float32) / 32768.0
                        audio_buffer = np.concatenate([audio_buffer, float_chunk])
                    except queue.Empty: break
                
                current_time = time.time()
                if current_time - last_send_time >= self.send_interval_seconds and len(audio_buffer) > 0:
                    full_audio = np.concatenate([overlap_buffer, audio_buffer]) if len(overlap_buffer) > 0 else audio_buffer
                    
                    if self.remove_dc: full_audio = full_audio - np.mean(full_audio)
                    full_audio = np.clip(full_audio * self.gain, -1.0, 1.0)
                    
                    if self._calculate_db(full_audio) >= self.db_threshold:
                        resampled = self._resample(full_audio, self.input_rate, self.target_rate)
                        pcm_bytes = (resampled * 32767).astype(np.int16).tobytes()
                        if self.loop and self.loop.is_running():
                            asyncio.run_coroutine_threadsafe(self.client.send_audio_chunk(pcm_bytes), self.loop)
                    
                    overlap_buffer = audio_buffer[-samples_per_overlap:].copy() if len(audio_buffer) > samples_per_overlap else audio_buffer.copy()
                    audio_buffer = np.array([], dtype=np.float32)
                    last_send_time = current_time
                
                time.sleep(0.02)