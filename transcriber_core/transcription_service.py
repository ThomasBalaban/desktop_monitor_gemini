# transcriber_core/transcription_service.py
"""
Transcription Service - Manages both Microphone (Parakeet) and Desktop (faster-whisper) transcribers.

Architecture:
- Runs transcription engines in a separate PROCESS to isolate heavy ML models from the GUI.
- Microphone uses Parakeet MLX streaming (great for natural speech with pauses)
- Desktop uses faster-whisper with VAD batch processing (reliable for continuous audio)
"""

import time
import multiprocessing
import traceback
import sys
import os
import signal
from queue import Empty
from difflib import SequenceMatcher

try:
    from transcriber_core import MicrophoneTranscriber
    from transcriber_core import DesktopTranscriber
except ImportError:
    print("CRITICAL ERROR: Could not import 'transcriber_core'. Make sure the module is properly installed.")
    raise


class TranscriptionDeduplicator:
    """Filters out overlapping/duplicate transcriptions with continuation detection"""
    
    def __init__(self, similarity_threshold=0.65, time_window=2.5, overlap_words=3):
        self.recent_transcripts = {}  # source -> (text, timestamp)
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.overlap_words = overlap_words
        
    def _get_overlap_length(self, prev_text, curr_text):
        """Calculate how many words overlap at the end of prev and start of curr"""
        prev_words = prev_text.lower().split()
        curr_words = curr_text.lower().split()
        
        max_check = min(len(prev_words), len(curr_words), 10)
        overlap_length = 0
        
        for i in range(1, max_check + 1):
            if prev_words[-i:] == curr_words[:i]:
                overlap_length = i
        
        return overlap_length
    
    def _is_substring_or_similar(self, prev_text, curr_text):
        """Check if curr_text is contained in prev_text or vice versa"""
        prev_lower = prev_text.lower().strip()
        curr_lower = curr_text.lower().strip()
        
        if curr_lower in prev_lower or prev_lower in curr_lower:
            return True
        
        ratio = SequenceMatcher(None, prev_lower, curr_lower).ratio()
        return ratio > self.similarity_threshold
    
    def process(self, text, source):
        """
        Process a transcription and decide whether to skip, merge, or output.
        Returns: (should_output, final_text, is_partial)
        """
        current_time = time.time()
        is_partial = False
        
        # Handle partial updates
        if source.endswith("_partial"):
            is_partial = True
            return True, text, is_partial
        
        # Get most recent from this source
        if source in self.recent_transcripts:
            prev_text, prev_timestamp = self.recent_transcripts[source]
            
            # Check if too old
            if current_time - prev_timestamp > self.time_window:
                self.recent_transcripts[source] = (text, current_time)
                return True, text, is_partial
            
            # Check for duplicate/substring
            if self._is_substring_or_similar(prev_text, text):
                if text.lower().strip() in prev_text.lower():
                    return False, None, is_partial
                elif prev_text.lower().strip() in text.lower():
                    self.recent_transcripts[source] = (text, current_time)
                    return True, text, is_partial
                else:
                    return False, None, is_partial
            
            # Check for continuation
            overlap_length = self._get_overlap_length(prev_text, text)
            
            if overlap_length >= self.overlap_words:
                curr_words = text.split()
                unique_part = " ".join(curr_words[overlap_length:])
                
                if unique_part.strip():
                    merged_text = prev_text + " " + unique_part
                    self.recent_transcripts[source] = (merged_text, current_time)
                    return True, merged_text, is_partial
                else:
                    return False, None, is_partial
            
            self.recent_transcripts[source] = (text, current_time)
            return True, text, is_partial
        
        else:
            self.recent_transcripts[source] = (text, current_time)
            return True, text, is_partial


# Global references for cleanup
_mic_transcriber = None
_desktop_transcriber = None


def transcription_process_target(result_queue, stop_event):
    """
    This function runs in a separate PROCESS.
    It isolates the heavy ML models from the GUI.
    """
    global _mic_transcriber, _desktop_transcriber
    
    print("🧠 [SENSES] Initializing Transcription Engines...")
    print("   • Microphone: Parakeet MLX (streaming)")
    print("   • Desktop: faster-whisper (VAD batch)")
    
    def signal_handler(signum, frame):
        print(f"\n🛑 [SENSES] Received signal {signum}, cleaning up...")
        stop_event.set()
        cleanup_transcribers()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    def cleanup_transcribers():
        global _mic_transcriber, _desktop_transcriber
        
        if _mic_transcriber:
            try:
                _mic_transcriber.stop_event.set()
                if hasattr(_mic_transcriber, 'cleanup'):
                    _mic_transcriber.cleanup()
            except Exception as e:
                print(f"⚠️ Error cleaning up mic transcriber: {e}")
        
        if _desktop_transcriber:
            try:
                _desktop_transcriber.stop_event.set()
                if hasattr(_desktop_transcriber, 'cleanup'):
                    _desktop_transcriber.cleanup()
            except Exception as e:
                print(f"⚠️ Error cleaning up desktop transcriber: {e}")
    
    try:
        # Initialize Engines
        _mic_transcriber = MicrophoneTranscriber(keep_files=False)
        _desktop_transcriber = DesktopTranscriber(keep_files=False)
        deduplicator = TranscriptionDeduplicator(similarity_threshold=0.70)

        # Start Threads
        import threading
        mic_thread = threading.Thread(target=_mic_transcriber.run, daemon=True, name="MicThread")
        desktop_thread = threading.Thread(target=_desktop_transcriber.run, daemon=True, name="DesktopThread")
        
        desktop_thread.start()
        time.sleep(1)
        mic_thread.start()
        
        print("✅ [SENSES] Both transcription systems running")

        while not stop_event.is_set():
            # 1. Check Microphone Queue
            try:
                if not _mic_transcriber.result_queue.empty():
                    # Microphone output: (text, filename, source, confidence)
                    text, filename, source, conf = _mic_transcriber.result_queue.get_nowait()
                    should_output, final_text, is_partial = deduplicator.process(text, "microphone")
                    
                    if should_output and final_text:
                        payload = {
                            "type": "transcript",
                            "source": "microphone",
                            "text": final_text,
                            "confidence": conf,
                            "timestamp": time.time(),
                            "is_partial": is_partial
                        }
                        result_queue.put(payload)
                    _mic_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Mic Queue Error: {e}")

            # 2. Check Desktop Queue
            try:
                if not _desktop_transcriber.result_queue.empty():
                    # Desktop output: (text, session_id, source, confidence)
                    text, session_id, source, conf = _desktop_transcriber.result_queue.get_nowait()
                    should_output, final_text, is_partial = deduplicator.process(text, source)
                    
                    if should_output and final_text:
                        payload = {
                            "type": "transcript",
                            "source": "desktop",
                            "text": final_text,
                            "session_id": session_id,
                            "confidence": conf,
                            "is_partial": is_partial,
                            "timestamp": time.time()
                        }
                        result_queue.put(payload)
                    _desktop_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Desktop Queue Error: {e}")

            time.sleep(0.02)

    except Exception as e:
        print(f"❌ [SENSES] Critical Transcription Failure: {e}")
        traceback.print_exc()
    finally:
        print("🛑 [SENSES] Stopping Transcription Engines...")
        cleanup_transcribers()
        print("✅ [SENSES] Cleanup complete.")


class TranscriptionService:
    """
    Main service class that manages the transcription subprocess.
    Use from the main application to start/stop transcription and get results.
    """
    
    def __init__(self):
        self.process = None
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

    def start(self):
        """Start the transcription subprocess."""
        if self.process is not None and self.process.is_alive():
            return
        
        self.stop_event.clear()
        self.process = multiprocessing.Process(
            target=transcription_process_target,
            args=(self.result_queue, self.stop_event),
            daemon=True
        )
        self.process.start()
    
    def stop(self):
        """Stop the transcription subprocess gracefully."""
        print("🛑 [TranscriptionService] Stopping...")
        self.stop_event.set()
        
        if self.process:
            self.process.join(timeout=5)
            
            if self.process.is_alive():
                print("⚠️ [TranscriptionService] Process didn't stop gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=2)
                
                if self.process.is_alive():
                    print("⚠️ [TranscriptionService] Force killing process...")
                    self.process.kill()
            
            self.process = None
        
        # Clear the queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        
        print("✅ [TranscriptionService] Stopped.")

    def get_results(self):
        """
        Get all available transcription results from the queue.
        Returns a list of result dictionaries.
        """
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break
        return results