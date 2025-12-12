import time
import multiprocessing
import traceback
import sys
import os
from queue import Empty
from difflib import SequenceMatcher

try:
    from transcriber_core import MicrophoneTranscriber
    from transcriber_core import DesktopTranscriber
except ImportError:
    print("CRITICAL ERROR: Could not import 'transcriber_core'. Make sure you copied the folder from audio_mon to this project's root.")

class TranscriptionDeduplicator:
    """Filters out overlapping/duplicate transcriptions with continuation detection"""
    
    def __init__(self, similarity_threshold=0.65, time_window=2.5, overlap_words=3):
        self.recent_transcripts = {}  # source -> (text, timestamp)
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.overlap_words = overlap_words  # Min words to check for overlap
        
    def _get_overlap_length(self, prev_text, curr_text):
        """Calculate how many words overlap at the end of prev and start of curr"""
        prev_words = prev_text.lower().split()
        curr_words = curr_text.lower().split()
        
        max_check = min(len(prev_words), len(curr_words), 10)  # Check up to 10 words
        overlap_length = 0
        
        for i in range(1, max_check + 1):
            if prev_words[-i:] == curr_words[:i]:
                overlap_length = i
        
        return overlap_length
    
    def _is_substring_or_similar(self, prev_text, curr_text):
        """Check if curr_text is contained in prev_text or vice versa"""
        prev_lower = prev_text.lower().strip()
        curr_lower = curr_text.lower().strip()
        
        # Check substring
        if curr_lower in prev_lower or prev_lower in curr_lower:
            return True
        
        # Check high similarity
        ratio = SequenceMatcher(None, prev_lower, curr_lower).ratio()
        return ratio > self.similarity_threshold
    
    def process(self, text, source):
        """
        Process a transcription and decide whether to skip, merge, or output.
        Returns: (should_output, final_text)
        """
        current_time = time.time()
        
        # Get most recent from this source
        if source in self.recent_transcripts:
            prev_text, prev_timestamp = self.recent_transcripts[source]
            
            # Check if too old
            if current_time - prev_timestamp > self.time_window:
                # Old enough to be separate
                self.recent_transcripts[source] = (text, current_time)
                return True, text
            
            # Check for duplicate/substring
            if self._is_substring_or_similar(prev_text, text):
                # Skip if current is contained in previous
                if text.lower().strip() in prev_text.lower():
                    return False, None
                # Update if current contains previous (it's longer)
                elif prev_text.lower().strip() in text.lower():
                    self.recent_transcripts[source] = (text, current_time)
                    return True, text
                else:
                    # Very similar, skip
                    return False, None
            
            # Check for continuation (overlapping words)
            overlap_length = self._get_overlap_length(prev_text, text)
            
            if overlap_length >= self.overlap_words:
                # This is a continuation, merge them
                curr_words = text.split()
                
                # Remove overlapping words from current
                unique_part = " ".join(curr_words[overlap_length:])
                
                if unique_part.strip():
                    # Merge: previous + new unique part
                    merged_text = prev_text + " " + unique_part
                    self.recent_transcripts[source] = (merged_text, current_time)
                    return True, merged_text
                else:
                    # No new content, skip
                    return False, None
            
            # Not a continuation or duplicate, treat as new
            self.recent_transcripts[source] = (text, current_time)
            return True, text
        
        else:
            # First time seeing this source
            self.recent_transcripts[source] = (text, current_time)
            return True, text

def transcription_process_target(result_queue, stop_event):
    """
    This function runs in a separate PROCESS.
    It isolates the heavy ML models from the GUI.
    """
    print("🧠 [SENSES] Initializing Local Transcription Engines...")
    
    try:
        # Initialize Engines
        mic_transcriber = MicrophoneTranscriber(keep_files=False)
        desktop_transcriber = DesktopTranscriber(keep_files=False)
        deduplicator = TranscriptionDeduplicator(similarity_threshold=0.70)

        # Start Threads
        import threading
        mic_thread = threading.Thread(target=mic_transcriber.run, daemon=True, name="MicThread")
        desktop_thread = threading.Thread(target=desktop_transcriber.run, daemon=True, name="DesktopThread")
        
        desktop_thread.start()
        time.sleep(1) # Small stagger to prevent CPU spike
        mic_thread.start()
        
        print("✅ [SENSES] Local Ears Open (Parakeet/Whisper Running)")

        while not stop_event.is_set():
            # 1. Check Microphone
            try:
                if not mic_transcriber.result_queue.empty():
                    text, filename, source, conf = mic_transcriber.result_queue.get_nowait()
                    should_output, final_text = deduplicator.process(text, "microphone")
                    
                    if should_output:
                        payload = {
                            "type": "transcript",
                            "source": "microphone",
                            "text": final_text,
                            "confidence": conf,
                            "timestamp": time.time()
                        }
                        result_queue.put(payload)
                    mic_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Mic Queue Error: {e}")

            # 2. Check Desktop
            try:
                if not desktop_transcriber.result_queue.empty():
                    text, filename, audio_type, conf = desktop_transcriber.result_queue.get_nowait()
                    should_output, final_text = deduplicator.process(text, "desktop")
                    
                    if should_output:
                        payload = {
                            "type": "transcript",
                            "source": "desktop",
                            "audio_type": audio_type, # speech or music
                            "text": final_text,
                            "confidence": conf,
                            "timestamp": time.time()
                        }
                        result_queue.put(payload)
                    desktop_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Desktop Queue Error: {e}")

            time.sleep(0.05) # Prevent CPU spinning

    except Exception as e:
        print(f"❌ [SENSES] Critical Transcription Failure: {e}")
        traceback.print_exc()
    finally:
        print("🛑 [SENSES] Stopping Transcription Engines...")
        try:
            mic_transcriber.stop_event.set()
            desktop_transcriber.stop_event.set()
        except:
            pass

class TranscriptionService:
    def __init__(self):
        self.process = None
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

    def start(self):
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
        self.stop_event.set()
        if self.process:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
            self.process = None

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break
        return results