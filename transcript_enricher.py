import asyncio
import json
import time
import threading
from openai import OpenAI

class TranscriptEnricher:
    """
    Takes raw Whisper transcriptions and enriches them with:
    - Speaker identification (character names OR descriptive labels)
    - Sound effect labels [SFX: ...]
    - Music descriptions [Music: ...]
    - Emotional/vocal tone descriptions
    - Timestamps
    """
    
    def __init__(self, api_key, on_enriched_transcript=None):
        self.client = OpenAI(api_key=api_key)
        self.on_enriched_transcript = on_enriched_transcript
        
        # Context buffers
        self.visual_context = ""  # What Gemini sees on screen
        self.recent_transcripts = []  # Last few transcripts for continuity
        self.max_history = 8
        
        # Speaker tracking - persists across transcripts
        self.known_speakers = {}  # Maps descriptions to consistent labels
        self.speaker_counter = {"female": 0, "male": 0, "unknown": 0}
        
        # Track timing
        self.session_start = time.time()
        
        # Processing queue
        self.queue = []
        self.lock = threading.Lock()
        self.running = False
        self.process_thread = None
        
    def start(self):
        """Start the enrichment processor."""
        self.running = True
        self.session_start = time.time()
        self.known_speakers = {}
        self.speaker_counter = {"female": 0, "male": 0, "unknown": 0}
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        print("🎭 Transcript Enricher started (with speaker tracking)")
        
    def stop(self):
        """Stop the enrichment processor."""
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
            
    def update_visual_context(self, context):
        """Update what's currently visible on screen (from Gemini)."""
        self.visual_context = context
        
    def enrich(self, raw_transcript, transcript_id=None):
        """Queue a raw transcript for enrichment."""
        if not raw_transcript or len(raw_transcript.strip()) < 2:
            return
            
        timestamp = time.time() - self.session_start
        
        with self.lock:
            self.queue.append({
                "text": raw_transcript,
                "timestamp": timestamp,
                "visual_context": self.visual_context,
                "id": transcript_id
            })
    
    def _process_loop(self):
        """Background loop that processes queued transcripts."""
        while self.running:
            item = None
            with self.lock:
                if self.queue:
                    item = self.queue.pop(0)
            
            if item:
                try:
                    enriched = self._enrich_transcript(item)
                    if enriched and self.on_enriched_transcript:
                        self.on_enriched_transcript(enriched, item.get("id"))
                except Exception as e:
                    print(f"⚠️ Enrichment error: {e}")
                    # Fall back to raw transcript
                    if self.on_enriched_transcript:
                        ts = self._format_timestamp(item["timestamp"])
                        self.on_enriched_transcript(f"[{ts}] {item['text']}", item.get("id"))
            else:
                time.sleep(0.1)
    
    def _format_timestamp(self, seconds):
        """Format seconds as M:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    def _get_speaker_history(self):
        """Format known speakers for context."""
        if not self.known_speakers:
            return "No speakers identified yet."
        
        lines = ["Previously identified speakers:"]
        for desc, label in self.known_speakers.items():
            lines.append(f"  - {label}: {desc}")
        return "\n".join(lines)
    
    def _enrich_transcript(self, item):
        """Use GPT-4o to enrich a single transcript."""
        
        raw_text = item["text"]
        timestamp = self._format_timestamp(item["timestamp"])
        visual = item["visual_context"] or "No visual context available"
        
        # Build context from recent transcripts
        history = ""
        if self.recent_transcripts:
            history = "Recent transcript history (for continuity):\n"
            for h in self.recent_transcripts[-5:]:
                history += f"{h}\n"
        
        speaker_history = self._get_speaker_history()
        
        prompt = f"""You are a professional transcript formatter.

CURRENT VISUAL CONTEXT:
{visual}

{speaker_history}

{history}

RAW AUDIO:
"{raw_text}"

TIMESTAMP: [{timestamp}]

TASK: Format the raw transcription.
- Identify SPEAKER (Character Name if known from visuals, else "Male Voice 1", etc.)
- Add TONE in parentheses: (sarcastic), (whispering)
- Add SFX/Music if implied.
- Output ONLY the formatted line.

OUTPUT:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a transcript formatter. Output only the formatted line."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            
            enriched = response.choices[0].message.content.strip()
            self._track_speaker(enriched)
            
            self.recent_transcripts.append(enriched)
            if len(self.recent_transcripts) > self.max_history:
                self.recent_transcripts.pop(0)
            
            return enriched
            
        except Exception as e:
            print(f"⚠️ GPT-4o enrichment failed: {e}")
            return f"[{timestamp}] {raw_text}"
    
    def _track_speaker(self, enriched_line):
        import re
        match = re.search(r'\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:', enriched_line)
        if match:
            speaker = match.group(1).strip()
            if any(x in speaker.lower() for x in ['female', 'male', 'voice', 'singer', 'girl', 'boy', 'woman', 'man']):
                key = speaker.lower()
                if key not in self.known_speakers:
                    self.known_speakers[key] = speaker