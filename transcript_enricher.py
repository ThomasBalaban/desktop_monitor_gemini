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
        
    def enrich(self, raw_transcript):
        """Queue a raw transcript for enrichment."""
        if not raw_transcript or len(raw_transcript.strip()) < 2:
            return
            
        timestamp = time.time() - self.session_start
        
        with self.lock:
            self.queue.append({
                "text": raw_transcript,
                "timestamp": timestamp,
                "visual_context": self.visual_context
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
                        self.on_enriched_transcript(enriched)
                except Exception as e:
                    print(f"⚠️ Enrichment error: {e}")
                    # Fall back to raw transcript
                    if self.on_enriched_transcript:
                        ts = self._format_timestamp(item["timestamp"])
                        self.on_enriched_transcript(f"[{ts}] {item['text']}")
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
        
        prompt = f"""You are a professional transcript formatter for animated shows, music videos, and video content.

CURRENT VISUAL CONTEXT (what's on screen right now):
{visual}

{speaker_history}

{history}

RAW AUDIO TRANSCRIPTION (just captured):
"{raw_text}"

TIMESTAMP: [{timestamp}]

YOUR TASK: Transform the raw transcription into a richly formatted transcript line.

SPEAKER IDENTIFICATION RULES:
1. If you can identify a CHARACTER NAME from the visual context, use it (e.g., "Charlie:", "Alastor:", "Cherri Bomb:")
2. If you DON'T know the character name, use DESCRIPTIVE LABELS with consistent numbering:
   - "Female Singer 1:", "Female Singer 2:" (for different female voices)
   - "Male Voice 1:", "Male Voice 2:" (for different male voices)
   - "Girl (blonde):", "Woman (red dress):" (use visual descriptions if visible)
   - "Narrator:", "Announcer:", "Chorus:" (for special roles)
3. KEEP SPEAKERS CONSISTENT - if "Female Singer 1" was used before for the same voice, use it again
4. Add distinguishing details in the label when possible: "Female Voice (raspy):", "Male Singer (deep baritone):"

FORMAT RULES:
1. Add vocal/emotional tone in parentheses after the speaker: (singing softly), (shouting), (whispering), (crying), etc.
2. If it's sung lyrics, describe the singing style
3. For sound effects, use [SFX: description] - explosions, doors, footsteps, etc.
4. For notable music changes, use [Music: description] - tempo changes, instrument shifts, mood changes
5. Put actual dialogue/lyrics in quotes

OUTPUT FORMAT EXAMPLES:
[{timestamp}] Charlie: (singing hopefully) "Inside of every demon is a rainbow!"
[{timestamp}] Female Singer 1: (belting powerfully) "I can hear you calling!"
[{timestamp}] Male Voice (gravelly): (speaking menacingly) "You shouldn't have come here."
[{timestamp}] [SFX: Glass shattering] [Music: Drums kick in aggressively]
[{timestamp}] [SFX: Explosion] Girl (pink hair): (screaming) "Watch out!"

CRITICAL:
- Output ONLY the single formatted line, nothing else
- Keep the original words/lyrics, just add formatting
- Be consistent with speaker labels across the session
- If multiple things happen, combine them on one line

OUTPUT:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a transcript formatter. Output only the formatted line, nothing else. Be consistent with speaker identification."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            
            enriched = response.choices[0].message.content.strip()
            
            # Try to extract and track speaker for consistency
            self._track_speaker(enriched)
            
            # Add to history for continuity
            self.recent_transcripts.append(enriched)
            if len(self.recent_transcripts) > self.max_history:
                self.recent_transcripts.pop(0)
            
            return enriched
            
        except Exception as e:
            print(f"⚠️ GPT-4o enrichment failed: {e}")
            return f"[{timestamp}] {raw_text}"
    
    def _track_speaker(self, enriched_line):
        """Extract and track speaker labels for consistency."""
        # Try to find speaker pattern: [timestamp] Speaker: 
        import re
        match = re.search(r'\[\d+:\d+\]\s*(?:\[.*?\]\s*)?([^:(]+?)(?:\s*\([^)]+\))?:', enriched_line)
        if match:
            speaker = match.group(1).strip()
            # Only track generic speakers (not character names which are already consistent)
            if any(x in speaker.lower() for x in ['female', 'male', 'voice', 'singer', 'girl', 'boy', 'woman', 'man']):
                # Create a simplified key
                key = speaker.lower()
                if key not in self.known_speakers:
                    self.known_speakers[key] = speaker