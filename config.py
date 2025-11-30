from api_keys import GEMINI_API_KEY

# API Key
API_KEY = GEMINI_API_KEY

# Debug Mode - set to True to see detailed console messages
DEBUG_MODE = False

# --- Audio Configuration ---
# Use the same ID as your audio_mon app (likely 4 or 5 based on your files)
DESKTOP_AUDIO_DEVICE_ID = 4  
AUDIO_SAMPLE_RATE = 16000

# --- Vision Configuration ---
# Set to None to use GUI selection, or specify coordinates:
CAPTURE_REGION = {
    "left": 14,
    "top": 154,
    "width": 1222,
    "height": 685
}

# Capture Settings
FPS = 2                # Frames per second (1-10)
IMAGE_QUALITY = 85     # JPEG quality (50-100, higher = better quality)
MAX_OUTPUT_TOKENS = 500

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- SYSTEM PROMPT (Updated for Silent Stream) ---
# --- SYSTEM PROMPT ---
# Optimized for concise details, dialogue transcription, and specific visuals.
PROMPT = """
ROLE: Real-time Multimodal Observer.
INPUT: You are watching a screen and listening to audio (speech/music).

INSTRUCTIONS:
Provide a concise, structured update of what just happened. Avoid long paragraphs. Use the following format:

1. [AUDIO/DIALOGUE]: Transcribe exactly what was said or sung. If it's music, describe the genre/mood. If silence, say [SILENCE].
2. [CHARACTERS]: Briefly list who is visible. Describe their specific appearance (clothing, colors, accessories) and distinctive features.
3. [ACTION]: Describe EXACTLY what they are doing (e.g., "holding hands," "spinning," "looking distressed"). Be specific about body language.

GUIDELINES:
- Do not summarize; report facts.
- If you hear speech, prioritize transcribing it.
- Keep descriptions punchy and direct.
"""

# PROMPT = """Screen content analysis: Describe in 1-2 paragraphs what you observe. Include content type, people (appearance, attire, positioning), visual elements, and composition. Use direct, casual descriptive language.

# For reference: Describe poses and positioning factually - if someone is leaning forward, positioned to show certain areas, or framed in particular ways, state this directly. Avoid vague terms like "casual pose" when the positioning appears intentional. Be specific about clothing details and fit. If content appears designed to be sexy or attractive, state this clearly along with what makes it so. Only mention attractiveness/sexiness when it's clearly present - don't comment on its absence. Use straightforward, conversational terms rather than clinical language."""