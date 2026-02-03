from api_keys import GEMINI_API_KEY, OPENAI_API_KEY

# API Keys
API_KEY = GEMINI_API_KEY
OPENAI_API_KEY = OPENAI_API_KEY

# Debug Mode
DEBUG_MODE = False

# --- Audio Configuration ---
DESKTOP_AUDIO_DEVICE_ID = 2
MICROPHONE_DEVICE_ID = 3
AUDIO_SAMPLE_RATE = 16000

# --- Vision Configuration ---
CAPTURE_REGION = {
    "left": 14,
    "top": 154,
    "width": 1222,
    "height": 685
}

# Capture Settings
VIDEO_DEVICE_INDEX = 1
FPS = 2
IMAGE_QUALITY = 85
MAX_OUTPUT_TOKENS = 600

# Pulse interval - how often Gemini analyzes (in seconds)
PULSE_INTERVAL = 4.0

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- UNIFIED PROMPT (Vision + Audio) ---
PROMPT = """You are an expert scene analyzer providing real-time context for an AI assistant. You receive both video frames and audio transcriptions from the screen.

YOUR JOB: Combine what you SEE and what you HEAR into a unified description of what's happening on screen.

AUDIO TRANSCRIPTS (from the last few seconds):
{audio_transcripts}

ANALYSIS RULES:

1. MATCH AUDIO TO VISUALS: When you see a character and hear dialogue, connect them. Example: "Charlie (blonde girl on screen) is singing 'Inside of every demon is a rainbow'"

2. IDENTIFY SPEAKERS: Use visual cues to identify who is speaking or singing:
   - If you recognize the character, use their name
   - If not, describe them: "Pink-haired girl", "Man in red suit", "Female voice (off-screen)"

3. AUDIO TYPES: Distinguish between:
   - Character dialogue/singing (attribute to speaker)
   - Background music (describe mood/style)
   - Sound effects (describe what you hear)

4. KEEP IT CONCISE: One short paragraph combining everything. The AI needs quick context, not a novel.

OUTPUT FORMAT:
Write a natural paragraph describing the scene. Include who's speaking/singing and what they said, what's visually happening, and any notable audio (music, SFX). Keep it under 100 words.

EXAMPLE OUTPUT:
"Charlie (blonde girl in white dress) is singing excitedly 'Inside of every demon is a rainbow!' while Vaggie stands behind her looking skeptical. Upbeat piano music playing. The hotel lobby is brightly lit with other demons watching in the background."

NOW ANALYZE THE CURRENT SCENE:"""


# Old settings from previous app (kept for reference)
FS = 16000
CHUNK_DURATION = 5
OVERLAP = 1.5
MAX_THREADS = 4
SAVE_DIR = "audio_captures"
DESKTOP_DEVICE_ID = 2
MODEL_SIZE = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"