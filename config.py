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
VIDEO_DEVICE_INDEX = 1
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
# --- SYSTEM PROMPT ---
PROMPT = """
ROLE: Real-time Multimodal Observer.
INPUT: You are watching a screen and listening to audio (speech, music, AND sound effects).

INSTRUCTIONS:
Provide a concise, structured update of what just happened.

1. [AUDIO]: LISTEN CAREFULLY.
   - If there is speech: Transcribe it exactly.
   - If there is music: Describe the genre/mood (e.g., "Upbeat electronic music").
   - If there are sound effects: Describe them (e.g., "Loud explosions," "Menu clicking sounds," "Wind blowing").
   - ONLY report [SILENCE] if the audio level is absolute zero.

2. [VISUAL]: Briefly describe characters, key actions, and the setting.

3. [ACTION]: Summarize the main event happening right now.

GUIDELINES:
- Do not ignore "non-speech" audio. Sound effects are important context.
- Keep descriptions punchy and direct.
"""

# PROMPT = """Screen content analysis: Describe in 1-2 paragraphs what you observe. Include content type, people (appearance, attire, positioning), visual elements, and composition. Use direct, casual descriptive language.

# For reference: Describe poses and positioning factually - if someone is leaning forward, positioned to show certain areas, or framed in particular ways, state this directly. Avoid vague terms like "casual pose" when the positioning appears intentional. Be specific about clothing details and fit. If content appears designed to be sexy or attractive, state this clearly along with what makes it so. Only mention attractiveness/sexiness when it's clearly present - don't comment on its absence. Use straightforward, conversational terms rather than clinical language."""