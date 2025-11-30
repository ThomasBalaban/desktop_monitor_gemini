from api_keys import GEMINI_API_KEY

# API Key
API_KEY = GEMINI_API_KEY

# Debug Mode
DEBUG_MODE = False

# --- Audio Configuration ---
# Cam Link 4K ID
DESKTOP_AUDIO_DEVICE_ID = 4  
# MATCHED TO YOUR DEVICE (Crucial Fix)
AUDIO_SAMPLE_RATE = 48000

# --- Vision Configuration ---
CAPTURE_REGION = {
    "left": 14,
    "top": 154,
    "width": 1222,
    "height": 685
}

# Capture Settings
FPS = 2                
IMAGE_QUALITY = 85     
MAX_OUTPUT_TOKENS = 500

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- SYSTEM PROMPT ---
# Using the safe version to ensure connection stability
PROMPT = """
ROLE: You are an AI Observer watching a live screen and listening to audio.
CONTEXT: The audio is from a direct feed (Cam Link).

INSTRUCTIONS:
1. CONTINUOUS WATCH: You will receive a stream of images and audio.
2. SILENT MODE: Do NOT output text for every frame. I will signal when I want you to speak.
3. TRIGGER EVENTS: When I signal "turn_complete":
   - If you see/hear something new or significant, describe it briefly.
   - If nothing has changed, output exactly: [WAIT]
   - If I asked a question in the context, answer it.

GUIDELINES:
- Describe visual and audio events factually.
- Use direct, casual descriptive language.
"""