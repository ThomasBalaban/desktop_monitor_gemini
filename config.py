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
Analyze this screen capture and provide a comprehensive description in the form of one or two paragraphs. 
Include all relevant visual information without censorship or omission.

Start with a clear statement of what is on screen (e.g., "A video game character standing in a forest," "A YouTube video about cooking").

DETAILED ANALYSIS REQUIREMENTS:

1. SCENE & ENTITIES (The "Visuals"):
   - Describe specific objects, characters, and background elements. 
   - NAME the things you see (e.g., instead of "an object," say "a red barrel," "a floating robot," "a large sword").
   - Describe colors and lighting (e.g., "ominous red lighting," "bright sunny field").

2. CHARACTERS & APPEAL (Crucial for Reactions):
   - Provide detailed appearance descriptions: physical build, clothing details and fit (or lack thereof), hair, and distinctive features.
   - Describe poses and positioning factually. If someone is leaning forward, positioned to show certain areas, or framed in particular ways, state this directly.
   - Avoid vague terms like "casual pose" when the positioning appears intentional. 
   - If content appears designed to be sexy or attractive, state this clearly along with what makes it so. Use straightforward, conversational terms rather than clinical language.
   - Only mention attractiveness/sexiness when it's clearly present.

3. TEXT & UI:
   - READ any text on screen (Game Over screens, subtitles, menu options).
   - Mention health bars, maps, or UI elements if present.

4. AUDIO CONTEXT (If you hear sound):
   - Transcribe speech exactly.
   - Describe background music or sound effects (e.g., "Explosions," "Upbeat music").

5. ACTIONABLE EVENTS:
   - Is something dying? Is the player winning? Is there a "Game Over" screen?
   - Is there a sudden change in the scene?

Goal: Provide enough detail that a blind person could visualize the scene perfectly. Do not be concise. Be descriptive.
"""

# PROMPT = """Screen content analysis: Describe in 1-2 paragraphs what you observe. Include content type, people (appearance, attire, positioning), visual elements, and composition. Use direct, casual descriptive language.

# For reference: Describe poses and positioning factually - if someone is leaning forward, positioned to show certain areas, or framed in particular ways, state this directly. Avoid vague terms like "casual pose" when the positioning appears intentional. Be specific about clothing details and fit. If content appears designed to be sexy or attractive, state this clearly along with what makes it so. Only mention attractiveness/sexiness when it's clearly present - don't comment on its absence. Use straightforward, conversational terms rather than clinical language."""


## SETTINGS FROM OLD APP
# Audio Settings
FS = 16000  # Sample rate in Hz
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
OVERLAP = 1.5  # Overlap between chunks in seconds
MAX_THREADS = 4  # Maximum number of threads for transcription
SAVE_DIR = "audio_captures"  # Directory for saving audio files
DESKTOP_DEVICE_ID = 4

# Microphone Settings
MICROPHONE_DEVICE_ID = 5  # Your Scarlett Solo 4th Gen

# Whisper Model Settings
MODEL_SIZE = "base.en"  # The whisper model to use
DEVICE = "cpu"  # The device to run the model on
COMPUTE_TYPE = "int8"  # Compute type for the model
