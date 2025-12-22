# transcriber_core/config.py
"""
Configuration for the transcription system.
"""

# Audio Settings
FS = 16000  # Sample rate in Hz
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
OVERLAP = 1.5  # Overlap between chunks in seconds
MAX_THREADS = 4  # Maximum number of threads for transcription
SAVE_DIR = "audio_captures"  # Directory for saving audio files

# Device IDs (run helper/sound_devices.py to find yours)
DESKTOP_DEVICE_ID = 4  # BlackHole or similar loopback device
MICROPHONE_DEVICE_ID = 5  # Your microphone (e.g., Scarlett Solo 4th Gen)

# ==============================================================================
# FASTER-WHISPER SETTINGS (for Desktop Audio)
# ==============================================================================
# Model sizes: "tiny", "tiny.en", "base", "base.en", "small", "small.en", 
#              "medium", "medium.en", "large-v2", "large-v3"
# 
# For Apple Silicon Macs, use "cpu" with "int8" for good performance.
# For NVIDIA GPUs, use "cuda" with "float16" for best performance.
# ==============================================================================

WHISPER_MODEL_SIZE = "base.en"  # English-only model, good balance of speed/accuracy
WHISPER_DEVICE = "cpu"          # "cpu" or "cuda" 
WHISPER_COMPUTE_TYPE = "int8"   # "int8", "float16", or "float32"

# ==============================================================================
# PARAKEET SETTINGS (for Microphone - handled by parakeet_mlx automatically)
# ==============================================================================
# Parakeet MLX uses the mlx-community/parakeet-tdt-0.6b-v2 model by default.
# This is optimized for Apple Silicon and provides excellent streaming performance.
# No additional configuration needed - it auto-detects the best settings.
# ==============================================================================