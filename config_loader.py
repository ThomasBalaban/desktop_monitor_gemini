"""
Configuration loader for Screen Watcher
"""

class ConfigLoader:
    """Handles loading and providing configuration values"""
    
    def __init__(self):
        self.api_key = ""
        self.capture_region = None
        self.fps = 2
        self.image_quality = 85
        self.prompt = ""
        self.safety_settings = None
        self.max_output_tokens = 500
        self.debug_mode = False
        
        # New Audio Config
        self.audio_sample_rate = 16000 # Default fallback
        self.audio_device_id = 0
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config.py"""
        try:
            import config
            self.api_key = getattr(config, 'API_KEY', "")
            self.capture_region = getattr(config, 'CAPTURE_REGION', None)
            self.fps = getattr(config, 'FPS', 2)
            self.image_quality = getattr(config, 'IMAGE_QUALITY', 85)
            self.prompt = getattr(config, 'PROMPT', "")
            self.safety_settings = getattr(config, 'SAFETY_SETTINGS', None)
            self.max_output_tokens = getattr(config, 'MAX_OUTPUT_TOKENS', 500)
            self.debug_mode = getattr(config, 'DEBUG_MODE', False)
            
            # Load Audio Settings
            self.audio_sample_rate = getattr(config, 'AUDIO_SAMPLE_RATE', 16000)
            self.audio_device_id = getattr(config, 'DESKTOP_AUDIO_DEVICE_ID', 0)
            
        except ImportError:
            print("Warning: config.py not found. Using default settings.")
    
    def is_api_key_configured(self):
        return bool(self.api_key)
    
    def get_region_description(self):
        if self.capture_region:
            return f"{self.capture_region['width']}x{self.capture_region['height']} at ({self.capture_region['left']}, {self.capture_region['top']})"
        return "Will select during startup"