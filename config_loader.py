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
        self.prompt = "Watch this screen region and describe what you see. Alert me of any significant changes or interesting activity."
        self.safety_settings = None  # Default to Gemini's built-in safety
        self.max_output_tokens = 500  # Limit response length
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config.py"""
        try:
            import config
            self.api_key = getattr(config, 'API_KEY', "")
            self.capture_region = getattr(config, 'CAPTURE_REGION', None)
            self.fps = getattr(config, 'FPS', 2)
            self.image_quality = getattr(config, 'IMAGE_QUALITY', 85)
            self.prompt = getattr(config, 'PROMPT', self.prompt)
            self.safety_settings = getattr(config, 'SAFETY_SETTINGS', None)
            self.max_output_tokens = getattr(config, 'MAX_OUTPUT_TOKENS', 500)
        except ImportError:
            print("Warning: config.py not found. Using default settings.")
    
    def is_api_key_configured(self):
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def is_capture_region_configured(self):
        """Check if capture region is configured"""
        return self.capture_region is not None
    
    def get_region_description(self):
        """Get human-readable description of capture region"""
        if self.capture_region:
            return f"{self.capture_region['width']}x{self.capture_region['height']} at ({self.capture_region['left']}, {self.capture_region['top']})"
        return "Will select during startup"
    
    def get_settings_description(self):
        """Get human-readable description of settings"""
        safety_desc = "Default" if self.safety_settings is None else "Custom"
        return f"FPS: {self.fps}, Quality: {self.image_quality}, Safety: {safety_desc}, Auto-restart: 30s"