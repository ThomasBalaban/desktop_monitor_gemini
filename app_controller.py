from datetime import datetime
from config_loader import ConfigLoader
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from app_gui import AppGUI
from websocket_server import WebSocketServer, WEBSOCKET_PORT
import tkinter as tk 

class AppController:
    """Main application class that connects the GUI and backend services."""

    def __init__(self):
        self.config = ConfigLoader()
        print("Gemini Screen Watcher - Starting up in GUI mode...")

        # Initialize core components
        self.screen_capture = ScreenCapture(self.config.image_quality)
        self.gemini_client = GeminiClient(
            self.config.api_key, self.config.prompt, self.config.safety_settings,
            self._on_gemini_response, self.config.max_output_tokens, self.config.debug_mode
        )
        self.streaming_manager = StreamingManager(
            self.screen_capture, self.gemini_client, self.config.fps,
            restart_interval=30, debug_mode=self.config.debug_mode
        )
        
        self.streaming_manager.set_restart_callback(self.on_stream_restart)
        
        self.websocket_server = WebSocketServer()
        self.current_response_buffer = ""
        self.gui = AppGUI(self)
        self._initialize_capture_region()

    def run(self):
        """Starts the application by running the GUI main loop."""
        if not self.config.is_api_key_configured():
            self.gui.update_status("ERROR: API_KEY not configured in config.py", "red")
            print("ERROR: API_KEY is not configured in config.py. Please set it and restart.")
        
        self.websocket_server.start()
        self.gui.run()

    def update_websocket_gui_status(self):
        """Called by the GUI after it has started to update its status label."""
        self.gui.update_websocket_status(f"Running at ws://localhost:{WEBSOCKET_PORT}", "#4CAF50")

    def on_stream_restart(self):
        """Callback to notify the GUI that the stream is restarting."""
        self.gui.add_reset_separator()

    def _initialize_capture_region(self):
        """Sets the screen capture region from config, or handles the missing case."""
        if self.config.capture_region:
            self.screen_capture.set_capture_region(self.config.capture_region)
            print(f"Using capture region from config: {self.config.get_region_description()}")
        else:
            # --- FIX: Handle the case where no region is configured more gracefully ---
            error_msg = "No capture region set in config.py"
            self.gui.update_status(error_msg, "red")
            print(f"ERROR: {error_msg}")
            # Disable the start button since we can't proceed
            self.gui.start_button.config(state=tk.DISABLED)

    # --- FIX: Corrected method signature from (self.text_chunk) to (self, text_chunk) ---
    def _on_gemini_response(self, text_chunk):
        """Handles incoming text from Gemini and updates the GUI and WebSocket clients."""
        self.current_response_buffer += text_chunk
        
        if self.current_response_buffer.strip().endswith(('.', '!', '?')):
            final_text = self.current_response_buffer.strip()
            
            self.gui.add_response(final_text)
            
            response_data = {
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": final_text
            }
            self.websocket_server.broadcast(response_data)
            self.current_response_buffer = ""

    def start_streaming(self):
        """Callback for the 'Start' button."""
        if not self.screen_capture.capture_region:
            self.gui.update_status("Cannot start. No screen region is configured.", "red")
            return
        
        print("Starting the screen streaming process...")
        self.streaming_manager.start_streaming()
        self.gui.update_status("Connecting...", "orange")
        self.gui.update_button_states(is_streaming=True)
        self.streaming_manager.set_status_callback(self.gui.update_status)

    def stop_streaming(self):
        """Callback for the 'Stop' button."""
        print("Stopping the screen streaming process...")
        self.streaming_manager.stop_streaming()
        self.gui.update_status("Stopped", "red")
        self.gui.update_button_states(is_streaming=False)

    def get_prompt(self):
        return self.config.prompt

    def get_timestamp(self):
        return datetime.now().strftime("%I:%M:%S %p")