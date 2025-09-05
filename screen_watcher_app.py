import asyncio
import json
from datetime import datetime

from config_loader import ConfigLoader
from gemini_client import GeminiClient
from screen_capture import ScreenCapture
from streaming_manager import StreamingManager
from web_and_socket_server import WebAndSocketServer

class ScreenWatcherApp:
    """Main application class - runs as a headless server."""

    def __init__(self):
        """Initializes the application components and server."""
        self.config = ConfigLoader()
        print("Gemini Screen Watcher - Starting up in server mode...")

        # Initialize core components
        self.screen_capture = ScreenCapture(self.config.image_quality)
        self._initialize_capture_region()

        # Pass the config loader to the server so it can serve config details
        self.server = WebAndSocketServer(config_loader=self.config)

        self.gemini_client = GeminiClient(
            self.config.api_key,
            self.config.prompt,
            self.config.safety_settings,
            self._on_gemini_response,
            self.config.max_output_tokens,
            self.config.debug_mode
        )
        
        self.streaming_manager = StreamingManager(
            self.screen_capture,
            self.gemini_client,
            self.config.fps,
            restart_interval=30,
            debug_mode=self.config.debug_mode
        )
        
        # Buffer to accumulate text chunks from Gemini
        self.current_response_buffer = ""

    def _initialize_capture_region(self):
        """Sets the screen capture region, prompting user if not configured."""
        if self.config.capture_region:
            self.screen_capture.set_capture_region(self.config.capture_region)
            print(f"Using capture region from config: {self.config.get_region_description()}")
        else:
            print("No capture region in config. Please select a region interactively.")
            # A temporary Tkinter root is needed for the interactive selection
            import tkinter as tk
            root = tk.Tk()
            self.screen_capture.select_region_interactive(root)
            root.destroy()
            
            if not self.screen_capture.capture_region:
                print("ERROR: No screen region was selected. Exiting.")
                exit() # Exit if no region is selected
            print(f"Region selected: {self.config.get_region_description()}")


    def _on_gemini_response(self, text_chunk):
        """
        Handles incoming text from Gemini, buffers it, and broadcasts complete paragraphs.
        """
        self.current_response_buffer += text_chunk
        
        # We assume a response is "complete" when it ends with a common punctuation mark.
        # This helps ensure we send full paragraphs instead of fragmented sentences.
        if self.current_response_buffer.strip().endswith(('.', '!', '?')):
            
            # Prepare a simple JSON object to send to clients
            response_data = {
                "type": "text_update",
                "timestamp": datetime.now().isoformat(),
                "content": self.current_response_buffer.strip()
            }
            
            # The server runs in a different thread with its own asyncio event loop.
            # We must use `run_coroutine_threadsafe` to safely schedule the broadcast.
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self.server.broadcast(response_data),
                loop
            )
            
            # Clear the buffer for the next response
            self.current_response_buffer = ""

    def run(self):
        """Starts the screen streaming process and the web/socket server."""
        if not self.config.is_api_key_configured():
            print("ERROR: API_KEY is not configured in config.py. Please set it and restart.")
            return

        # Start capturing the screen and streaming to Gemini
        print("Starting the screen streaming process...")
        self.streaming_manager.start_streaming()

        # The `run` method of the server is a blocking call that starts both
        # the HTTP server for the UI and the WebSocket server for AI clients.
        self.server.run()
