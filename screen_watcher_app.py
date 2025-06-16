"""
Main application class for Screen Watcher
"""

import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time

from config_loader import ConfigLoader
from screen_capture import ScreenCapture
from gemini_client import GeminiClient
from streaming_manager import StreamingManager

class ScreenWatcherApp:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Screen Watcher")
        self.root.geometry("500x400")
        
        # Load configuration
        self.config = ConfigLoader()
        
        # Initialize components
        self.screen_capture = ScreenCapture(self.config.image_quality)
        if self.config.capture_region:
            self.screen_capture.set_capture_region(self.config.capture_region)
        
        self.gemini_client = GeminiClient(
            self.config.api_key,
            self.config.prompt,
            self.config.safety_settings,
            self._on_gemini_response
        )
        
        self.streaming_manager = StreamingManager(
            self.screen_capture,
            self.gemini_client,
            self.config.fps,
            restart_interval=30
        )
        
        # Set up callbacks
        self.streaming_manager.set_status_callback(self._update_status)
        self.streaming_manager.set_restart_callback(self._on_restart)
        
        # UI state
        self.auto_scroll = tk.BooleanVar(value=True)
        self.user_scrolled = False
        self.current_response = ""  # Accumulate response chunks
        self.response_start_time = None
        
        # Setup UI
        self._setup_ui()
        
        # Start countdown timer
        self._update_restart_countdown()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration display
        self._setup_config_frame(main_frame)
        
        # Screen region selection (if not configured)
        if not self.config.is_capture_region_configured():
            self._setup_region_selection_frame(main_frame)
        
        # Control buttons
        self._setup_control_buttons(main_frame)
        
        # Status display
        self._setup_status_frame(main_frame)
        
        # Response display
        self._setup_response_frame(main_frame)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def _setup_config_frame(self, parent):
        """Setup configuration display frame"""
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # API Key section
        ttk.Label(config_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        api_status = "✓ Configured" if self.config.is_api_key_configured() else "⚠ Not set"
        api_color = "green" if self.config.is_api_key_configured() else "red"
        ttk.Label(config_frame, text=api_status, foreground=api_color).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Region section
        ttk.Label(config_frame, text="Screen Region:").grid(row=1, column=0, sticky=tk.W, pady=2)
        region_text = self.config.get_region_description()
        region_color = "green" if self.config.is_capture_region_configured() else "orange"
        self.region_label = ttk.Label(config_frame, text=region_text, foreground=region_color)
        self.region_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Settings
        ttk.Label(config_frame, text="Settings:").grid(row=2, column=0, sticky=tk.W, pady=2)
        settings_text = self.config.get_settings_description()
        ttk.Label(config_frame, text=settings_text).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Prompt display
        ttk.Label(config_frame, text="Prompt:").grid(row=3, column=0, sticky=(tk.W, tk.N), pady=2)
        prompt_display = tk.Text(config_frame, height=3, width=50, wrap=tk.WORD)
        prompt_display.insert("1.0", self.config.prompt)
        prompt_display.config(state=tk.DISABLED)
        prompt_display.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Edit configuration button
        ttk.Button(config_frame, text="Edit config.py", command=self._show_config_help).grid(row=4, column=0, columnspan=3, pady=10)
        
        config_frame.columnconfigure(1, weight=1)
    
    def _setup_region_selection_frame(self, parent):
        """Setup screen region selection frame"""
        region_frame = ttk.LabelFrame(parent, text="Screen Region Selection", padding="10")
        region_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(region_frame, text="No region configured in config.py").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.region_button = ttk.Button(region_frame, text="Select Region Now", command=self._select_screen_region)
        self.region_button.grid(row=0, column=1, sticky=tk.W, pady=5)
    
    def _setup_control_buttons(self, parent):
        """Setup control buttons frame"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Watching", command=self._start_streaming)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_streaming, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Test Connection", command=self._test_connection).pack(side=tk.LEFT, padx=5)
    
    def _setup_status_frame(self, parent):
        """Setup status display frame"""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(status_frame, text="Connection:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Restart countdown
        ttk.Label(status_frame, text="Next restart:").grid(row=1, column=0, sticky=tk.W)
        self.restart_label = ttk.Label(status_frame, text="Not running", foreground="gray")
        self.restart_label.grid(row=1, column=1, sticky=tk.W)
    
    def _setup_response_frame(self, parent):
        """Setup response display frame"""
        response_frame = ttk.LabelFrame(parent, text="Gemini Response", padding="10")
        response_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Response text with scrollbar
        text_frame = ttk.Frame(response_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.response_text = tk.Text(text_frame, height=8, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.response_text.yview)
        self.response_text.configure(yscrollcommand=scrollbar.set)
        
        self.response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind scroll events to detect user scrolling
        self.response_text.bind('<Button-4>', self._on_user_scroll)  # Mouse wheel up
        self.response_text.bind('<Button-5>', self._on_user_scroll)  # Mouse wheel down
        self.response_text.bind('<Prior>', self._on_user_scroll)     # Page up
        self.response_text.bind('<Next>', self._on_user_scroll)      # Page down
        self.response_text.bind('<Up>', self._on_user_scroll)        # Arrow up
        self.response_text.bind('<Down>', self._on_user_scroll)      # Arrow down
        self.response_text.bind('<Key>', self._on_user_scroll)       # Any key press
        
        # Response controls
        response_controls = ttk.Frame(response_frame)
        response_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(response_controls, text="Clear", command=self._clear_responses).pack(side=tk.LEFT)
        
        # Auto-scroll checkbox
        auto_scroll_check = ttk.Checkbutton(response_controls, text="Auto-scroll", 
                                          variable=self.auto_scroll, command=self._toggle_auto_scroll)
        auto_scroll_check.pack(side=tk.LEFT, padx=10)
        
        # Jump to bottom button
        self.bottom_button = ttk.Button(response_controls, text="Jump to Bottom", 
                                       command=self._scroll_to_bottom)
        self.bottom_button.pack(side=tk.LEFT, padx=5)
    
    def _show_config_help(self):
        """Show help for editing config.py"""
        help_text = """To configure the application, edit the config.py file:

1. Set your API_KEY: 
   API_KEY = "your_actual_api_key_here"

2. Configure screen region (optional):
   CAPTURE_REGION = {
       "left": 100, "top": 100, 
       "width": 800, "height": 600
   }
   Or set to None to select visually each time.

3. Adjust settings:
   FPS = 2              # Frames per second
   IMAGE_QUALITY = 85   # JPEG quality (50-100)

4. Configure safety settings (optional):
   # Use default Gemini safety (recommended):
   SAFETY_SETTINGS = None
   
   # Or reduce safety restrictions:
   SAFETY_SETTINGS = [
       {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
   ]
   
   # Other thresholds: BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH

5. The prompt is already set and doesn't need changing.

6. Restart the application after making changes.

Note: Using BLOCK_NONE may require Google's review for your API key.
"""
        messagebox.showinfo("Configuration Help", help_text)
    
    def _select_screen_region(self):
        """Allow user to select a screen region"""
        selected_region = self.screen_capture.select_region_interactive(self.root)
        if selected_region:
            # Update display
            region_text = f"{selected_region['width']}x{selected_region['height']} at ({selected_region['left']}, {selected_region['top']})"
            self.region_label.config(text=region_text, foreground="green")
    
    def _test_connection(self):
        """Test connection to Gemini API"""
        if not self.config.is_api_key_configured():
            messagebox.showerror("Error", "Please set your API key in config.py first")
            return
            
        def test_api():
            success, message = self.gemini_client.test_connection()
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Success", message))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", message))
                
        threading.Thread(target=test_api, daemon=True).start()
    
    def _start_streaming(self):
        """Start the streaming process"""
        if not self.config.is_api_key_configured():
            messagebox.showerror("Error", "Please set your API key in config.py first")
            return
            
        if not self.screen_capture.capture_region:
            messagebox.showerror("Error", "Please select a screen region first")
            return
        
        if self.streaming_manager.start_streaming():
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
    
    def _stop_streaming(self):
        """Stop the streaming process"""
        self.streaming_manager.stop_streaming()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", foreground="red")
        self.restart_label.config(text="Not running", foreground="gray")
    
    def _update_status(self, status, color):
        """Update status display"""
        def update_ui():
            self.status_label.config(text=status, foreground=color)
        self.root.after(0, update_ui)
    
    def _on_restart(self):
        """Handle restart notification"""
        def update_ui():
            self.response_text.insert(tk.END, "=== AUTO-RESTART: Restarting streaming session to maintain fresh connection ===\n\n")
            if self.auto_scroll.get() and not self.user_scrolled:
                self.response_text.see(tk.END)
        self.root.after(0, update_ui)
    
    def _on_gemini_response(self, text):
        """Handle response from Gemini - simpler streaming approach"""
        def update_ui():
            # If this is the start of a new response, add timestamp
            if not self.current_response:
                self.response_start_time = time.strftime('%H:%M:%S')
                self.response_text.insert(tk.END, f"[{self.response_start_time}] ")
            
            # Add the text chunk
            self.response_text.insert(tk.END, text)
            self.current_response += text
            
            # If response seems complete, add newlines and reset
            if text.strip().endswith(('.', '!', '?')) and len(self.current_response.strip()) > 50:
                self.response_text.insert(tk.END, "\n\n")
                self.current_response = ""
                self.response_start_time = None
            
            # Only auto-scroll if enabled and user hasn't manually scrolled
            if self.auto_scroll.get() and not self.user_scrolled:
                self.response_text.see(tk.END)
            
        self.root.after(0, update_ui)
    
    def _on_user_scroll(self, event):
        """Detect when user manually scrolls"""
        # Disable auto-scroll when user manually scrolls
        if self.auto_scroll.get():
            self.auto_scroll.set(False)
        self.user_scrolled = True
    
    def _toggle_auto_scroll(self):
        """Toggle auto-scroll behavior"""
        if self.auto_scroll.get():
            # Re-enable auto-scroll and jump to bottom
            self.user_scrolled = False
            self._scroll_to_bottom()
    
    def _scroll_to_bottom(self):
        """Manually scroll to the bottom of the response text"""
        self.response_text.see(tk.END)
        self.auto_scroll.set(True)
        self.user_scrolled = False
    
    def _clear_responses(self):
        """Clear the response text area"""
        self.response_text.delete("1.0", tk.END)
        self.auto_scroll.set(True)
        self.user_scrolled = False
    
    def _update_restart_countdown(self):
        """Update the restart countdown display"""
        remaining = self.streaming_manager.get_time_until_restart()
        
        if remaining is not None:
            if remaining > 0:
                self.restart_label.config(text=f"{remaining:.1f}s", foreground="blue")
            else:
                self.restart_label.config(text="Restarting session...", foreground="orange")
        else:
            self.restart_label.config(text="Not running", foreground="gray")
        
        # Schedule next update
        self.root.after(100, self._update_restart_countdown)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()