import asyncio
import base64
import json
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
from io import BytesIO
import websockets
import mss
from PIL import Image
import cv2
import numpy as np
import os
import sys
import subprocess

# Import configuration
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using default settings.")
    class config:
        API_KEY = ""
        CAPTURE_REGION = None
        FPS = 2
        IMAGE_QUALITY = 85
        PROMPT = "Watch this screen region and describe what you see. Alert me of any significant changes or interesting activity."

class ScreenWatcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Screen Watcher")
        self.root.geometry("500x400")
        
        # Load configuration from config.py
        self.api_key = config.API_KEY
        self.capture_region = config.CAPTURE_REGION
        self.fps = config.FPS
        self.image_quality = config.IMAGE_QUALITY
        self.prompt = config.PROMPT
        
        # Runtime variables
        self.websocket = None
        self.is_streaming = False
        self.restart_interval = 30  # Restart every 30 seconds
        self.session_start_time = None
        self.restart_timer = None
        self.current_tasks = []  # Track running async tasks
        
        # Screen capture
        self.sct = mss.mss()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration display
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # API Key section
        ttk.Label(config_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        api_status = "✓ Configured" if self.api_key else "⚠ Not set"
        api_color = "green" if self.api_key else "red"
        ttk.Label(config_frame, text=api_status, foreground=api_color).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Region section
        ttk.Label(config_frame, text="Screen Region:").grid(row=1, column=0, sticky=tk.W, pady=2)
        if self.capture_region:
            region_text = f"{self.capture_region['width']}x{self.capture_region['height']} at ({self.capture_region['left']}, {self.capture_region['top']})"
            region_color = "green"
        else:
            region_text = "Will select during startup"
            region_color = "orange"
        self.region_label = ttk.Label(config_frame, text=region_text, foreground=region_color)
        self.region_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Settings
        ttk.Label(config_frame, text="Settings:").grid(row=2, column=0, sticky=tk.W, pady=2)
        settings_text = f"FPS: {self.fps}, Quality: {self.image_quality}, Auto-restart: {self.restart_interval}s"
        ttk.Label(config_frame, text=settings_text).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Prompt display
        ttk.Label(config_frame, text="Prompt:").grid(row=3, column=0, sticky=(tk.W, tk.N), pady=2)
        prompt_display = tk.Text(config_frame, height=3, width=50, wrap=tk.WORD)
        prompt_display.insert("1.0", self.prompt)
        prompt_display.config(state=tk.DISABLED)
        prompt_display.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Edit configuration button
        ttk.Button(config_frame, text="Edit config.py", command=self.open_config_help).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Screen region selection (if not configured)
        if not self.capture_region:
            region_frame = ttk.LabelFrame(main_frame, text="Screen Region Selection", padding="10")
            region_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Label(region_frame, text="No region configured in config.py").grid(row=0, column=0, sticky=tk.W, pady=5)
            self.region_button = ttk.Button(region_frame, text="Select Region Now", command=self.select_screen_region)
            self.region_button.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Watching", command=self.start_streaming)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_streaming, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Test Connection", command=self.test_connection).pack(side=tk.LEFT, padx=5)
        
        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(status_frame, text="Connection:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Restart countdown
        ttk.Label(status_frame, text="Next restart:").grid(row=1, column=0, sticky=tk.W)
        self.restart_label = ttk.Label(status_frame, text="Not running", foreground="gray")
        self.restart_label.grid(row=1, column=1, sticky=tk.W)
        
        # Response display
        response_frame = ttk.LabelFrame(main_frame, text="Gemini Response", padding="10")
        response_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Response text with scrollbar
        text_frame = ttk.Frame(response_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.response_text = tk.Text(text_frame, height=8, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.response_text.yview)
        self.response_text.configure(yscrollcommand=scrollbar.set)
        
        self.response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Auto-scroll control variables
        self.auto_scroll = tk.BooleanVar(value=True)
        self.user_scrolled = False
        
        # Bind scroll events to detect user scrolling
        self.response_text.bind('<Button-4>', self.on_user_scroll)  # Mouse wheel up
        self.response_text.bind('<Button-5>', self.on_user_scroll)  # Mouse wheel down
        self.response_text.bind('<Prior>', self.on_user_scroll)     # Page up
        self.response_text.bind('<Next>', self.on_user_scroll)      # Page down
        self.response_text.bind('<Up>', self.on_user_scroll)        # Arrow up
        self.response_text.bind('<Down>', self.on_user_scroll)      # Arrow down
        self.response_text.bind('<Key>', self.on_user_scroll)       # Any key press
        
        # Response controls
        response_controls = ttk.Frame(response_frame)
        response_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(response_controls, text="Clear", command=self.clear_responses).pack(side=tk.LEFT)
        
        # Auto-scroll checkbox
        auto_scroll_check = ttk.Checkbutton(response_controls, text="Auto-scroll", 
                                          variable=self.auto_scroll, command=self.toggle_auto_scroll)
        auto_scroll_check.pack(side=tk.LEFT, padx=10)
        
        # Jump to bottom button
        self.bottom_button = ttk.Button(response_controls, text="Jump to Bottom", 
                                       command=self.scroll_to_bottom)
        self.bottom_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        config_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def open_config_help(self):
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

4. The prompt is already set and doesn't need changing.

5. Restart the application after making changes.
"""
        messagebox.showinfo("Configuration Help", help_text)
        
    def on_user_scroll(self, event):
        """Detect when user manually scrolls"""
        # Disable auto-scroll when user manually scrolls
        if self.auto_scroll.get():
            self.auto_scroll.set(False)
        self.user_scrolled = True
        
    def toggle_auto_scroll(self):
        """Toggle auto-scroll behavior"""
        if self.auto_scroll.get():
            # Re-enable auto-scroll and jump to bottom
            self.user_scrolled = False
            self.scroll_to_bottom()
            
    def scroll_to_bottom(self):
        """Manually scroll to the bottom of the response text"""
        self.response_text.see(tk.END)
        self.auto_scroll.set(True)
        self.user_scrolled = False
        
    def clear_responses(self):
        """Clear the response text area"""
        self.response_text.delete("1.0", tk.END)
        self.auto_scroll.set(True)
        self.user_scrolled = False
        
    def test_connection(self):
        """Test connection to Gemini API"""
        if not self.api_key:
            messagebox.showerror("Error", "Please set your API key in config.py first")
            return
            
        def test_api():
            try:
                import requests
                # Test with Gemini 2.0 Flash model that supports the Live API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
                response = requests.post(url, json={
                    "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
                }, timeout=10)
                
                if response.status_code == 200:
                    self.root.after(0, lambda: messagebox.showinfo("Success", "API connection successful! Gemini 2.0 Flash is ready."))
                elif response.status_code == 404:
                    # Try the regular Gemini Flash model as fallback
                    url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
                    response_fallback = requests.post(url_fallback, json={
                        "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
                    }, timeout=10)
                    
                    if response_fallback.status_code == 200:
                        self.root.after(0, lambda: messagebox.showwarning("Partial Success", 
                            "API key works but Gemini 2.0 Flash may not be available. Live streaming might not work. Check your API access."))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", 
                            f"API key invalid or no access to Gemini models: {response_fallback.status_code}"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"API test failed: {response.status_code} - {response.text}"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Connection test failed: {e}"))
                
        threading.Thread(target=test_api, daemon=True).start()
        
    def select_screen_region(self):
        """Allow user to select a screen region by dragging"""
        self.root.withdraw()  # Hide main window
        
        # Create overlay for region selection
        overlay = tk.Toplevel()
        overlay.attributes('-fullscreen', True)
        overlay.attributes('-alpha', 0.3)
        overlay.configure(bg='red')
        overlay.attributes('-topmost', True)
        
        canvas = tk.Canvas(overlay, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        start_x = start_y = end_x = end_y = 0
        rect_id = None
        
        def start_drag(event):
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='blue', width=2)
            
        def drag(event):
            nonlocal rect_id
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)
                
        def end_drag(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x, event.y
            overlay.destroy()
            self.root.deiconify()  # Show main window
            
            # Set the capture region
            self.capture_region = {
                'left': min(start_x, end_x),
                'top': min(start_y, end_y),
                'width': abs(end_x - start_x),
                'height': abs(end_y - start_y)
            }
            
            # Update display
            region_text = f"{self.capture_region['width']}x{self.capture_region['height']} at ({self.capture_region['left']}, {self.capture_region['top']})"
            self.region_label.config(text=region_text, foreground="green")
            
        canvas.bind('<Button-1>', start_drag)
        canvas.bind('<B1-Motion>', drag)
        canvas.bind('<ButtonRelease-1>', end_drag)
        
        # Instructions
        instructions = tk.Label(overlay, text="Drag to select screen region, then release", 
                              fg='white', bg='black', font=('Arial', 16))
        instructions.pack(pady=20)
        
    def capture_frame(self):
        """Capture a frame from the selected screen region"""
        if not self.capture_region:
            return None
            
        try:
            # Capture screenshot
            screenshot = self.sct.grab(self.capture_region)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            
            # Resize for efficiency (optional)
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
            
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=self.image_quality)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
        
    def update_restart_countdown(self):
        """Update the restart countdown display"""
        if self.is_streaming and self.session_start_time:
            elapsed = time.time() - self.session_start_time
            remaining = max(0, self.restart_interval - elapsed)
            
            if remaining > 0:
                self.restart_label.config(text=f"{remaining:.1f}s", foreground="blue")
                # Schedule next update
                self.root.after(100, self.update_restart_countdown)
            else:
                self.restart_label.config(text="Restarting session...", foreground="orange")
                # Trigger restart of streaming session
                self.restart_streaming_session()
        else:
            self.restart_label.config(text="Not running", foreground="gray")
            
    def restart_streaming_session(self):
        """Restart the streaming session (stop and start again)"""
        try:
            self.update_response_display("=== AUTO-RESTART: Restarting streaming session to maintain fresh connection ===")
            
            # Stop current streaming and cancel all tasks
            self.is_streaming = False
            
            # Cancel any running async tasks
            for task in self.current_tasks:
                if not task.done():
                    task.cancel()
            self.current_tasks.clear()
            
            # Close websocket if it exists
            if self.websocket and not self.websocket.close:
                try:
                    # Create a simple async function to close the websocket
                    async def close_ws():
                        try:
                            await self.websocket.close()
                        except Exception as e:
                            print(f"Error closing websocket: {e}")
                    
                    # Run the close in a new event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, schedule the close
                            asyncio.create_task(close_ws())
                        else:
                            # If no loop, run it
                            asyncio.run(close_ws())
                    except Exception as e:
                        print(f"Error during websocket close: {e}")
                        
                except Exception as e:
                    print(f"Error in websocket cleanup: {e}")
            
            self.websocket = None
            
            # Schedule restart after a short delay to allow cleanup
            def restart_after_cleanup():
                if not self.is_streaming:  # Only restart if we're still stopped
                    self.start_streaming()
            
            # Restart after 500ms for faster resumption
            self.root.after(500, restart_after_cleanup)
            
        except Exception as e:
            print(f"Error restarting streaming session: {e}")
            self.update_response_display(f"Error restarting session: {e}")
            # Fallback to stopping streaming
            self.stop_streaming()
        
    async def connect_to_gemini(self):
        """Connect to Gemini Live API"""
        try:
            # Use the correct WebSocket endpoint for Gemini 2.0 Flash Live API
            uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
            
            self.websocket = await websockets.connect(uri)
            
            # Send initial setup message for Gemini 2.0 Flash
            setup_message = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "speech_config": {
                            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
                        }
                    }
                }
            }
            
            await self.websocket.send(json.dumps(setup_message))
            
            # Wait for setup confirmation
            response = await self.websocket.recv()
            setup_response = json.loads(response)
            
            if "setupComplete" in setup_response:
                return True
            else:
                print(f"Setup failed: {setup_response}")
                return False
                
        except Exception as e:
            print(f"Error connecting to Gemini: {e}")
            return False
            
    async def send_frame_to_gemini(self, image):
        """Send frame to Gemini for analysis"""
        if not self.websocket or self.websocket.closed or not self.is_streaming:
            return
            
        try:
            base64_image = self.image_to_base64(image)
            
            message = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [
                            {"text": self.prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }],
                    "turn_complete": True
                }
            }
            
            await self.websocket.send(json.dumps(message))
            
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed during send")
            self.is_streaming = False
        except Exception as e:
            if self.is_streaming:  # Only log if we're supposed to be streaming
                print(f"Error sending frame: {e}")
            self.is_streaming = False
            
    async def listen_for_responses(self):
        """Listen for responses from Gemini"""
        try:
            while self.is_streaming and self.websocket and not self.websocket.closed:
                try:
                    response = await self.websocket.recv()
                    data = json.loads(response)
                    
                    if "serverContent" in data:
                        content = data["serverContent"]
                        if "modelTurn" in content:
                            parts = content["modelTurn"].get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    text = part["text"]
                                    self.update_response_display(text)
                except asyncio.CancelledError:
                    print("Response listener cancelled")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break
                except Exception as e:
                    if self.is_streaming:  # Only log if we're supposed to be streaming
                        print(f"Error in response listener: {e}")
                    break
                    
        except Exception as e:
            print(f"Error listening for responses: {e}")
        finally:
            print("Response listener stopped")
            
    def update_response_display(self, text):
        """Update the response display in the UI"""
        def update_ui():
            self.response_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n\n")
            
            # Only auto-scroll if enabled and user hasn't manually scrolled
            if self.auto_scroll.get() and not self.user_scrolled:
                self.response_text.see(tk.END)
            
        self.root.after(0, update_ui)
        
    async def streaming_loop(self):
        """Main streaming loop"""
        frame_interval = 1.0 / self.fps
        
        try:
            while self.is_streaming:
                try:
                    # Check if we need to restart the streaming session
                    if self.is_streaming and self.session_start_time and (time.time() - self.session_start_time) >= self.restart_interval:
                        break  # Exit loop to trigger session restart
                    
                    # Capture frame
                    frame = self.capture_frame()
                    if frame and self.is_streaming:
                        await self.send_frame_to_gemini(frame)
                        
                    await asyncio.sleep(frame_interval)
                    
                except asyncio.CancelledError:
                    print("Streaming loop cancelled")
                    break
                except Exception as e:
                    if self.is_streaming:  # Only log if we're supposed to be streaming
                        print(f"Error in streaming loop: {e}")
                    break
        except Exception as e:
            print(f"Streaming loop error: {e}")
        finally:
            print("Streaming loop stopped")
                
    def start_streaming(self):
        """Start the streaming process"""
        if not self.api_key:
            messagebox.showerror("Error", "Please set your API key in config.py first")
            return
            
        if not self.capture_region:
            messagebox.showerror("Error", "Please select a screen region first")
            return
            
        # Set session start time
        self.session_start_time = time.time()
        
        self.is_streaming = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Connecting...", foreground="orange")
        
        # Start countdown display
        self.update_restart_countdown()
        
        # Start streaming in a separate thread
        def run_streaming():
            asyncio.run(self.run_streaming_session())
            
        streaming_thread = threading.Thread(target=run_streaming, daemon=True)
        streaming_thread.start()
        
    async def run_streaming_session(self):
        """Run the complete streaming session"""
        try:
            # Connect to Gemini
            if await self.connect_to_gemini():
                self.root.after(0, lambda: self.status_label.config(text="Streaming...", foreground="green"))
                
                # Start listening for responses
                listener_task = asyncio.create_task(self.listen_for_responses())
                self.current_tasks.append(listener_task)
                
                # Start streaming frames
                streaming_task = asyncio.create_task(self.streaming_loop())
                self.current_tasks.append(streaming_task)
                
                # Wait for either task to complete
                try:
                    await asyncio.gather(listener_task, streaming_task, return_exceptions=True)
                except Exception as e:
                    print(f"Error in task gathering: {e}")
                
                # Check if we need to restart the streaming session
                if self.is_streaming and self.session_start_time and (time.time() - self.session_start_time) >= self.restart_interval:
                    self.root.after(0, self.restart_streaming_session)
                    
            else:
                self.root.after(0, lambda: self.status_label.config(text="Connection failed", foreground="red"))
                
        except Exception as e:
            print(f"Streaming session error: {e}")
            self.root.after(0, lambda: self.status_label.config(text="Error occurred", foreground="red"))
        finally:
            # Clean up tasks
            for task in self.current_tasks:
                if not task.done():
                    task.cancel()
            self.current_tasks.clear()
            
            # Clean up websocket connection
            if self.websocket and not self.websocket.closed:
                try:
                    await self.websocket.close()
                except Exception as e:
                    print(f"Error closing websocket in finally: {e}")
                    
            self.websocket = None
            
            # Check if this was a restart or regular stop
            restart_happened = (self.session_start_time and 
                              (time.time() - self.session_start_time) >= self.restart_interval)
            
            if not restart_happened:
                # Only reset UI if we're not restarting the session
                self.root.after(0, self.reset_ui_after_stop)
            
    def stop_streaming(self):
        """Stop the streaming process"""
        self.is_streaming = False
        self.session_start_time = None
        
        # Cancel any running async tasks
        for task in self.current_tasks:
            if not task.done():
                task.cancel()
        self.current_tasks.clear()
        
    def reset_ui_after_stop(self):
        """Reset UI after streaming stops"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", foreground="red")
        self.restart_label.config(text="Not running", foreground="gray")
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ScreenWatcher()
    app.run()