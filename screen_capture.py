"""
Screen capture functionality for Screen Watcher, targeting only the main monitor.
"""

import base64
import tkinter as tk
from io import BytesIO
import mss # type: ignore
from PIL import Image # type: ignore

class ScreenCapture:
    """Handles screen capture operations, focusing on the primary display."""
    
    def __init__(self, image_quality=85):
        self.sct = mss.mss()
        self.image_quality = image_quality
        self.capture_region = None
    
    def set_capture_region(self, region):
        """Set the screen region to capture."""
        self.capture_region = region
    
    def select_region_interactive(self, parent_window):
        """Allow user to select a screen region on the main monitor."""
        parent_window.withdraw()  # Hide main window

        # --- NEW: Simplified single-monitor logic ---
        # mss.monitors[1] is the primary monitor. [0] is all monitors combined.
        try:
            monitor = self.sct.monitors[1]
        except IndexError:
            print("Could not find primary monitor. Using all-in-one display.")
            monitor = self.sct.monitors[0]

        overlay = tk.Toplevel()
        # Set geometry to match the primary monitor
        overlay.geometry(f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}")
        overlay.attributes('-alpha', 0.3)
        overlay.configure(bg='red')
        overlay.attributes('-topmost', True)
        overlay.overrideredirect(True) # Remove window decorations

        canvas = tk.Canvas(overlay, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        start_x = start_y = 0
        rect_id = None
        selected_region = None
        
        def start_drag(event):
            nonlocal start_x, start_y, rect_id
            # Get coordinates relative to the monitor
            start_x, start_y = event.x, event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='blue', width=2)
            
        def drag(event):
            nonlocal rect_id
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)
                
        def end_drag(event):
            nonlocal selected_region
            end_x, end_y = event.x, event.y
            
            # The coordinates are relative to the monitor, so we add the monitor's offset
            # to get the true screen coordinates.
            left = min(start_x, end_x) + monitor['left']
            top = min(start_y, end_y) + monitor['top']
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)
            
            selected_region = {'left': left, 'top': top, 'width': width, 'height': height}
            
            overlay.destroy()
            parent_window.deiconify()  # Show main window
            
        canvas.bind('<Button-1>', start_drag)
        canvas.bind('<B1-Motion>', drag)
        canvas.bind('<ButtonRelease-1>', end_drag)
        
        instructions = tk.Label(overlay, text="Drag to select screen region, then release", 
                              fg='white', bg='black', font=('Arial', 16))
        instructions.pack(pady=20)
        
        overlay.wait_window()
        
        if selected_region and selected_region['width'] > 0 and selected_region['height'] > 0:
            self.capture_region = selected_region
            
        return self.capture_region

    def capture_frame(self):
        """Capture a frame from the selected screen region."""
        if not self.capture_region:
            return None
            
        try:
            screenshot = self.sct.grab(self.capture_region)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=self.image_quality)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')