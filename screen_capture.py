"""
Capture functionality handling both Screen Scraping (MSS) and Direct Video (OpenCV).
"""

import base64
import tkinter as tk
from io import BytesIO
import mss # type: ignore
import cv2 # type: ignore
import numpy as np
from PIL import Image

class ScreenCapture:
    def __init__(self, image_quality=85, video_index=None):
        self.image_quality = image_quality
        self.video_index = video_index
        self.cap = None
        self.sct = None
        self.capture_region = None
        
        if self.video_index is not None:
            # --- CAMERA MODE ---
            print(f"Initializing Video Capture on Index {self.video_index}...")
            self.cap = cv2.VideoCapture(self.video_index)
            # Try to force high resolution (optional, remove if it causes issues)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video device {self.video_index}")
        else:
            # --- SCREEN MODE ---
            print("Initializing Screen Capture (MSS)...")
            self.sct = mss.mss()

    def set_capture_region(self, region):
        """Set the screen region (Only used in Screen Mode)."""
        self.capture_region = region
    
    def is_ready(self):
        """Checks if we have a valid source configured."""
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.sct is not None and self.capture_region is not None:
            return True
        return False

    def select_region_interactive(self, parent_window):
        """Allows region selection (Only used in Screen Mode)."""
        if self.cap is not None:
            print("Using Camera Mode - Skipping region selection.")
            return None

        parent_window.withdraw()
        try:
            monitor = self.sct.monitors[1]
        except IndexError:
            monitor = self.sct.monitors[0]

        overlay = tk.Toplevel()
        overlay.geometry(f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}")
        overlay.attributes('-alpha', 0.3)
        overlay.configure(bg='red')
        overlay.attributes('-topmost', True)
        overlay.overrideredirect(True)

        canvas = tk.Canvas(overlay, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        start_x = start_y = 0
        rect_id = None
        selected_region = None
        
        def start_drag(event):
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            if rect_id: canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='blue', width=2)
            
        def drag(event):
            nonlocal rect_id
            if rect_id: canvas.coords(rect_id, start_x, start_y, event.x, event.y)
                
        def end_drag(event):
            nonlocal selected_region
            end_x, end_y = event.x, event.y
            left = min(start_x, end_x) + monitor['left']
            top = min(start_y, end_y) + monitor['top']
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)
            selected_region = {'left': left, 'top': top, 'width': width, 'height': height}
            overlay.destroy()
            parent_window.deiconify()
            
        canvas.bind('<Button-1>', start_drag)
        canvas.bind('<B1-Motion>', drag)
        canvas.bind('<ButtonRelease-1>', end_drag)
        overlay.wait_window()
        
        if selected_region and selected_region['width'] > 0:
            self.capture_region = selected_region
        return self.capture_region

    def capture_frame(self):
        """Captures a frame from either Camera or Screen."""
        
        # 1. Camera Mode
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                return None
            # OpenCV is BGR, Pillow needs RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize if huge (to save bandwidth)
            max_size = 1024
            if img.width > max_size:
                ratio = max_size / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_size, new_height), Image.Resampling.LANCZOS)
            return img

        # 2. Screen Mode
        elif self.sct and self.capture_region:
            try:
                screenshot = self.sct.grab(self.capture_region)
                img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
                
                max_size = 800
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                return img
            except Exception as e:
                print(f"Error capturing screen: {e}")
                return None
        
        return None
    
    def image_to_base64(self, image):
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=self.image_quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def release(self):
        if self.cap:
            self.cap.release()