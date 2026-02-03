import tkinter as tk
from tkinter import scrolledtext, font, ttk 
from PIL import ImageTk, Image

class AppGUI:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Gemini 2.0 Unified Monitor")
        self.root.geometry("1000x950") 
        self.root.configure(bg="#2E2E2E")
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=11)
        
        self.root.protocol("WM_DELETE_WINDOW", self.controller.stop)
        
        # --- Top Frame ---
        top_frame = tk.Frame(self.root, bg="#2E2E2E", padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        # --- Audio Settings Frame ---
        settings_frame = tk.Frame(self.root, bg="#3E3E3E", padx=10, pady=5)
        settings_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Mic Selection & Volume Meter
        tk.Label(settings_frame, text="🎤 Mic:", bg="#3E3E3E", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        self.mic_var = tk.StringVar()
        self.combo_mic = ttk.Combobox(settings_frame, textvariable=self.mic_var, state="readonly", width=25)
        self.combo_mic.pack(side=tk.LEFT, padx=(0, 5))
        self.combo_mic.bind("<<ComboboxSelected>>", self.controller.on_mic_changed)
        
        self.mic_meter = tk.Canvas(settings_frame, width=60, height=15, bg="#1E1E1E", highlightthickness=0)
        self.mic_meter.pack(side=tk.LEFT, padx=(0, 15))
        self.mic_bar = self.mic_meter.create_rectangle(0, 0, 0, 15, fill="#4CAF50")

        # Desktop Selection & Volume Meter
        tk.Label(settings_frame, text="🔊 Desktop:", bg="#3E3E3E", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        self.desktop_var = tk.StringVar()
        self.combo_desktop = ttk.Combobox(settings_frame, textvariable=self.desktop_var, state="readonly", width=25)
        self.combo_desktop.pack(side=tk.LEFT, padx=(0, 5))
        self.combo_desktop.bind("<<ComboboxSelected>>", self.controller.on_desktop_changed)

        self.desktop_meter = tk.Canvas(settings_frame, width=60, height=15, bg="#1E1E1E", highlightthickness=0)
        self.desktop_meter.pack(side=tk.LEFT, padx=(0, 15))
        self.desktop_bar = self.desktop_meter.create_rectangle(0, 0, 0, 15, fill="#2196F3")

        # Refresh Button
        self.btn_refresh = tk.Label(
            settings_frame, 
            text="🔄",               # kept simple with just the icon
            bg="#555555",            # Matches your dark theme
            fg="white", 
            font=("Helvetica", 14),
            padx=8, 
            pady=2
        )
        self.btn_refresh.pack(side=tk.LEFT, padx=(10, 0))
        
        # Add interactivity (Click and Hover effects)
        self.btn_refresh.bind("<Button-1>", lambda e: self.controller.refresh_audio_devices())
        
        # Hover animation to make it feel like a real button
        self.btn_refresh.bind("<Enter>", lambda e: self.btn_refresh.config(bg="#777777"))
        self.btn_refresh.bind("<Leave>", lambda e: self.btn_refresh.config(bg="#555555"))
        
        # --- Preview Area ---
        self.preview_frame = tk.Frame(self.root, bg="#000000", height=300)
        self.preview_frame.pack(fill=tk.X, padx=10, pady=5)
        self.preview_frame.pack_propagate(False) 
        
        self.preview_label = tk.Label(self.preview_frame, bg="black", text="Waiting for stream...", fg="gray", font=("Helvetica", 14))
        self.preview_label.pack(expand=True, fill=tk.BOTH)
        
        # --- Main Content ---
        main_frame = tk.Frame(self.root, bg="#2E2E2E", padx=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        feed_label = tk.Label(main_frame, text="Gemini Thoughts", bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 14, "bold"))
        feed_label.grid(row=0, column=0, sticky="w", pady=(10, 5))
        
        self.feed_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#E0E0E0", font=("Helvetica", 12))
        self.feed_text.grid(row=1, column=0, sticky="nsew")
        
        self.error_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#FF7B7B", height=5)
        self.error_text.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        # --- Status Bar ---
        status_frame = tk.Frame(self.root, bg="#1E1E1E", padx=10, pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="Status: Idle", bg="#1E1E1E", fg="#FFFFFF")
        self.status_label.pack(side=tk.LEFT)
        self.websocket_status_label = tk.Label(status_frame, text="WebSocket: Inactive", bg="#1E1E1E", fg="#FFFFFF")
        self.websocket_status_label.pack(side=tk.RIGHT)

    def set_volume_meter(self, source, level):
        """Updates the visual level meter (0.0 to 1.0)"""
        def _task():
            canvas = self.mic_meter if source == "mic" else self.desktop_meter
            bar = self.mic_bar if source == "mic" else self.desktop_bar
            width = int(level * 60)
            canvas.coords(bar, 0, 0, width, 15)
        self.root.after(0, _task)

    def update_preview(self, pil_image):
        def _task():
            base_height = 300
            w_percent = (base_height / float(pil_image.size[1]))
            w_size = int((float(pil_image.size[0]) * float(w_percent)))
            img_resized = pil_image.resize((w_size, base_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo 
        self.root.after(0, _task)

    def run(self):
        self.root.after(100, self.controller.update_websocket_gui_status)
        self.root.mainloop()

    def add_response(self, text):
        def _task():
            self.feed_text.configure(state=tk.NORMAL)
            timestamp = self.controller.get_timestamp()
            self.feed_text.insert('1.0', f"{text}\n\n", "response")
            self.feed_text.insert('1.0', f"--- {timestamp} ---\n", "timestamp")
            self.feed_text.configure(state=tk.DISABLED)
            self.feed_text.tag_config("timestamp", foreground="#BB86FC", font=("Helvetica", 10, "italic"))
        self.root.after(0, _task)

    def add_reset_separator(self):
        def _task():
            self.feed_text.configure(state=tk.NORMAL)
            self.feed_text.insert('1.0', f"\n{'─' * 80}\n\n", "separator")
            self.feed_text.configure(state=tk.DISABLED)
            self.feed_text.tag_config("separator", foreground="#03A9F4", justify='center')
        self.root.after(0, _task)

    def add_error(self, text):
        def _task():
            self.error_text.configure(state=tk.NORMAL)
            timestamp = self.controller.get_timestamp()
            self.error_text.insert('1.0', f"[{timestamp}] {text}\n")
            self.error_text.configure(state=tk.DISABLED)
        self.root.after(0, _task)

    def update_status(self, message, color="white"):
        def _task():
            self.status_label.config(text=f"Status: {message}", fg=color)
        self.root.after(0, _task)

    def update_websocket_status(self, message, color="white"):
        def _task():
            self.websocket_status_label.config(text=f"WebSocket: {message}", fg=color)
        self.root.after(0, _task)
        
    def set_device_lists(self, mic_list, desktop_list, current_mic_idx, current_desktop_idx):
        def _task():
            self.combo_mic['values'] = mic_list
            self.combo_desktop['values'] = desktop_list
            for item in mic_list:
                if item.startswith(f"{current_mic_idx}:"):
                    self.combo_mic.set(item)
                    break
            for item in desktop_list:
                if item.startswith(f"{current_desktop_idx}:"):
                    self.combo_desktop.set(item)
                    break
        self.root.after(0, _task)