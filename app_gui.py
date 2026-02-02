import tkinter as tk
from tkinter import scrolledtext, font, ttk 
from PIL import ImageTk, Image

class AppGUI:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Gemini 2.0 Unified Monitor")
        self.root.geometry("1000x950") # Slightly taller for settings
        self.root.configure(bg="#2E2E2E")
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=11)
        
        self.root.protocol("WM_DELETE_WINDOW", self.controller.stop)
        
        # --- Top Frame ---
        top_frame = tk.Frame(self.root, bg="#2E2E2E", padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_analyze = tk.Button(top_frame, text="Analyze Last 5s", 
                                     command=self.controller.request_analysis, 
                                     bg="#4CAF50", fg="white", 
                                     font=("Helvetica", 10, "bold"),
                                     relief=tk.FLAT, padx=10)
        self.btn_analyze.pack(side=tk.RIGHT)

        # --- Audio Settings Frame (NEW) ---
        settings_frame = tk.Frame(self.root, bg="#3E3E3E", padx=10, pady=5)
        settings_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Mic Selection
        tk.Label(settings_frame, text="🎤 Mic:", bg="#3E3E3E", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        self.mic_var = tk.StringVar()
        self.combo_mic = ttk.Combobox(settings_frame, textvariable=self.mic_var, state="readonly", width=30)
        self.combo_mic.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_mic.bind("<<ComboboxSelected>>", self.controller.on_mic_changed)

        # Desktop Selection
        tk.Label(settings_frame, text="🔊 Desktop:", bg="#3E3E3E", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        self.desktop_var = tk.StringVar()
        self.combo_desktop = ttk.Combobox(settings_frame, textvariable=self.desktop_var, state="readonly", width=30)
        self.combo_desktop.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_desktop.bind("<<ComboboxSelected>>", self.controller.on_desktop_changed)

        # Refresh Button
        self.btn_refresh = tk.Button(settings_frame, text="🔄 Refresh", 
                                   command=self.controller.refresh_audio_devices,
                                   bg="#555555", fg="white", relief=tk.FLAT, font=("Helvetica", 9))
        self.btn_refresh.pack(side=tk.LEFT)
        
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
        self.feed_text.grid(row=1, column=0, sticky="nsew", padx=(0, 0))
        
        self.error_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#FF7B7B", height=5)
        self.error_text.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        # --- Status Bar ---
        status_frame = tk.Frame(self.root, bg="#1E1E1E", padx=10, pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, text="Status: Idle", bg="#1E1E1E", fg="#FFFFFF")
        self.status_label.pack(side=tk.LEFT)
        self.websocket_status_label = tk.Label(status_frame, text="WebSocket: Inactive", bg="#1E1E1E", fg="#FFFFFF")
        self.websocket_status_label.pack(side=tk.RIGHT)

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
            scroll_pos = self.feed_text.yview()
            is_at_top = scroll_pos[0] == 0.0
            top_index = self.feed_text.index("@0,0")
            self.feed_text.configure(state=tk.NORMAL)
            timestamp = self.controller.get_timestamp()
            self.feed_text.insert('1.0', f"{text}\n\n", "response")
            self.feed_text.insert('1.0', f"--- {timestamp} ---\n", "timestamp")
            if not is_at_top:
                self.feed_text.see(top_index)
            self.feed_text.configure(state=tk.DISABLED)
            self.feed_text.tag_config("timestamp", foreground="#BB86FC", font=("Helvetica", 10, "italic"))
            self.feed_text.tag_config("response", lmargin1=10, lmargin2=10)
        self.root.after(0, _task)

    def add_reset_separator(self):
        def _task():
            scroll_pos = self.feed_text.yview()
            is_at_top = scroll_pos[0] == 0.0
            top_index = self.feed_text.index("@0,0")
            self.feed_text.configure(state=tk.NORMAL)
            separator_text = "─" * 80
            self.feed_text.insert('1.0', f"\n{separator_text}\n\n", "separator")
            if not is_at_top:
                self.feed_text.see(top_index)
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
        """Populate comboboxes with device lists"""
        def _task():
            self.combo_mic['values'] = mic_list
            self.combo_desktop['values'] = desktop_list
            
            # Set current selections
            for item in mic_list:
                if item.startswith(f"{current_mic_idx}:"):
                    self.combo_mic.set(item)
                    break
            
            for item in desktop_list:
                if item.startswith(f"{current_desktop_idx}:"):
                    self.combo_desktop.set(item)
                    break
        self.root.after(0, _task)