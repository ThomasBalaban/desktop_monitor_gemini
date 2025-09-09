import tkinter as tk
from tkinter import scrolledtext, font

class AppGUI:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Gemini Screen Watcher")
        self.root.geometry("900x850")  # Increased height for error log
        self.root.configure(bg="#2E2E2E")
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=11)
        top_frame = tk.Frame(self.root, bg="#2E2E2E", padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        main_frame = tk.Frame(self.root, bg="#2E2E2E", padx=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=0) # New row for error log
        status_frame = tk.Frame(self.root, bg="#1E1E1E", padx=10, pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        button_font = ("Helvetica", 12, "bold")
        self.start_button = tk.Button(top_frame, text="Start Watching", command=self.controller.start_streaming,
                                      bg="white", fg="black", font=button_font,
                                      highlightbackground="#4CAF50", highlightthickness=2, bd=0, padx=10, pady=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(top_frame, text="Stop Watching", command=self.controller.stop_streaming,
                                     bg="white", fg="black", font=button_font, state=tk.DISABLED,
                                     highlightbackground="#F44336", highlightthickness=2, bd=0, padx=10, pady=2)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        feed_label = tk.Label(main_frame, text="Live Gemini Feed", bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 14, "bold"))
        feed_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.feed_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#E0E0E0", font=("Helvetica", 12))
        self.feed_text.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        prompt_label = tk.Label(main_frame, text="Current Prompt", bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 14, "bold"))
        prompt_label.grid(row=0, column=1, sticky="w", pady=(0, 5))
        self.prompt_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, bg="#1E1E1E", fg="#C5C5C5")
        self.prompt_text.grid(row=1, column=1, sticky="nsew")
        self.prompt_text.insert(tk.END, self.controller.get_prompt())
        self.prompt_text.configure(state=tk.DISABLED)

        # --- NEW: Error Log Section ---
        error_label = tk.Label(main_frame, text="Error Log", bg="#2E2E2E", fg="#FFFFFF", font=("Helvetica", 14, "bold"))
        error_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 5))
        self.error_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1E1E1E", fg="#FF7B7B", height=5)
        self.error_text.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 10))


        self.status_label = tk.Label(status_frame, text="Status: Idle", bg="#1E1E1E", fg="#FFFFFF")
        self.status_label.pack(side=tk.LEFT)
        self.websocket_status_label = tk.Label(status_frame, text="WebSocket: Inactive", bg="#1E1E1E", fg="#FFFFFF")
        self.websocket_status_label.pack(side=tk.RIGHT)

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
        """Adds an error message to the error log."""
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

    def update_button_states(self, is_streaming):
        def _task():
            if is_streaming:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
            else:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
        self.root.after(0, _task)