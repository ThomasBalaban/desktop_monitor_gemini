from google import genai
from google.genai import types
import threading
import time
import cv2
from PIL import Image
import io

class GeminiClient:
    def __init__(self, api_key, system_prompt, safety_settings, response_callback, error_callback, max_output_tokens=500, debug_mode=False, audio_sample_rate=None):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.response_callback = response_callback
        self.error_callback = error_callback
        self.debug_mode = debug_mode
        self.max_output_tokens = max_output_tokens

        # 1. Initialize the V2 Client
        self.client = genai.Client(api_key=self.api_key)

        # 2. Model Configuration
        self.model_name = "gemini-2.5-flash" 
        
        # 3. Create a Chat Session (Stateful)
        # The new SDK handles config inside the chat creation or per message
        self._init_chat()

        # Concurrency Control
        self._is_processing = False
        self._lock = threading.Lock()

    def _init_chat(self):
        """Initializes or resets the chat session."""
        try:
            # Create a chat session with system instructions
            self.chat = self.client.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    max_output_tokens=self.max_output_tokens,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE"
                        ),
                    ]
                )
            )
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to init chat: {e}")

    def test_connection(self):
        """
        Tests the API connection.
        """
        try:
            if self.debug_mode:
                print(f"GeminiClient: Testing connection to {self.model_name}...")
            
            # Simple stateless call to check credentials
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Reply with 'OK' if you receive this."
            )
            
            if response and response.text:
                return True, "Connection Successful"
            else:
                return False, "No response text received"
        except Exception as e:
            return False, str(e)

    def send_message(self, frame, text_prompt=None):
        """
        Sends an image + context to the chat model.
        """
        if self._is_processing:
            if self.debug_mode:
                print("GeminiClient: Skipped frame (API Busy)")
            return

        threading.Thread(target=self._process_request, args=(frame, text_prompt), daemon=True).start()

    def _process_request(self, frame, text_prompt):
        with self._lock:
            self._is_processing = True

        try:
            # 1. Convert Frame to Bytes (JPEG)
            # The new SDK works best with explicit Part types
            if hasattr(frame, 'shape'): 
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
            else:
                pil_image = frame

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=80)
            img_bytes = img_byte_arr.getvalue()

            # 2. Build Content Parts
            parts = []
            
            # Add Image Part
            parts.append(types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg"
            ))

            # Add Text Part (if exists)
            if text_prompt:
                parts.append(types.Part.from_text(text=text_prompt))
                if self.debug_mode:
                    print(f"GeminiClient: Sending context len: {len(text_prompt)}")

            # 3. Send Stream Request
            # Note: We use the Chat session we created earlier
            response_stream = self.chat.send_message_stream(
                message=parts
            )

            for chunk in response_stream:
                if chunk.text:
                    if self.response_callback:
                        self.response_callback(chunk.text)

        except Exception as e:
            print(f"GeminiClient Error: {e}")
            if self.error_callback:
                self.error_callback(str(e))
        
        finally:
            with self._lock:
                self._is_processing = False

    def reset_chat(self):
        self._init_chat()
        if self.debug_mode:
            print("GeminiClient: Chat history reset.")