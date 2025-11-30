# gemini_client.py
import asyncio
import json
import requests
import websockets
import time
import base64

class GeminiClient:
    def __init__(self, api_key, prompt, safety_settings=None, response_callback=None, error_callback=None, max_output_tokens=150, debug_mode=False, audio_sample_rate=16000):
        self.api_key = api_key
        self.prompt = prompt
        self.safety_settings = safety_settings
        self.response_callback = response_callback
        self.error_callback = error_callback
        self.max_output_tokens = max_output_tokens
        self.debug_mode = debug_mode
        self.audio_sample_rate = audio_sample_rate
        
        self.websocket = None
        self.is_connected = False
        self.connection_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        
    def debug_print(self, message):
        if self.debug_mode:
            print(f"[DEBUG] {message}")

    def info_print(self, message):
        print(message)

    def test_connection(self):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            response = requests.post(url, json={
                "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
            }, timeout=10)
            if response.status_code == 200:
                return True, "API connection successful! Gemini 2.0 Flash is ready."
            elif response.status_code == 404:
                return False, f"Gemini 2.0 Flash model not found. Check if the API key has access to 'gemini-2.0-flash-exp'."
            else:
                return False, f"API test failed: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Connection test failed: {e}"

    async def connect(self):
        """
        Main connect method with Auto-Retry / Fallback logic.
        """
        if self.connection_lock:
            async with self.connection_lock:
                return await self._connect_with_fallback()
        else:
            return await self._connect_with_fallback()

    async def _connect_with_fallback(self):
        # Attempt 1: The "Modern" way (System Prompt in Setup)
        self.info_print("Attempt 1: Connecting with System Prompt in Setup...")
        success = await self._do_connect_attempt(use_system_instruction=True)
        
        if success:
            return True
            
        # If Attempt 1 failed, wait briefly and try Attempt 2
        self.info_print("⚠️ Attempt 1 failed (likely Error 1007). Retrying with Fallback Strategy...")
        await asyncio.sleep(1)
        
        # Attempt 2: The "Classic" way (Bare Setup + Prompt as Message)
        self.info_print("Attempt 2: Connecting with Bare Setup...")
        success = await self._do_connect_attempt(use_system_instruction=False)
        
        if success:
            self.info_print("✅ Fallback connection successful! Sending Prompt manually...")
            # Manually send the prompt now
            await self._send_initial_prompt_message()
            return True
            
        self.info_print("❌ All connection attempts failed.")
        return False

    async def _do_connect_attempt(self, use_system_instruction=True):
        await self._cleanup_connection()
        try:
            uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
            self.websocket = await asyncio.wait_for(
                websockets.connect(uri, ping_interval=20, ping_timeout=10),
                timeout=15.0
            )
            
            # Construct Setup Message
            setup_payload = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "max_output_tokens": self.max_output_tokens
                    }
                }
            }
            
            # Conditionally add System Instruction
            if use_system_instruction:
                setup_payload["setup"]["system_instruction"] = {
                    "parts": [{"text": self.prompt}]
                }
                
            if self.safety_settings:
                setup_payload["setup"]["safety_settings"] = self.safety_settings
            
            # Send Setup
            await self.websocket.send(json.dumps(setup_payload))
            
            # Wait for setup confirmation
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=15.0
            )
            setup_response = json.loads(response)
            
            if "setupComplete" in setup_response:
                self.is_connected = True
                return True
            else:
                self.debug_print(f"Setup response invalid: {setup_response}")
                return False
                
        except websockets.exceptions.ConnectionClosed as e:
            self.debug_print(f"Connection closed immediately: Code {e.code} ({e.reason})")
            return False
        except Exception as e:
            self.debug_print(f"Connection exception: {e}")
            return False

    async def _send_initial_prompt_message(self):
        """Sends the system prompt as a normal user message (Fallback method)."""
        if not self.is_connected: return
        try:
            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": f"SYSTEM INSTRUCTIONS: {self.prompt}"}]
                    }],
                    "turn_complete": False
                }
            }
            await self.websocket.send(json.dumps(msg))
        except Exception as e:
            self.debug_print(f"Failed to send initial prompt: {e}")

    async def send_multimodal_frame(self, base64_image, audio_bytes=None, turn_complete=False, text=None):
        if not self.is_connected or not self.websocket:
            return False

        try:
            parts = []
            
            if text:
                parts.append({"text": text})
            
            if audio_bytes:
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                parts.append({
                    "inline_data": {
                        "mime_type": f"audio/pcm;rate={self.audio_sample_rate}", 
                        "data": base64_audio
                    }
                })

            if base64_image:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            
            if not parts:
                return True

            message = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": parts
                    }],
                    "turn_complete": turn_complete 
                }
            }
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except websockets.exceptions.ConnectionClosed:
            self.info_print("Socket closed during send.")
            self.is_connected = False
            return False
        except Exception as e:
            self.debug_print(f"Error sending frame: {e}")
            self.is_connected = False
            return False

    async def listen_for_responses(self):
        try:
            while self.is_healthy():
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    
                    if "serverContent" in data:
                        content = data["serverContent"]
                        
                        if "modelTurn" not in content:
                            if "turnComplete" in content:
                                continue
                        
                        if "modelTurn" in content:
                            parts = content["modelTurn"].get("parts", [])
                            for part in parts:
                                if "text" in part and self.response_callback:
                                    text = part["text"]
                                    if "[WAIT]" not in text:
                                        self.response_callback(text)
                                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    self.info_print(f"WebSocket listener closed: {e.code} - {e.reason}")
                    self.is_connected = False
                    break
                except Exception as e:
                    if self.is_connected:
                        self.info_print(f"Error in response listener: {e}")
                    break
        finally:
            self.is_connected = False

    async def disconnect(self):
        self.info_print("Disconnecting from Gemini...")
        await self._cleanup_connection()

    async def _cleanup_connection(self):
        self.is_connected = False
        if self.websocket:
            try:
                if not self.websocket.closed:
                    await asyncio.wait_for(self.websocket.close(), timeout=5.0)
            except AttributeError:
                pass
            except Exception:
                pass
        self.websocket = None

    def is_healthy(self):
        if not self.is_connected or not self.websocket:
            return False
        try:
            return not self.websocket.closed
        except AttributeError:
            return False