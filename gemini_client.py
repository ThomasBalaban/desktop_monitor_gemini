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
        self.info_print("Attempt 1: Connecting with System Prompt in Setup...")
        success = await self._do_connect_attempt(use_system_instruction=True)
        
        if success:
            return True
            
        self.info_print("⚠️ Attempt 1 failed. Retrying with Fallback Strategy...")
        await asyncio.sleep(1)
        
        self.info_print("Attempt 2: Connecting with Bare Setup...")
        success = await self._do_connect_attempt(use_system_instruction=False)
        
        if success:
            self.info_print("✅ Fallback connection successful! Sending Prompt manually...")
            await self._send_initial_prompt_message()
            return True
            
        self.info_print("❌ All connection attempts failed.")
        return False

    async def _do_connect_attempt(self, use_system_instruction=True):
        await self._cleanup_connection()
        try:
            # Use v1beta (Correct for Gemini 2.0)
            uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={self.api_key}"
            
            self.websocket = await asyncio.wait_for(
                websockets.connect(uri, ping_interval=20, ping_timeout=10),
                timeout=15.0
            )
            
            # Setup Message (camelCase for v1beta)
            setup_payload = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                        "maxOutputTokens": self.max_output_tokens
                    }
                }
            }
            
            if use_system_instruction:
                setup_payload["setup"]["systemInstruction"] = {
                    "parts": [{"text": self.prompt}]
                }
            
            await self.websocket.send(json.dumps(setup_payload))
            
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=15.0
            )
            setup_response = json.loads(response)
            
            if "setupComplete" in setup_response:
                self.info_print("Gemini Setup Complete.")
                self.is_connected = True
                return True
            else:
                self.debug_print(f"Setup response invalid: {setup_response}")
                return False
                
        except websockets.exceptions.ConnectionClosed as e:
            self.debug_print(f"Connection closed during setup: Code {e.code} ({e.reason})")
            return False
        except Exception as e:
            self.debug_print(f"Connection exception: {e}")
            return False

    async def _send_initial_prompt_message(self):
        if not self.is_connected: return
        try:
            msg = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": f"SYSTEM INSTRUCTIONS: {self.prompt}"}]
                    }],
                    "turnComplete": False
                }
            }
            await self.websocket.send(json.dumps(msg))
        except Exception as e:
            self.debug_print(f"Failed to send initial prompt: {e}")

    async def send_multimodal_frame(self, base64_image, audio_bytes=None, turn_complete=False, text=None):
        """
        Hybrid Approach:
        - Audio -> realtimeInput (required for streaming)
        - Video -> clientContent (inline_data) (mimics old working code)
        - Text -> clientContent
        """
        if not self.is_connected or not self.websocket:
            return False

        try:
            # 1. Send Audio via realtimeInput (Streaming)
            if audio_bytes:
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                audio_msg = {
                    "realtimeInput": {
                        "mediaChunks": [{
                            "mimeType": f"audio/pcm;rate={self.audio_sample_rate}",
                            "data": base64_audio
                        }]
                    }
                }
                await self.websocket.send(json.dumps(audio_msg))

            # 2. Construct Client Content (Video + Text + Trigger)
            # We bundle these into a single 'clientContent' message to emulate a "Turn"
            parts = []
            
            if text:
                parts.append({"text": text})

            # Send Image as Inline Data (The "Old Way" that worked)
            if base64_image:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image
                    }
                })

            # If we have content to send as a Turn
            if parts:
                msg = {
                    "clientContent": {
                        "turns": [{
                            "role": "user",
                            "parts": parts
                        }],
                        "turnComplete": turn_complete 
                    }
                }
                await self.websocket.send(json.dumps(msg))
            
            # If we strictly have NO parts but need to trigger a turn complete (e.g. audio only trigger)
            elif turn_complete:
                 await self.websocket.send(json.dumps({
                    "clientContent": { "turnComplete": True }
                }))
            
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