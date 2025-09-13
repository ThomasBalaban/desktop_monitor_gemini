# gemini_client.py
import asyncio
import json
import requests # type: ignore
import websockets # type: ignore
import time

class GeminiClient:
    def __init__(self, api_key, prompt, safety_settings=None, response_callback=None, error_callback=None, max_output_tokens=150, debug_mode=False):
        self.api_key = api_key
        self.prompt = prompt
        self.safety_settings = safety_settings
        self.response_callback = response_callback
        self.error_callback = error_callback
        self.max_output_tokens = max_output_tokens
        self.debug_mode = debug_mode
        self.websocket = None
        self.is_connected = False
        self.connection_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        self.last_send_time = 0
        self.min_send_interval = 0.1

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
                url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
                response_fallback = requests.post(url_fallback, json={
                    "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
                }, timeout=10)
                if response_fallback.status_code == 200:
                    return True, "API key works but Gemini 2.0 Flash may not be available. Live streaming might not work."
                else:
                    return False, f"API key invalid or no access to Gemini models: {response_fallback.status_code}"
            else:
                return False, f"API test failed: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Connection test failed: {e}"

    async def connect(self):
        if self.connection_lock:
            async with self.connection_lock:
                return await self._do_connect()
        else:
            return await self._do_connect()

    async def _do_connect(self):
        await self._cleanup_connection()
        try:
            self.info_print("Connecting to Gemini WebSocket...")
            uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
            self.websocket = await asyncio.wait_for(
                websockets.connect(uri, ping_interval=20, ping_timeout=10),
                timeout=15.0
            )
            self.debug_print("WebSocket connected successfully")
            setup_message = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generation_config": {
                        "response_modalities": ["TEXT"]
                    }
                }
            }
            if self.max_output_tokens and 50 <= self.max_output_tokens <= 8192:
                setup_message["setup"]["generation_config"]["max_output_tokens"] = self.max_output_tokens
            if self.safety_settings:
                setup_message["setup"]["safety_settings"] = self.safety_settings
            await asyncio.wait_for(
                self.websocket.send(json.dumps(setup_message)),
                timeout=10.0
            )
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=15.0
            )
            setup_response = json.loads(response)
            if "setupComplete" in setup_response:
                self.is_connected = True
                self.info_print("Gemini connection established successfully")
                return True
            else:
                self.info_print(f"Setup failed: {setup_response}")
                if self.error_callback:
                    self.error_callback(f"Gemini setup failed: {setup_response}")
                await self._cleanup_connection()
                return False
        except Exception as e:
            self.info_print(f"Error connecting to Gemini: {e}")
            if self.error_callback:
                self.error_callback(f"Connection error: {e}")
            await self._cleanup_connection()
            return False

    async def send_image(self, base64_image):
        if not self.is_healthy():
            self.debug_print("Cannot send image: WebSocket not connected or healthy")
            return False

        current_time = time.time()
        time_since_last_send = current_time - self.last_send_time
        if time_since_last_send < self.min_send_interval:
            await asyncio.sleep(self.min_send_interval - time_since_last_send)

        try:
            message = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [
                            {"text": self.prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                        ]
                    }],
                    "turn_complete": True
                }
            }
            await asyncio.wait_for(
                self.websocket.send(json.dumps(message)),
                timeout=10.0
            )
            self.last_send_time = time.time()
            return True
        except Exception as e:
            self.info_print(f"Error sending image: {e}")
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
                                    self.response_callback(part["text"])
                except asyncio.TimeoutError:
                    self.debug_print("Response timeout - connection may be stale")
                    self.is_connected = False
                    break
                except websockets.exceptions.ConnectionClosed:
                    self.info_print("WebSocket connection closed")
                    self.is_connected = False
                    break
                except Exception as e:
                    if self.is_connected:
                        self.info_print(f"Error in response listener: {e}")
                        if self.error_callback:
                            self.error_callback(f"Listener error: {e}")
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
                self.debug_print("WebSocket object has no 'closed' attribute. It may be a different object type or an error state.")
            except Exception as e:
                self.debug_print(f"Error closing websocket: {e}")
        self.websocket = None

    def is_healthy(self):
        """More robustly checks if the connection is active."""
        if not self.is_connected or not self.websocket:
            return False
        try:
            return not self.websocket.closed
        except AttributeError:
            # If the object doesn't have a '.closed' attribute, assume it's unhealthy.
            return False