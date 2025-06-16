"""
Gemini API client for Screen Watcher
"""

import asyncio
import json
import requests # type: ignore
import websockets # type: ignore

class GeminiClient:
    """Handles communication with Gemini API"""
    
    def __init__(self, api_key, prompt, safety_settings=None, response_callback=None, max_output_tokens=150):
        self.api_key = api_key
        self.prompt = prompt
        self.safety_settings = safety_settings
        self.response_callback = response_callback
        self.max_output_tokens = max_output_tokens
        self.websocket = None
        self.is_connected = False
    
    def test_connection(self):
        """Test connection to Gemini API"""
        try:
            # Test with Gemini 2.0 Flash model that supports the Live API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            response = requests.post(url, json={
                "contents": [{"parts": [{"text": "Hello, this is a connection test."}]}]
            }, timeout=10)
            
            if response.status_code == 200:
                return True, "API connection successful! Gemini 2.0 Flash is ready."
            elif response.status_code == 404:
                # Try the regular Gemini Flash model as fallback
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
        """Connect to Gemini Live API"""
        try:
            print("Attempting to connect to Gemini WebSocket...")
            # Use the correct WebSocket endpoint for Gemini 2.0 Flash Live API
            uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
            
            self.websocket = await websockets.connect(uri)
            print("WebSocket connected successfully")
            
            # Send initial setup message for Gemini 2.0 Flash
            setup_message = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generation_config": {
                        "response_modalities": ["TEXT"]
                    }
                }
            }
            
            # Add max_output_tokens if it's a reasonable value
            if self.max_output_tokens and 50 <= self.max_output_tokens <= 8192:
                setup_message["setup"]["generation_config"]["max_output_tokens"] = self.max_output_tokens
                print(f"Added max_output_tokens: {self.max_output_tokens}")
            
            # Add safety settings if configured
            if self.safety_settings:
                setup_message["setup"]["safety_settings"] = self.safety_settings
                print(f"Added safety settings: {self.safety_settings}")
            
            print(f"Sending setup message: {setup_message}")
            await self.websocket.send(json.dumps(setup_message))
            print("Setup message sent")
            
            # Wait for setup confirmation
            print("Waiting for setup confirmation...")
            response = await self.websocket.recv()
            setup_response = json.loads(response)
            print(f"Setup response received: {setup_response}")
            
            if "setupComplete" in setup_response:
                self.is_connected = True
                print("Setup completed successfully")
                return True
            else:
                print(f"Setup failed: {setup_response}")
                return False
                
        except Exception as e:
            print(f"Error connecting to Gemini: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def send_image(self, base64_image):
        """Send image to Gemini for analysis"""
        if not self.websocket or self.websocket.closed or not self.is_connected:
            print("Cannot send image: WebSocket not connected")
            return False
            
        try:
            print("Preparing to send image...")
            
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
            
            print(f"Sending message with image data length: {len(base64_image)} characters")
            print(f"Prompt: {self.prompt[:100]}...")  # Show first 100 chars of prompt
            
            await self.websocket.send(json.dumps(message))
            print("Image message sent successfully")
            return True
            
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed during send")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Error sending image: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
            return False
    
    async def listen_for_responses(self):
        """Listen for responses from Gemini"""
        try:
            print("Starting response listener...")
            while self.is_connected and self.websocket and not self.websocket.closed:
                try:
                    print("Waiting for response...")
                    response = await self.websocket.recv()
                    print(f"Received response: {response}")
                    data = json.loads(response)
                    
                    if "serverContent" in data:
                        content = data["serverContent"]
                        if "modelTurn" in content:
                            parts = content["modelTurn"].get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    text = part["text"]
                                    print(f"Extracted text: {text}")
                                    if self.response_callback:
                                        self.response_callback(text)
                                        
                except asyncio.CancelledError:
                    print("Response listener cancelled")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    self.is_connected = False
                    break
                except Exception as e:
                    if self.is_connected:
                        print(f"Error in response listener: {e}")
                        import traceback
                        traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"Error listening for responses: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Response listener stopped")
    
    async def disconnect(self):
        """Disconnect from Gemini API"""
        self.is_connected = False
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")
        self.websocket = None