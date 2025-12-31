import asyncio
import json
import base64
import websockets
import logging

class OpenAIRealtimeClient:
    # UPDATED MODEL NAME HERE vvv
    def __init__(self, api_key, model="gpt-4o-realtime-preview", 
                 on_text_delta=None, on_error=None):
        self.url = f"wss://api.openai.com/v1/realtime?model={model}"
        self.api_key = api_key
        self.websocket = None
        self.is_connected = False
        
        # Callbacks
        self.on_text_delta = on_text_delta
        self.on_error = on_error
        
        self.SAMPLE_RATE = 24000 
        
    async def connect(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            self.websocket = await websockets.connect(self.url, extra_headers=headers)
            self.is_connected = True
            print("✅ Connected to OpenAI Realtime API")
            
            asyncio.create_task(self._receive_loop())
            await self._update_session()
            return True
        except Exception as e:
            if self.on_error: self.on_error(f"Connection failed: {e}")
            return False

    async def _update_session(self):
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "temperature": 0.7,
                "instructions": (
                    "You are an expert theatrical transcriber. "
                    "TRANSCRIPTION RULES:\n"
                    "1. ALWAYS output text if you hear human voices or singing.\n"
                    "2. Identify speakers (e.g. 'Charlie:', 'Alastor:').\n"
                    "3. Describe SFX in brackets [Like This].\n"
                    "4. If lyrics are unclear, transcribe what you hear phonetically or describe the music style."
                ),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        print("📤 Sending Session Update...")
        await self.websocket.send(json.dumps(session_update))

    async def send_audio_chunk(self, audio_bytes):
        if not self.is_connected or not self.websocket:
            return

        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        try:
            await self.websocket.send(json.dumps(event))
        except Exception as e:
            print(f"Error sending audio: {e}")

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                event = json.loads(message)
                event_type = event.get("type", "unknown")
                
                # --- EVENTS OF INTEREST ---
                
                # 1. Text Delta
                if event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    if self.on_text_delta:
                        self.on_text_delta(delta)

                # 2. VAD Trigger
                elif event_type == "input_audio_buffer.speech_started":
                    print("🗣️  [VAD] Speech Detected...")
                
                # 3. Response Creation
                elif event_type == "response.created":
                    print("🤖 [AI] Thinking...")

                # 4. Response Done
                elif event_type == "response.done":
                    response = event.get("response", {})
                    status = response.get("status")
                    output = response.get("output", [])
                    
                    if status == "completed":
                        has_content = False
                        for item in output:
                            if item.get("content"): has_content = True
                        
                        if not has_content:
                            print("⚠️ [AI] Finished with NO CONTENT. (Model decided not to speak)")
                        else:
                            print("✅ [AI] Response complete.")
                            
                    elif status == "incomplete":
                         print(f"❌ [AI] Response INCOMPLETE. Reason: {response.get('status_details')}")
                    elif status == "failed":
                         print(f"❌ [AI] Response FAILED. Error: {response.get('status_details')}")
                    else:
                        print(f"ℹ️ [AI] Response finished with status: {status}")

                # 5. Error
                elif event_type == "error":
                    error_msg = event.get("error", {}).get("message", "Unknown error")
                    print(f"❌ OpenAI Error: {error_msg}")
                    if self.on_error: self.on_error(error_msg)
                    
        except websockets.exceptions.ConnectionClosed:
            print("OpenAI WebSocket closed")
            self.is_connected = False
        except Exception as e:
            print(f"Receive loop error: {e}")