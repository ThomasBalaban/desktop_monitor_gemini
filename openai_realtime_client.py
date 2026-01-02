import asyncio
import json
import websockets
import base64
import logging

class OpenAIRealtimeClient:
    def __init__(self, api_key, on_transcript, on_error):
        self.api_key = api_key
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.ws = None
        self.loop = None
        self._closing = False
        
        # Realtime API Config
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

    async def connect(self):
        """
        Async connection loop.
        Must be called with await or loop.run_until_complete().
        """
        self.loop = asyncio.get_running_loop()
        self._closing = False
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        print(f"🔗 Connecting to OpenAI Realtime API...")

        try:
            async with websockets.connect(self.url, additional_headers=headers) as ws:
                self.ws = ws
                print("✅ Connected to OpenAI Realtime API")
                
                await self._send_session_update()

                async for message in ws:
                    if self._closing:
                        break
                    await self._handle_message(message)
                    
        except asyncio.CancelledError:
            print("🔌 OpenAI connection cancelled")
        except Exception as e:
            if not self._closing:
                print(f"❌ OpenAI Connection Error: {e}")
                self.on_error(f"Connection failed: {e}")
        finally:
            self.ws = None
            print("OpenAI Connection Closed")

    async def disconnect(self):
        """Gracefully close the WebSocket connection."""
        self._closing = True
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

    async def _send_session_update(self):
        if not self.ws: return
        
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700
                }
            }
        }
        await self.ws.send(json.dumps(session_update))
        print("📤 Sent session configuration to OpenAI Realtime API")

    async def _handle_message(self, message):
        try:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "conversation.item.input_audio_transcription.completed":
                text = data.get("transcript", "")
                if text and text.strip():
                    self.on_transcript(text)
            
            elif event_type == "conversation.item.input_audio_transcription.delta":
                pass
                
            elif event_type == "response.audio_transcript.done":
                text = data.get("transcript", "")
                if text and text.strip():
                    print(f"🤖 [AI Response]: {text}")
            
            elif event_type == "session.created":
                print("✅ OpenAI Realtime session created")
                
            elif event_type == "session.updated":
                print("✅ OpenAI Realtime session configured")
                
            elif event_type == "error":
                err = data.get("error", {})
                err_msg = err.get("message", "Unknown error")
                err_code = err.get("code", "unknown")
                print(f"❌ OpenAI API Error [{err_code}]: {err_msg}")
                self.on_error(f"API Error: {err_msg}")
                
            elif event_type == "input_audio_buffer.speech_started":
                print("🎤 Speech detected in audio buffer")
                
            elif event_type == "input_audio_buffer.speech_stopped":
                print("🎤 Speech ended, processing...")
                
            elif event_type == "input_audio_buffer.committed":
                print("📝 Audio buffer committed for transcription")

        except Exception as e:
            print(f"Message Parse Error: {e}")
            import traceback
            traceback.print_exc()

    async def send_audio_chunk(self, audio_bytes):
        """
        Async method to send audio chunks to the OpenAI Realtime API.
        """
        if self.ws and not self._closing:
            try:
                encoded = base64.b64encode(audio_bytes).decode("utf-8")
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": encoded
                }
                await self.ws.send(json.dumps(event))
            except Exception as e:
                if not self._closing:
                    print(f"Send failed: {e}")