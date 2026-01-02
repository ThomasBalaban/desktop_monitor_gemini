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
        
        # Realtime API Config
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

    async def connect(self):
        """
        Async connection loop.
        Must be called with await or loop.run_until_complete().
        """
        self.loop = asyncio.get_running_loop()
        
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
                    await self._handle_message(message)
                    
        except Exception as e:
            print(f"❌ OpenAI Connection Error: {e}")
            self.on_error(f"Connection failed: {e}")
        finally:
            self.ws = None
            print("OpenAI Connection Closed")

    async def _send_session_update(self):
        if not self.ws: return
        
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        await self.ws.send(json.dumps(session_update))

    async def _handle_message(self, message):
        try:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "response.audio_transcript.done":
                text = data.get("transcript", "")
                if text:
                    self.on_transcript(text)
            
            elif event_type == "error":
                err_msg = data.get("error", {}).get("message", "Unknown error")
                self.on_error(f"API Error: {err_msg}")

        except Exception as e:
            print(f"Message Parse Error: {e}")

    # --- ROBUST THREAD-SAFE SENDING ---
    
    def send_audio_chunk(self, audio_bytes):
        """
        Thread-safe method called from PyAudio thread.
        Uses call_soon_threadsafe to avoid coroutine object issues.
        """
        if self.ws and self.loop:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            event = {
                "type": "input_audio_buffer.append",
                "audio": encoded
            }
            # Pass the raw data string to the helper
            data_str = json.dumps(event)
            self.loop.call_soon_threadsafe(self._schedule_send, data_str)

    def _schedule_send(self, data_str):
        """
        Helper that runs INSIDE the asyncio loop.
        It is safe to create tasks here.
        """
        if self.ws:
            asyncio.create_task(self._internal_send(data_str))

    async def _internal_send(self, data_str):
        """Actual async send."""
        try:
            await self.ws.send(data_str)
        except Exception as e:
            print(f"Send failed: {e}")