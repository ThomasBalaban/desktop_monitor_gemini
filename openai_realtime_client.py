import asyncio
import json
import websockets # type: ignore
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
        
        # Latency Management
        self.audio_accumulated_sec = 0.0
        # Reduced to 3.0s for faster updates during continuous singing
        self.FORCE_COMMIT_INTERVAL = 3.0  

    async def connect(self):
        """Async connection loop."""
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
        self._closing = True
        if self.ws:
            try: await self.ws.close()
            except: pass
            self.ws = None

    async def _send_session_update(self):
        if not self.ws: return
        
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en"
                },
                # TUNED FOR SINGING/FAST SPEECH
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.35,      # More sensitive (was 0.45)
                    "prefix_padding_ms": 300, 
                    "silence_duration_ms": 600 # Much faster commit (was 1000)
                }
            }
        }
        await self.ws.send(json.dumps(session_update))
        print("📤 Sent session config (Fast VAD 600ms, English)")

    async def _handle_message(self, message):
        try:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "conversation.item.input_audio_transcription.completed":
                text = data.get("transcript", "")
                self.audio_accumulated_sec = 0.0

                if text and text.strip():
                    cleaned = self._filter_transcript(text.strip())
                    if cleaned:
                        self.on_transcript(cleaned)
            
            elif event_type == "conversation.item.input_audio_transcription.delta":
                pass
                
            elif event_type == "input_audio_buffer.speech_stopped":
                # print("🎤 [VAD] Speech ended...")
                self.audio_accumulated_sec = 0.0
                
            elif event_type == "input_audio_buffer.committed":
                # print("📝 [VAD] Buffer committed")
                self.audio_accumulated_sec = 0.0
                
            elif event_type == "error":
                err = data.get("error", {})
                err_msg = err.get("message", "Unknown error")
                
                # Suppress race-condition errors
                if "buffer too small" in err_msg or "buffer only has" in err_msg:
                    return

                print(f"❌ OpenAI API Error: {err_msg}")
                self.on_error(f"API Error: {err_msg}")

        except Exception as e:
            print(f"Message Parse Error: {e}")

    def _filter_transcript(self, text):
        if not text: return None
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if len(text) > 0 and (ascii_chars / len(text)) < 0.7:
            print(f"🚫 Filtered non-English: {text}")
            return None
        return text
    
    async def send_audio_chunk(self, audio_bytes):
        if self.ws and not self._closing:
            try:
                # 1. Send Audio
                encoded = base64.b64encode(audio_bytes).decode("utf-8")
                append_event = {
                    "type": "input_audio_buffer.append",
                    "audio": encoded
                }
                await self.ws.send(json.dumps(append_event))
                
                # 2. Track Duration
                duration = len(audio_bytes) / 48000.0
                self.audio_accumulated_sec += duration
                
                # 3. Force Commit if Threshold Exceeded
                if self.audio_accumulated_sec >= self.FORCE_COMMIT_INTERVAL:
                    # print(f"⏱️ [Timer] Forcing commit...")
                    commit_event = {"type": "input_audio_buffer.commit"}
                    await self.ws.send(json.dumps(commit_event))
                    self.audio_accumulated_sec = 0.0
                
            except Exception as e:
                if not self._closing:
                    print(f"Send failed: {e}")