import asyncio
import json
import base64
import websockets
import time

class OpenAIRealtimeClient:
    def __init__(self, api_key, model="gpt-4o-realtime-preview", 
                 on_transcript=None, on_error=None):
        self.url = f"wss://api.openai.com/v1/realtime?model={model}"
        self.api_key = api_key
        self.websocket = None
        self.is_connected = False
        
        # Callbacks
        self.on_transcript = on_transcript
        self.on_error = on_error
        
        self.SAMPLE_RATE = 24000
        
        # Track audio state
        self._audio_bytes_sent = 0
        self._last_commit_time = 0
        self._commit_task = None
        self._min_audio_for_commit = 24000 * 2 * 0.15  # 150ms of audio at 24kHz, 16-bit (2 bytes per sample)
        
    async def connect(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            self.websocket = await websockets.connect(self.url, extra_headers=headers)
            self.is_connected = True
            print(f"✅ Connected to OpenAI Realtime API")
            
            asyncio.create_task(self._receive_loop())
            await self._update_session()
            
            # Start periodic commit task
            self._commit_task = asyncio.create_task(self._periodic_commit())
            
            return True
        except Exception as e:
            if self.on_error: self.on_error(f"Connection failed: {e}")
            return False

    async def _update_session(self):
        """Configure session for continuous transcription without VAD."""
        session_update = {
            "type": "session.update",
            "session": {
                # Enable Whisper transcription
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                # Disable VAD - we manually commit
                "turn_detection": None
            }
        }
        print("📤 Configuring Continuous Transcription Mode...")
        await self.websocket.send(json.dumps(session_update))

    async def _periodic_commit(self):
        """Periodically commit audio buffer to force transcription."""
        while self.is_connected:
            try:
                await asyncio.sleep(3.0)  # Check every 3 seconds
                
                # Only commit if we have enough audio buffered
                if self._audio_bytes_sent >= self._min_audio_for_commit:
                    if self.is_connected and self.websocket:
                        commit_event = {"type": "input_audio_buffer.commit"}
                        await self.websocket.send(json.dumps(commit_event))
                        # Reset counter after commit
                        self._audio_bytes_sent = 0
                        self._last_commit_time = time.time()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't spam errors - just silently retry
                pass

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
            # Track how much audio we've sent since last commit
            self._audio_bytes_sent += len(audio_bytes)
        except Exception as e:
            print(f"Error sending audio: {e}")

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                event = json.loads(message)
                event_type = event.get("type", "unknown")
                
                # ========== TRANSCRIPTION EVENTS ==========
                
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    if transcript and self.on_transcript:
                        self.on_transcript(transcript)
                
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    error = event.get("error", {})
                    error_msg = error.get("message", "Unknown transcription error")
                    # Only print if it's not a "buffer too small" error
                    if "buffer too small" not in error_msg.lower():
                        print(f"⚠️ Transcription failed: {error_msg}")
                
                # ========== BUFFER EVENTS ==========
                
                elif event_type == "input_audio_buffer.committed":
                    pass  # Successfully committed
                
                elif event_type == "input_audio_buffer.cleared":
                    self._audio_bytes_sent = 0
                
                # ========== ERROR EVENTS ==========
                
                elif event_type == "error":
                    error_msg = event.get("error", {}).get("message", "Unknown error")
                    # Filter out buffer too small errors - these are expected sometimes
                    if "buffer too small" not in error_msg.lower():
                        print(f"❌ OpenAI Error: {error_msg}")
                        if self.on_error: self.on_error(error_msg)
                
                # ========== SESSION EVENTS ==========
                
                elif event_type == "session.created":
                    print("📡 Session created")
                    
                elif event_type == "session.updated":
                    print("✅ Continuous transcription mode active")
                    
        except websockets.exceptions.ConnectionClosed:
            print("OpenAI WebSocket closed")
            self.is_connected = False
        except Exception as e:
            print(f"Receive loop error: {e}")
        finally:
            if self._commit_task:
                self._commit_task.cancel()