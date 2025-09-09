import asyncio
import time
import threading
from enum import Enum

class StreamingState(Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RESTARTING = "restarting"
    ERROR = "error"

class StreamingManager:
    def __init__(self, screen_capture, gemini_client, fps=1, restart_interval=30, debug_mode=False):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.fps = fps
        self.restart_interval = restart_interval
        self.debug_mode = debug_mode
        self.state = StreamingState.STOPPED
        self.session_start_time = None
        self.current_loop = None
        self.streaming_task = None
        self.listener_task = None
        self.health_check_task = None # New health check task
        self.restart_lock = threading.Lock()
        self.status_callback = None
        self.restart_callback = None
        self.error_callback = None # New error callback

    def debug_print(self, message):
        if self.debug_mode:
            print(f"[DEBUG] {message}")

    def info_print(self, message):
        print(message)

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_restart_callback(self, callback):
        self.restart_callback = callback

    def set_error_callback(self, callback):
        self.error_callback = callback

    def _update_status(self, status, color="black"):
        if self.status_callback:
            self.status_callback(status, color)

    def _report_error(self, message):
        if self.error_callback:
            self.error_callback(message)

    def start_streaming(self):
        if self.state != StreamingState.STOPPED:
            return False
        self.state = StreamingState.CONNECTING
        self.session_start_time = time.time()
        def run_streaming():
            self.current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.current_loop)
            try:
                self.current_loop.run_until_complete(self._run_streaming_session())
            except Exception as e:
                self.info_print(f"Streaming thread error: {e}")
                self.state = StreamingState.ERROR
                self._update_status("Error occurred", "red")
                self._report_error(f"Streaming thread error: {e}")
            finally:
                if self.current_loop and not self.current_loop.is_closed():
                    self.current_loop.close()
                self.current_loop = None
                if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                    self.state = StreamingState.STOPPED
        threading.Thread(target=run_streaming, daemon=True).start()
        return True

    def stop_streaming(self):
        self.info_print("Stopping streaming...")
        self.state = StreamingState.STOPPED
        self.session_start_time = None
        self._cleanup_async_tasks()
        self._update_status("Stopped", "red")
        return True

    def restart_streaming_session(self):
        with self.restart_lock:
            if self.state == StreamingState.RESTARTING or self.state == StreamingState.STOPPED:
                return
            self.info_print("Restarting streaming session...")
            self.state = StreamingState.RESTARTING
            if self.restart_callback:
                self.restart_callback()
            self._cleanup_async_tasks()
            time.sleep(1.5)
            if self.state != StreamingState.RESTARTING:
                return
            self.state = StreamingState.STOPPED
            if not self.start_streaming():
                self.info_print("Failed to restart streaming session")
                self.state = StreamingState.ERROR
                self._update_status("Restart failed", "red")
                self._report_error("Failed to restart streaming session")


    def _cleanup_async_tasks(self):
        if self.current_loop and not self.current_loop.is_closed():
            tasks = [self.streaming_task, self.listener_task, self.health_check_task]
            for task in tasks:
                if task and not task.done():
                    future = asyncio.run_coroutine_threadsafe(self._cancel_task(task), self.current_loop)
                    try:
                        future.result(timeout=2.0)
                    except Exception as e:
                        self.debug_print(f"Error cancelling a task: {e}")

            future = asyncio.run_coroutine_threadsafe(self.gemini_client.disconnect(), self.current_loop)
            try:
                future.result(timeout=3.0)
            except Exception as e:
                self.debug_print(f"Disconnect timeout or error: {e}")

    async def _cancel_task(self, task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run_streaming_session(self):
        try:
            self._update_status("Connecting...", "orange")
            if await self.gemini_client.connect():
                if self.state == StreamingState.RESTARTING: return
                self.state = StreamingState.STREAMING
                self._update_status("Streaming...", "green")
                self.listener_task = asyncio.create_task(self.gemini_client.listen_for_responses())
                self.streaming_task = asyncio.create_task(self._streaming_loop())
                self.health_check_task = asyncio.create_task(self._health_check_loop()) # Start health check
                await asyncio.gather(self.streaming_task, self.listener_task, self.health_check_task, return_exceptions=True)
            else:
                self.state = StreamingState.ERROR
                self._update_status("Connection failed", "red")
                self._report_error("Failed to connect to Gemini")
        except Exception as e:
            self.info_print(f"Streaming session error: {e}")
            self.state = StreamingState.ERROR
            self._update_status("Error occurred", "red")
            self._report_error(f"Streaming session error: {e}")
        finally:
            await self._cancel_task(self.streaming_task)
            await self._cancel_task(self.listener_task)
            await self._cancel_task(self.health_check_task)
            await self.gemini_client.disconnect()
            if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                self.state = StreamingState.STOPPED
                self._update_status("Stopped", "red")


    async def _streaming_loop(self):
        frame_interval = 1.0 / self.fps
        while self.state == StreamingState.STREAMING:
            try:
                frame = self.screen_capture.capture_frame()
                if frame:
                    base64_image = self.screen_capture.image_to_base64(frame)
                    await self.gemini_client.send_image(base64_image)
                await asyncio.sleep(frame_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.info_print(f"Error in streaming loop: {e}")
                self._report_error(f"Streaming loop error: {e}")
                self.state = StreamingState.ERROR
                break

    async def _health_check_loop(self):
        """Periodically checks the connection health and session duration."""
        while self.state == StreamingState.STREAMING:
            await asyncio.sleep(30) # Check every 30 seconds
            if self.state != StreamingState.STREAMING: break

            # Health Check 1: Session Duration
            if self.session_start_time and (time.time() - self.session_start_time) >= self.restart_interval:
                self.info_print("Session time limit reached, initiating restart.")
                threading.Thread(target=self.restart_streaming_session, daemon=True).start()
                break

            # Health Check 2: WebSocket Connection
            if not self.gemini_client.is_healthy():
                self.info_print("Connection health check failed, initiating restart.")
                self._report_error("Connection lost. Attempting to reconnect...")
                threading.Thread(target=self.restart_streaming_session, daemon=True).start()
                break