"""
Improved Streaming management for Screen Watcher with robust restart logic
"""

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
    """Manages the screen streaming process with improved restart reliability"""
    
    def __init__(self, screen_capture, gemini_client, fps=1, restart_interval=15):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.fps = fps
        self.restart_interval = restart_interval
        
        self.state = StreamingState.STOPPED
        self.session_start_time = None
        self.current_loop = None
        self.streaming_task = None
        self.listener_task = None
        self.restart_lock = threading.Lock()  # Prevent multiple simultaneous restarts
        
        # Callbacks
        self.status_callback = None
        self.restart_callback = None
    
    def set_status_callback(self, callback):
        """Set callback for status updates"""
        self.status_callback = callback
    
    def set_restart_callback(self, callback):
        """Set callback for restart notifications"""
        self.restart_callback = callback
    
    def _update_status(self, status, color="black"):
        """Update status through callback"""
        if self.status_callback:
            self.status_callback(status, color)
    
    def start_streaming(self):
        """Start the streaming process"""
        if self.state != StreamingState.STOPPED:
            print(f"Cannot start streaming - current state: {self.state}")
            return False
            
        self.state = StreamingState.CONNECTING
        self.session_start_time = time.time()
        
        # Start streaming in a separate thread
        def run_streaming():
            # Create new event loop for this thread
            self.current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.current_loop)
            try:
                self.current_loop.run_until_complete(self._run_streaming_session())
            except Exception as e:
                print(f"Streaming thread error: {e}")
                self.state = StreamingState.ERROR
                self._update_status("Error occurred", "red")
            finally:
                # Clean up loop
                try:
                    if not self.current_loop.is_closed():
                        self.current_loop.close()
                except Exception as e:
                    print(f"Error closing event loop: {e}")
                self.current_loop = None
                if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                    self.state = StreamingState.STOPPED
            
        streaming_thread = threading.Thread(target=run_streaming, daemon=True)
        streaming_thread.start()
        return True
    
    def stop_streaming(self):
        """Stop the streaming process"""
        print("=== STOP: Stopping streaming ===")
        
        # Set state to stopped to prevent restarts
        old_state = self.state
        self.state = StreamingState.STOPPED
        self.session_start_time = None
        
        # Cancel tasks and disconnect
        self._cleanup_async_tasks()
        
        self._update_status("Stopped", "red")
        return True
    
    def restart_streaming_session(self):
        """Restart the streaming session with improved reliability"""
        with self.restart_lock:  # Prevent multiple simultaneous restarts
            if self.state == StreamingState.RESTARTING:
                print("Restart already in progress, skipping...")
                return
            
            if self.state == StreamingState.STOPPED:
                print("Cannot restart - streaming is stopped")
                return
            
            print("=== RESTART: Initiating session restart ===")
            
            # Set restarting state
            old_state = self.state
            self.state = StreamingState.RESTARTING
            
            # Notify UI of restart
            if self.restart_callback:
                self.restart_callback()
            
            # Clean up current session
            self._cleanup_async_tasks()
            
            # Wait for cleanup to complete
            time.sleep(1.5)
            
            # Check if we're still supposed to be restarting
            if self.state != StreamingState.RESTARTING:
                print("Restart cancelled - state changed during cleanup")
                return
            
            # Reset state and start new session
            self.state = StreamingState.STOPPED
            success = self.start_streaming()
            
            if not success:
                print("Failed to restart streaming session")
                self.state = StreamingState.ERROR
                self._update_status("Restart failed", "red")
    
    def _cleanup_async_tasks(self):
        """Clean up async tasks and connections"""
        if self.current_loop and not self.current_loop.is_closed():
            try:
                # Cancel streaming and listener tasks
                if self.streaming_task and not self.streaming_task.done():
                    future = asyncio.run_coroutine_threadsafe(
                        self._cancel_task(self.streaming_task), 
                        self.current_loop
                    )
                    try:
                        future.result(timeout=2.0)
                    except Exception as e:
                        print(f"Error cancelling streaming task: {e}")
                
                if self.listener_task and not self.listener_task.done():
                    future = asyncio.run_coroutine_threadsafe(
                        self._cancel_task(self.listener_task), 
                        self.current_loop
                    )
                    try:
                        future.result(timeout=2.0)
                    except Exception as e:
                        print(f"Error cancelling listener task: {e}")
                
                # Disconnect from Gemini
                future = asyncio.run_coroutine_threadsafe(
                    self.gemini_client.disconnect(), 
                    self.current_loop
                )
                try:
                    future.result(timeout=3.0)
                    print("Disconnected from Gemini successfully")
                except Exception as e:
                    print(f"Disconnect timeout or error: {e}")
                    
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    async def _cancel_task(self, task):
        """Safely cancel an async task"""
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error while cancelling task: {e}")
    
    def get_time_until_restart(self):
        """Get time remaining until next restart"""
        if self.state not in [StreamingState.STREAMING] or not self.session_start_time:
            return None
            
        elapsed = time.time() - self.session_start_time
        remaining = max(0, self.restart_interval - elapsed)
        return remaining
    
    def should_restart(self):
        """Check if it's time to restart the session"""
        if not self.session_start_time or self.state != StreamingState.STREAMING:
            return False
        return (time.time() - self.session_start_time) >= self.restart_interval
    
    async def _run_streaming_session(self):
        """Run the complete streaming session with improved error handling"""
        try:
            self._update_status("Connecting...", "orange")
            
            # Connect to Gemini
            if await self.gemini_client.connect():
                if self.state == StreamingState.RESTARTING:
                    print("Session cancelled during connection")
                    return
                
                self.state = StreamingState.STREAMING
                self._update_status("Streaming...", "green")
                
                # Start listening for responses
                self.listener_task = asyncio.create_task(
                    self.gemini_client.listen_for_responses()
                )
                
                # Start streaming frames
                self.streaming_task = asyncio.create_task(self._streaming_loop())
                
                # Wait for either task to complete or for manual stop/restart
                try:
                    while (self.state == StreamingState.STREAMING and 
                           not self.streaming_task.done() and 
                           not self.listener_task.done()):
                        
                        # Check if it's time to restart
                        if self.should_restart():
                            print("=== RESTART: Time limit reached ===")
                            # Schedule restart in separate thread to avoid blocking
                            threading.Thread(
                                target=self.restart_streaming_session, 
                                daemon=True
                            ).start()
                            return
                        
                        # Wait a bit before checking again
                        await asyncio.sleep(0.5)
                    
                    # If we get here, a task completed or state changed
                    print(f"Streaming session ended - State: {self.state}")
                    
                except Exception as e:
                    print(f"Error in streaming session: {e}")
                    
            else:
                self.state = StreamingState.ERROR
                self._update_status("Connection failed", "red")
                
        except Exception as e:
            print(f"Streaming session error: {e}")
            import traceback
            traceback.print_exc()
            self.state = StreamingState.ERROR
            self._update_status("Error occurred", "red")
        finally:
            # Clean up tasks
            await self._cancel_task(self.streaming_task)
            await self._cancel_task(self.listener_task)
            
            # Clean up websocket connection
            try:
                await self.gemini_client.disconnect()
            except Exception as e:
                print(f"Error during final disconnect: {e}")
            
            self.streaming_task = None
            self.listener_task = None
            
            # Update status if we're not restarting
            if self.state not in [StreamingState.RESTARTING, StreamingState.STOPPED]:
                self.state = StreamingState.STOPPED
                self._update_status("Stopped", "red")
    
    async def _streaming_loop(self):
        """Main streaming loop with improved error handling"""
        frame_interval = 1.0 / self.fps
        
        try:
            print(f"Starting streaming loop with {self.fps} FPS (interval: {frame_interval}s)")
            frame_count = 0
            consecutive_failures = 0
            max_failures = 5
            
            while self.state == StreamingState.STREAMING:
                try:
                    frame_count += 1
                    print(f"Capturing frame {frame_count}...")
                    
                    # Check if we should continue
                    if self.state != StreamingState.STREAMING:
                        print("Streaming state changed, exiting loop")
                        break
                    
                    # Capture frame
                    frame = self.screen_capture.capture_frame()
                    if frame and self.state == StreamingState.STREAMING:
                        print("Frame captured successfully, converting to base64...")
                        base64_image = self.screen_capture.image_to_base64(frame)
                        print(f"Base64 conversion complete, sending to Gemini...")
                        
                        success = await self.gemini_client.send_image(base64_image)
                        if success:
                            print("Image sent successfully")
                            consecutive_failures = 0  # Reset failure counter
                        else:
                            consecutive_failures += 1
                            print(f"Failed to send image (failure {consecutive_failures}/{max_failures})")
                            
                            if consecutive_failures >= max_failures:
                                print("Too many consecutive failures, stopping streaming")
                                self.state = StreamingState.ERROR
                                break
                    else:
                        print("No frame captured or streaming stopped")
                        if self.state != StreamingState.STREAMING:
                            break
                        
                    # Sleep with cancellation support
                    try:
                        await asyncio.sleep(frame_interval)
                    except asyncio.CancelledError:
                        print("Streaming loop sleep cancelled")
                        break
                    
                except asyncio.CancelledError:
                    print("Streaming loop cancelled")
                    break
                except Exception as e:
                    consecutive_failures += 1
                    if self.state == StreamingState.STREAMING:
                        print(f"Error in streaming loop: {e} (failure {consecutive_failures}/{max_failures})")
                        if consecutive_failures >= max_failures:
                            print("Too many consecutive errors, stopping streaming")
                            self.state = StreamingState.ERROR
                            break
                        # Wait a bit before retrying
                        await asyncio.sleep(1.0)
                    else:
                        break
                    
        except Exception as e:
            print(f"Streaming loop error: {e}")
            import traceback
            traceback.print_exc()
            if self.state == StreamingState.STREAMING:
                self.state = StreamingState.ERROR
        finally:
            print("Streaming loop stopped")