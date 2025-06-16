"""
Streaming management for Screen Watcher
"""

import asyncio
import time
import threading

class StreamingManager:
    """Manages the screen streaming process"""
    
    def __init__(self, screen_capture, gemini_client, fps=2, restart_interval=20):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.fps = fps
        self.restart_interval = restart_interval
        
        self.is_streaming = False
        self.session_start_time = None
        self.current_loop = None
        
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
        if self.is_streaming:
            return False
            
        self.session_start_time = time.time()
        self.is_streaming = True
        
        # Start streaming in a separate thread
        def run_streaming():
            # Create new event loop for this thread
            self.current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.current_loop)
            try:
                self.current_loop.run_until_complete(self._run_streaming_session())
            except Exception as e:
                print(f"Streaming thread error: {e}")
            finally:
                self.current_loop.close()
                self.current_loop = None
            
        streaming_thread = threading.Thread(target=run_streaming, daemon=True)
        streaming_thread.start()
        return True
    
    def stop_streaming(self):
        """Stop the streaming process"""
        print("=== STOP: Stopping streaming ===")
        self.is_streaming = False
        self.session_start_time = None
        
        # If we have a running loop, schedule the disconnect
        if self.current_loop and not self.current_loop.is_closed():
            try:
                # Schedule disconnect on the event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.gemini_client.disconnect(), 
                    self.current_loop
                )
                # Wait briefly for disconnect to complete
                try:
                    future.result(timeout=2.0)
                except Exception as e:
                    print(f"Disconnect timeout or error: {e}")
            except Exception as e:
                print(f"Error scheduling disconnect: {e}")
    
    def restart_streaming_session(self):
        """Restart the streaming session - simplified approach"""
        print("=== RESTART: Initiating session restart ===")
        
        # Notify UI of restart
        if self.restart_callback:
            self.restart_callback()
        
        # Simply stop and start again
        was_streaming = self.is_streaming
        if was_streaming:
            print("Stopping current session...")
            self.stop_streaming()
            
            # Wait a moment for cleanup, then restart
            def do_restart():
                print("Starting new session...")
                self.start_streaming()
            
            # Schedule restart after brief delay
            restart_timer = threading.Timer(1.0, do_restart)
            restart_timer.start()
    
    def get_time_until_restart(self):
        """Get time remaining until next restart"""
        if not self.is_streaming or not self.session_start_time:
            return None
            
        elapsed = time.time() - self.session_start_time
        remaining = max(0, self.restart_interval - elapsed)
        return remaining
    
    async def _run_streaming_session(self):
        """Run the complete streaming session"""
        try:
            self._update_status("Connecting...", "orange")
            
            # Connect to Gemini
            if await self.gemini_client.connect():
                self._update_status("Streaming...", "green")
                
                # Start listening for responses
                listener_task = asyncio.create_task(self.gemini_client.listen_for_responses())
                
                # Start streaming frames
                streaming_task = asyncio.create_task(self._streaming_loop())
                
                # Wait for either task to complete or restart time to be reached
                try:
                    done, pending = await asyncio.wait(
                        [listener_task, streaming_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel any pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                            
                except Exception as e:
                    print(f"Error in streaming session: {e}")
                
                # Check if we need to restart due to time limit
                if (self.is_streaming and self.session_start_time and 
                    (time.time() - self.session_start_time) >= self.restart_interval):
                    print("=== RESTART: Time limit reached, scheduling restart ===")
                    # Schedule restart from main thread
                    threading.Thread(target=self.restart_streaming_session, daemon=True).start()
                    return
                    
            else:
                self._update_status("Connection failed", "red")
                
        except Exception as e:
            print(f"Streaming session error: {e}")
            import traceback
            traceback.print_exc()
            self._update_status("Error occurred", "red")
        finally:
            # Clean up websocket connection
            try:
                await self.gemini_client.disconnect()
            except Exception as e:
                print(f"Error during disconnect: {e}")
            
            # Update status if we're not restarting
            if not self.is_streaming:
                self._update_status("Stopped", "red")
    
    async def _streaming_loop(self):
        """Main streaming loop"""
        frame_interval = 1.0 / self.fps
        
        try:
            print(f"Starting streaming loop with {self.fps} FPS (interval: {frame_interval}s)")
            frame_count = 0
            
            while self.is_streaming:
                try:
                    frame_count += 1
                    print(f"Capturing frame {frame_count}...")
                    
                    # Check if we need to restart due to time limit
                    if (self.session_start_time and 
                        (time.time() - self.session_start_time) >= self.restart_interval):
                        print("=== RESTART: Streaming loop detected time limit reached ===")
                        break  # Exit loop to trigger session restart
                    
                    # Capture frame
                    frame = self.screen_capture.capture_frame()
                    if frame and self.is_streaming:
                        print("Frame captured successfully, converting to base64...")
                        base64_image = self.screen_capture.image_to_base64(frame)
                        print(f"Base64 conversion complete, sending to Gemini...")
                        
                        success = await self.gemini_client.send_image(base64_image)
                        if success:
                            print("Image sent successfully")
                        else:
                            print("Failed to send image")
                            # Don't break immediately on send failure, try again
                            print("Continuing despite send failure...")
                    else:
                        print("No frame captured or streaming stopped")
                        if not self.is_streaming:
                            break
                        
                    print(f"Sleeping for {frame_interval}s...")
                    await asyncio.sleep(frame_interval)
                    
                except asyncio.CancelledError:
                    print("Streaming loop cancelled")
                    break
                except Exception as e:
                    if self.is_streaming:
                        print(f"Error in streaming loop: {e}")
                        import traceback
                        traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"Streaming loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Streaming loop stopped")