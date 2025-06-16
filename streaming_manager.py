"""
Streaming management for Screen Watcher
"""

import asyncio
import time
import threading

class StreamingManager:
    """Manages the screen streaming process"""
    
    def __init__(self, screen_capture, gemini_client, fps=2, restart_interval=30):
        self.screen_capture = screen_capture
        self.gemini_client = gemini_client
        self.fps = fps
        self.restart_interval = restart_interval
        
        self.is_streaming = False
        self.session_start_time = None
        self.current_tasks = []
        
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
            asyncio.run(self._run_streaming_session())
            
        streaming_thread = threading.Thread(target=run_streaming, daemon=True)
        streaming_thread.start()
        return True
    
    def stop_streaming(self):
        """Stop the streaming process"""
        self.is_streaming = False
        self.session_start_time = None
        
        # Cancel any running async tasks
        for task in self.current_tasks:
            if not task.done():
                task.cancel()
        self.current_tasks.clear()
    
    def restart_streaming_session(self):
        """Restart the streaming session"""
        print("=== RESTART: Initiating session restart ===")
        
        if self.restart_callback:
            self.restart_callback()
        
        # Stop current streaming
        was_streaming = self.is_streaming
        self.is_streaming = False
        
        # Cancel tasks
        for task in self.current_tasks:
            if not task.done():
                print(f"Cancelling task: {task}")
                task.cancel()
        self.current_tasks.clear()
        
        # Disconnect Gemini client
        if self.gemini_client.websocket:
            print("Disconnecting Gemini client...")
            try:
                async def close_ws():
                    await self.gemini_client.disconnect()
                
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(close_ws())
                    else:
                        asyncio.run(close_ws())
                except Exception as e:
                    print(f"Error during websocket close: {e}")
            except Exception as e:
                print(f"Error in websocket cleanup: {e}")
        
        # Only restart if we were actually streaming
        if was_streaming:
            print("Scheduling restart after cleanup...")
            # Restart after short delay
            def restart_after_cleanup():
                print("Executing delayed restart...")
                if not self.is_streaming:  # Double-check we're still stopped
                    print("Starting new streaming session...")
                    self.start_streaming()
                else:
                    print("Already streaming, skipping restart")
            
            # Use threading timer for restart
            restart_timer = threading.Timer(0.5, restart_after_cleanup)
            restart_timer.start()
        else:
            print("Was not streaming, skipping restart")
    
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
                self.current_tasks.append(listener_task)
                
                # Start streaming frames
                streaming_task = asyncio.create_task(self._streaming_loop())
                self.current_tasks.append(streaming_task)
                
                # Wait for either task to complete
                try:
                    await asyncio.gather(listener_task, streaming_task, return_exceptions=True)
                except Exception as e:
                    print(f"Error in task gathering: {e}")
                
                # Check if we need to restart the streaming session
                if (self.is_streaming and self.session_start_time and 
                    (time.time() - self.session_start_time) >= self.restart_interval):
                    print("=== RESTART: Time limit reached, scheduling restart ===")
                    # Schedule restart on main thread
                    restart_timer = threading.Timer(0, self.restart_streaming_session)
                    restart_timer.start()
                    return  # Exit the session
                    
            else:
                self._update_status("Connection failed", "red")
                
        except Exception as e:
            print(f"Streaming session error: {e}")
            self._update_status("Error occurred", "red")
        finally:
            # Clean up tasks
            for task in self.current_tasks:
                if not task.done():
                    task.cancel()
            self.current_tasks.clear()
            
            # Clean up websocket connection
            await self.gemini_client.disconnect()
            
            # Check if this was a restart or regular stop
            restart_happened = (self.session_start_time and 
                              (time.time() - self.session_start_time) >= self.restart_interval)
            
            if not restart_happened and not self.is_streaming:
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
                    
                    # Check if we need to restart the streaming session
                    if (self.is_streaming and self.session_start_time and 
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
                            break
                    else:
                        print("No frame captured or streaming stopped")
                        
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