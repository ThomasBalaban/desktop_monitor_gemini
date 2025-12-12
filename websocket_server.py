import asyncio
import json
import threading
import websockets # type: ignore
import time

WEBSOCKET_PORT = 8003

class WebSocketServer:
    """Manages a WebSocket server to broadcast data to clients."""

    def __init__(self):
        self.connected_clients = set()
        self.loop = None
        self.server_task = None

    def start(self):
        """Starts the WebSocket server in a new thread."""
        thread = threading.Thread(target=self._run_server_in_thread, daemon=True)
        thread.start()
        
        # Start Heartbeat Thread (New)
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        print(f"WebSocket server for AI clients starting on ws://localhost:{WEBSOCKET_PORT}")

    def _heartbeat_loop(self):
        """Sends a pulse every 5 seconds so clients know we are alive."""
        while True:
            time.sleep(5)
            self.broadcast({
                "type": "heartbeat",
                "timestamp": time.time(),
                "status": "active"
            })

    def _run_server_in_thread(self):
        """Sets up and runs the asyncio event loop for the server."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start_server())
        finally:
            self.loop.close()

    async def _start_server(self):
        """The main async task that starts the websockets server."""
        async with websockets.serve(self._connection_handler, "localhost", WEBSOCKET_PORT):
            await asyncio.Future()  # Run forever

    async def _connection_handler(self, websocket, path=None):
        """Handles new client connections."""
        self.connected_clients.add(websocket)
        print(f"New AI client connected. Total clients: {len(self.connected_clients)}")
        
        try:
            # Send a welcome message to the newly connected client
            welcome_message = {
                "type": "connection_established",
                "timestamp": asyncio.get_event_loop().time(),
                "message": "Connected to Gemini Screen Watcher WebSocket"
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Keep the connection alive by listening for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        response = {
                            "type": "pong",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")
                except Exception as e:
                    print(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected normally")
        except Exception as e:
            print(f"Error in connection handler: {e}")
        finally:
            # Clean up when client disconnects
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
            print(f"AI client disconnected. Total clients: {len(self.connected_clients)}")

    def _is_websocket_open(self, websocket):
        """Check if websocket is open, handling different websockets library versions"""
        try:
            if hasattr(websocket, 'closed'):
                return not websocket.closed
            elif hasattr(websocket, 'close_code'):
                return websocket.close_code is None
            elif hasattr(websocket, 'state'):
                return str(websocket.state) == "State.OPEN"
            else:
                return True
        except Exception as e:
            print(f"Error checking websocket state: {e}")
            return False

    async def _broadcast_coro(self, data):
        """The async coroutine that sends data to all clients."""
        if self.connected_clients:
            message = json.dumps(data)
            tasks = []
            clients_to_remove = set()
            
            for client in self.connected_clients.copy():
                try:
                    if not self._is_websocket_open(client):
                        clients_to_remove.add(client)
                        continue
                    tasks.append(client.send(message))
                except Exception as e:
                    print(f"Error preparing to send to client: {e}")
                    clients_to_remove.add(client)
            
            for client in clients_to_remove:
                self.connected_clients.discard(client)
            
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    print(f"Error broadcasting message: {e}")

    def broadcast(self, data):
        """Thread-safely broadcasts data to all connected clients."""
        # 1. Create the coroutine object
        coro = self._broadcast_coro(data)
        
        # 2. Check if we have a valid loop running in the background thread
        if self.loop and self.loop.is_running():
            # 3. Safely hand it over to the thread
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            
            # 4. Optional: Print errors if the broadcast fails silently
            try:
                # We don't wait for the result to keep it non-blocking, 
                # but we can catch immediate errors if needed.
                pass 
            except Exception as e:
                print(f"Broadcast error: {e}")
        else:
            print("⚠️ Cannot broadcast: WebSocket loop is not running.")