import asyncio
import json
import threading
import websockets # type: ignore

WEBSOCKET_PORT = 8001

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
        print(f"WebSocket server for AI clients starting on ws://localhost:{WEBSOCKET_PORT}")

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
            # This prevents the connection from closing immediately
            async for message in websocket:
                try:
                    # Echo back any messages received (optional)
                    data = json.loads(message)
                    print(f"Received message from client: {data}")
                    
                    # You can handle different message types here if needed
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
            # Handle different ways websockets library exposes connection state
            if hasattr(websocket, 'closed'):
                return not websocket.closed
            elif hasattr(websocket, 'close_code'):
                return websocket.close_code is None
            elif hasattr(websocket, 'state'):
                # For newer websockets versions
                return str(websocket.state) == "State.OPEN"
            else:
                # Fallback - assume it's open if we have a websocket object
                return True
        except Exception as e:
            print(f"Error checking websocket state: {e}")
            return False

    async def _broadcast_coro(self, data):
        """The async coroutine that sends data to all clients."""
        if self.connected_clients:
            message = json.dumps(data)
            # Create a list of tasks for concurrent sending
            tasks = []
            clients_to_remove = set()
            
            for client in self.connected_clients.copy():
                try:
                    # Check if client is still connected using our compatibility method
                    if not self._is_websocket_open(client):
                        clients_to_remove.add(client)
                        continue
                    tasks.append(client.send(message))
                except Exception as e:
                    print(f"Error preparing to send to client: {e}")
                    clients_to_remove.add(client)
            
            # Remove disconnected clients
            for client in clients_to_remove:
                self.connected_clients.discard(client)
            
            # Send messages concurrently to all connected clients
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    print(f"Error broadcasting message: {e}")

    def broadcast(self, data):
        """
        Thread-safely broadcasts data to all connected clients.
        This method is called from the main application thread.
        """
        if self.loop and self.loop.is_running():
            # Schedule the async broadcast task on the server's event loop
            asyncio.run_coroutine_threadsafe(
                self._broadcast_coro(data),
                self.loop
            )