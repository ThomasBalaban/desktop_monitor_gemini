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
        self.running = True  # Track running state for clean shutdown

    def start(self):
        """Starts the WebSocket server in a new thread."""
        thread = threading.Thread(target=self._run_server_in_thread, daemon=True)
        thread.start()
        
        # Start Heartbeat Thread
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        print(f"WebSocket server for AI clients starting on ws://localhost:{WEBSOCKET_PORT}")

    def stop(self):
        """Cleanly stops the WebSocket server and heartbeat logic."""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        print("WebSocket server stopped.")

    def _heartbeat_loop(self):
        """Sends a pulse every 5 seconds so clients know we are alive."""
        while self.running:
            time.sleep(5)
            if self.running:
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
            welcome_message = {
                "type": "connection_established",
                "timestamp": asyncio.get_event_loop().time(),
                "message": "Connected to Gemini Screen Watcher WebSocket"
            }
            await websocket.send(json.dumps(welcome_message))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        response = {
                            "type": "pong",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                        await websocket.send(json.dumps(response))
                except Exception as e:
                    print(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected normally")
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)

    def _is_websocket_open(self, websocket):
        """Check if websocket is open."""
        try:
            if hasattr(websocket, 'closed'):
                return not websocket.closed
            elif hasattr(websocket, 'state'):
                return str(websocket.state) == "State.OPEN"
            return True
        except:
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
                except:
                    clients_to_remove.add(client)
            
            for client in clients_to_remove:
                self.connected_clients.discard(client)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def broadcast(self, data):
        """Thread-safely broadcasts data to all connected clients."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast_coro(data), self.loop)