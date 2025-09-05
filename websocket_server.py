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

    async def _connection_handler(self, websocket, path):
        """Handles new client connections."""
        self.connected_clients.add(websocket)
        print(f"New AI client connected. Total clients: {len(self.connected_clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            print(f"AI client disconnected. Total clients: {len(self.connected_clients)}")

    async def _broadcast_coro(self, data):
        """The async coroutine that sends data to all clients."""
        if self.connected_clients:
            message = json.dumps(data)
            # Use asyncio.gather to send messages concurrently
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=False
            )

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