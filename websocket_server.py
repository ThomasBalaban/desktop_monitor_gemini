import asyncio
import json
import threading
import websockets  # type: ignore
import time
import queue

WEBSOCKET_PORT = 8003

class WebSocketServer:
    """Manages a WebSocket server to broadcast data to clients."""

    def __init__(self):
        self.connected_clients = set()
        self.loop = None
        self.server_task = None
        self.running = True
        # NEW: Thread-safe message queue
        self.message_queue = queue.Queue()

    def start(self):
        """Starts the WebSocket server in a new thread."""
        thread = threading.Thread(target=self._run_server_in_thread, daemon=True)
        thread.start()
        
        print(f"WebSocket server for AI clients starting on ws://localhost:{WEBSOCKET_PORT}")

    def stop(self):
        """Cleanly stops the WebSocket server."""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        print("WebSocket server stopped.")

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
        # Start the queue processor
        processor_task = asyncio.create_task(self._process_message_queue())
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        async with websockets.serve(self._connection_handler, "localhost", WEBSOCKET_PORT):
            await asyncio.Future()  # Run forever

    async def _heartbeat_loop(self):
        """Sends a pulse every 5 seconds so clients know we are alive."""
        while self.running:
            await asyncio.sleep(5)
            if self.running and self.connected_clients:
                await self._do_broadcast({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "status": "active"
                })

    async def _process_message_queue(self):
        """Continuously processes messages from the thread-safe queue."""
        while self.running:
            try:
                # Check for messages with a small timeout
                try:
                    data = self.message_queue.get_nowait()
                    await self._do_broadcast(data)
                except queue.Empty:
                    pass
                await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting
            except Exception as e:
                print(f"Queue processor error: {e}")

    async def _connection_handler(self, websocket, path=None):
        """Handles new client connections."""
        self.connected_clients.add(websocket)
        client_count = len(self.connected_clients)
        print(f"New AI client connected. Total clients: {client_count}")
        
        try:
            welcome_message = {
                "type": "connection_established",
                "timestamp": time.time(),
                "message": "Connected to Gemini Screen Watcher WebSocket"
            }
            await websocket.send(json.dumps(welcome_message))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        response = {
                            "type": "pong",
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(response))
                except Exception as e:
                    print(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected normally")
        except Exception as e:
            print(f"Connection handler error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            print(f"Client removed. Total clients: {len(self.connected_clients)}")

    async def _do_broadcast(self, data):
        """Actually sends data to all connected clients."""
        if not self.connected_clients:
            return
            
        message = json.dumps(data)
        dead_clients = set()
        
        for client in self.connected_clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                dead_clients.add(client)
            except Exception as e:
                print(f"Broadcast error to client: {e}")
                dead_clients.add(client)
        
        # Clean up dead clients
        for client in dead_clients:
            self.connected_clients.discard(client)

    def broadcast(self, data):
        """Thread-safe method to queue a message for broadcast."""
        try:
            self.message_queue.put_nowait(data)
            # DEBUG: Uncomment to verify broadcasts are being queued
            # msg_type = data.get('type', 'unknown')
            # print(f"📤 [WS] Queued: {msg_type}")
        except queue.Full:
            print("⚠️ [WS] Message queue full, dropping message")