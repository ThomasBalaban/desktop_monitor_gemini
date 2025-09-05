# web_and_socket_server.py
import asyncio
import http.server
import json
import socketserver
import threading
import websockets # type: ignore

PORT = 8000
WEBSOCKET_PORT = 8001

class WebAndSocketServer:
    """Manages both the web server for the UI and the WebSocket server for AI clients."""

    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.connected_clients = set()
        self.httpd = None

    def _start_http_server(self):
        """Starts the HTTP server to serve the web interface."""
        handler = self._create_handler()
        self.httpd = socketserver.TCPServer(("", PORT), handler)
        print(f"Web interface available at http://localhost:{PORT}")
        self.httpd.serve_forever()

    def _create_handler(self):
        """Creates a request handler that has access to the config object."""
        config = self.config_loader
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.path = 'index.html'
                elif self.path == '/get_config':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        'api_key_configured': config.is_api_key_configured(),
                        'region_description': config.get_region_description(),
                        'settings_description': config.get_settings_description(),
                        'prompt': config.prompt
                    }
                    self.wfile.write(json.dumps(response).encode())
                    return
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
        return CustomHandler

    async def _websocket_handler(self, websocket, path):
        """Handles new WebSocket connections."""
        self.connected_clients.add(websocket)
        print(f"New AI client connected. Total clients: {len(self.connected_clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            print(f"AI client disconnected. Total clients: {len(self.connected_clients)}")

    async def _start_websocket_server(self):
        """Starts the WebSocket server."""
        print(f"WebSocket server for AI clients running on ws://localhost:{WEBSOCKET_PORT}")
        async with websockets.serve(self._websocket_handler, "localhost", WEBSOCKET_PORT):
            await asyncio.Future()  # run forever

    def run(self):
        """Runs both servers in separate threads."""
        http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        http_thread.start()
        asyncio.run(self._start_websocket_server())

    async def broadcast(self, data):
        """Sends data to all connected WebSocket clients."""
        if self.connected_clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=False
            )

    def shutdown(self):
        """Shuts down the HTTP server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print("HTTP server shut down.")