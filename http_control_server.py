"""
http_control_server.py — Lightweight HTTP control server for desktop_mon_gemini.

Port: 8005
  GET  /health   → {"status": "ok"} — launcher health check
  POST /shutdown → triggers clean app shutdown from the launcher

Runs in a daemon thread alongside the tkinter GUI.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

CONTROL_PORT = 8005

_shutdown_callback = None
_server: HTTPServer | None = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access logs

    def _send_json(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "desktop_monitor", "port": CONTROL_PORT})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send_json(200, {"status": "shutting_down"})
            if _shutdown_callback:
                # Schedule via tkinter so it runs on the main thread
                threading.Thread(target=_shutdown_callback, daemon=True).start()
        else:
            self._send_json(404, {"error": "not found"})


def start(shutdown_callback):
    """
    Start the HTTP control server in a daemon thread.
    shutdown_callback — called when POST /shutdown is received.
    """
    global _shutdown_callback, _server
    _shutdown_callback = shutdown_callback

    _server = HTTPServer(("0.0.0.0", CONTROL_PORT), _Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True, name="HTTPControl")
    t.start()
    print(f"✅ HTTP control server on :{CONTROL_PORT}  (/health, /shutdown)")


def stop():
    global _server
    if _server:
        _server.shutdown()
        _server = None