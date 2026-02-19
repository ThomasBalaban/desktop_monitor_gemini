#!/usr/bin/env python3
"""
Gemini Screen Watcher - Main Entry Point
"""
import sys
import os
import signal
import threading

# Change cwd to script directory so all relative imports work correctly.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from app_controller import AppController
import http_control_server

_app: AppController | None = None


def _shutdown(*_):
    """Called on SIGTERM, SIGINT, or POST /shutdown — safe from any thread."""
    global _app
    if _app:
        # app.stop() is thread-safe: it uses a lock and schedules GUI teardown
        # via root.after() so it's safe to call from a signal handler or HTTP thread.
        _app.stop()


def main():
    global _app

    _app = AppController()

    # Start the HTTP control server so the launcher can health-check and
    # issue a clean shutdown without relying on SIGTERM alone.
    http_control_server.start(shutdown_callback=_shutdown)

    # Handle SIGTERM (from launcher p.terminate()) and SIGINT (Ctrl-C).
    # On macOS, tkinter's mainloop does not respond to SIGTERM by default,
    # so we register an explicit handler that calls app.stop().
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    _app.run()   # blocks until GUI closes


if __name__ == "__main__":
    main()