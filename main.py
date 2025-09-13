#!/usr/bin/env python3
"""
Gemini Screen Watcher - Main Entry Point
"""
import sys
from app_controller import AppController

def main():
    """Main entry point for the application"""
    # Check if the --autostart flag is present
    autostart = "--autostart" in sys.argv
    app = AppController(autostart=autostart)
    app.run()

if __name__ == "__main__":
    main()