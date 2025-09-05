#!/usr/bin/env python3
"""
Gemini Screen Watcher - Main Entry Point
"""

from app_controller import AppController

def main():
    """Main entry point for the application"""
    app = AppController()
    app.run()

if __name__ == "__main__":
    main()