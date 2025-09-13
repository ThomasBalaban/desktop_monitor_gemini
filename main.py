#!/usr/bin/env python3
"""
Gemini Screen Watcher - Main Entry Point
"""
import sys
import os

# Change the current working directory to the script's directory
# This ensures that all relative imports work correctly regardless of
# where the script is executed from.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from app_controller import AppController

def main():
    """Main entry point for the application"""
    app = AppController()
    app.run()

if __name__ == "__main__":
    main()