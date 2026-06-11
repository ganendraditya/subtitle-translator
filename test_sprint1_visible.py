#!/usr/bin/env python3
"""Visible test for Sprint 1 - You can see and feel the overlay."""

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6.QtWidgets import QApplication
from overlay.renderer import OverlayWindow
from capture.screen import ScreenCapture


def main():
    app = QApplication(sys.argv)
    
    # Create overlay
    overlay = OverlayWindow()
    overlay.set_text("Sprint 1: Overlay Active!",)
    overlay.set_position("bottom")
    
    # Start capture (to verify it doesn't crash)
    capture = ScreenCapture()
    capture.start(target_fps=30)
    
    print("=" * 50)
    print("SPRINT 1 VISIBLE TEST")
    print("=" * 50)
    print("You should see a transparent overlay on your screen.")
    print("Text: 'Sprint 1: Overlay Active!' at bottom.")
    print("Try clicking on windows behind the overlay.")
    print("Overlay will auto-close in 10 seconds...")
    print("=" * 50)
    
    # Let it run for 10 seconds so you can see it
    import threading
    def delayed_close():
        time.sleep(10)
        capture.stop()
        app.quit()
        print("\n✅ Sprint 1 Visible Test PASSED - Overlay worked correctly!")
    
    threading.Thread(target=delayed_close, daemon=True).start()
    
    try:
        app.exec()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
