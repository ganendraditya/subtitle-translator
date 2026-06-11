import sys
import time
from PyQt6.QtWidgets import QApplication

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow


class App:
    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._overlay: OverlayWindow | None = None
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0

    def run(self) -> None:
        app = QApplication(sys.argv)
        
        self._overlay = OverlayWindow()
        self._overlay.set_text("Sprint 1: Initializing...")
        self._overlay.set_position("bottom")

        self._capture.register_callback(self._on_frame)
        self._capture.start(target_fps=30)

        self._overlay.set_text("Sprint 1: Capture + Overlay Running")
        print("[App] Sprint 1 running. Press Ctrl+C or close window to exit.")
        
        sys.exit(app.exec())

    def _on_frame(self, frame) -> None:
        """Called on every captured frame (Sprint 1: update FPS counter)."""
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count
            self._frame_count = 0
            self._last_fps_time = now
            if self._overlay:
                self._overlay.set_text(f"Sprint 1 | FPS: {self._fps}")

    def stop(self) -> None:
        self._capture.stop()


def main() -> None:
    app = App()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[App] Interrupted by user.")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
