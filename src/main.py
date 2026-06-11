import sys
import time
from PyQt6.QtWidgets import QApplication

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine


class App:
    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._overlay: OverlayWindow | None = None
        self._ocr = OCREngine()
        self._frame_count = 0
        self._ocr_count = 0
        self._last_fps_time = time.perf_counter()
        self._last_ocr_time = 0.0
        self._ocr_fps = 0
        self._capture_fps = 0
        self._ocr_latency_ms = 0

    def run(self) -> None:
        app = QApplication(sys.argv)

        self._overlay = OverlayWindow()
        self._overlay.set_text("Sprint 2: Initializing OCR...")
        self._overlay.set_position("bottom")

        self._capture.register_callback(self._on_frame)
        self._capture.set_frame_interval(3)  # Process 1 of every 3 frames
        self._capture.start(target_fps=30)

        print("[App] Sprint 2 running: Capture + OCR displayed on overlay.")
        print("[App] Close the overlay window or press Ctrl+C to exit.")

        sys.exit(app.exec())

    def _on_frame(self, frame) -> None:
        self._frame_count += 1
        now = time.perf_counter()

        # Run OCR once per second-ish
        if now - self._last_ocr_time >= 1.0:
            self._last_ocr_time = now
            ocr_start = time.perf_counter()
            detections = self._ocr.run(frame, resize_scale=0.5, conf_thresh=0.7)
            ocr_elapsed = time.perf_counter() - ocr_start
            self._ocr_latency_ms = round(ocr_elapsed * 1000)
            self._ocr_count += 1
            self._update_overlay(detections)

        # FPS counter every 2 seconds
        elapsed = now - self._last_fps_time
        if elapsed >= 2.0:
            self._capture_fps = round(self._frame_count / elapsed)
            self._ocr_fps = round(self._ocr_count / elapsed * 2)
            self._frame_count = 0
            self._ocr_count = 0
            self._last_fps_time = now

    def _update_overlay(self, detections: list) -> None:
        """Build overlay text from OCR results."""
        if not detections:
            self._overlay.set_text(
                f"Sprint 2 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms | "
                "[No text detected]"
            )
            return

        # Pick highest-confidence detection
        best = max(detections, key=lambda d: d["confidence"])
        text = best["text"].strip()

        overlay_text = (
            f"Sprint 2 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms\n"
            f"├─ [{best['confidence']:.2f}] \"{text}\""
        )

        if len(detections) > 1:
            overlay_text += f"\n└─ +{len(detections) - 1} more"

        self._overlay.set_text(overlay_text)

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
