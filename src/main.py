import sys
import time
from PyQt6.QtWidgets import QApplication

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine
from subtitle.history import SubtitleHistory
from subtitle.scorer import SubtitleScorer
import numpy as np


class App:
    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._overlay: OverlayWindow | None = None
        self._ocr = OCREngine()
        self._history: SubtitleHistory | None = None
        self._scorer: SubtitleScorer | None = None
        self._frame_height: int | None = None

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
        self._overlay.set_text("Sprint 3: Initializing...")
        self._overlay.set_position("bottom")

        self._capture.register_callback(self._on_frame)
        self._capture.set_frame_interval(3)  # Process 1 of every 3 frames
        self._capture.start(target_fps=30)

        print("[App] Sprint 3 running: Capture + OCR + Subtitle filtering.")
        print("[App] Close the overlay window or press Ctrl+C to exit.")
        sys.stdout.flush()

        sys.exit(app.exec())

    def _on_frame(self, frame) -> None:
        self._frame_count += 1
        now = time.perf_counter()

        # Initialize scorer on first frame (need frame height)
        if self._frame_height is None:
            self._frame_height = frame.shape[0]
            self._history = SubtitleHistory(max_history_seconds=10.0)
            self._scorer = SubtitleScorer(frame_height=self._frame_height)

        # Run OCR once per second-ish
        if now - self._last_ocr_time >= 1.0:
            self._last_ocr_time = now
            ocr_start = time.perf_counter()
            detections = self._ocr.run(frame, resize_scale=0.5, conf_thresh=0.7)
            ocr_elapsed = time.perf_counter() - ocr_start
            self._ocr_latency_ms = round(ocr_elapsed * 1000)
            self._ocr_count += 1

            # Update history with detections
            self._history.update(detections, now)

            # Get best candidate
            best_entry = self._history.get_best_entry(self._scorer)
            if best_entry is not None:
                # Display the best subtitle text and its score
                score = self._scorer.score(best_entry)
                text = best_entry.text.strip()
                overlay_text = (
                    f"Sprint 3 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms\n"
                    f"Subtitle: [{score:.2f}] \"{text}\""
                )
            else:
                overlay_text = (
                    f"Sprint 3 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms\n"
                    f"No subtitle detected"
                )
            self._overlay.set_text(overlay_text)

        # FPS counter every 2 seconds
        elapsed = now - self._last_fps_time
        if elapsed >= 2.0:
            self._capture_fps = round(self._frame_count / elapsed)
            self._ocr_fps = round(self._ocr_count / elapsed * 2)
            self._frame_count = 0
            self._ocr_count = 0
            self._last_fps_time = now

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
