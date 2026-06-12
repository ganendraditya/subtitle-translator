import sys
import time
from PyQt6.QtWidgets import QApplication

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine
from subtitle.history import SubtitleHistory
from subtitle.scorer import SubtitleScorer
from translate.engine import TranslationEngine
import torch
import numpy as np
import cv2


class App:
    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._overlay: OverlayWindow | None = None
        self._history: SubtitleHistory | None = None
        self._scorer: SubtitleScorer | None = None
        self._frame_height: int | None = None

        self._frame_count = 0
        self._ocr_count = 0
        self._last_fps_time = time.perf_counter()
        self._last_ocr_time = 0.0
        self._ocr_latency_ms = 0
        self._capture_fps = 0

    def run(self) -> None:
        app = QApplication(sys.argv)

        self._overlay = OverlayWindow()
        self._overlay.set_text("Sprint 4: Initializing...")
        self._overlay.set_position("bottom")

        OCREngine.get_instance()  # Ensure singleton is initialized
        use_gpu = torch.cuda.is_available()
        self._translation_engine = TranslationEngine(use_gpu=use_gpu)

        self._capture.register_callback(self._on_frame)
        self._capture.set_frame_interval(5)  # Process 1 of every 5 frames
        self._capture.start(target_fps=30)

        print("[App] Sprint 4 running: Capture + OCR + Subtitle filtering + Translation.")
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
            detections = OCREngine.run(frame, resize_scale=None, conf_thresh=0.7)
            ocr_elapsed = time.perf_counter() - ocr_start
            self._ocr_latency_ms = round(ocr_elapsed * 1000)
            self._ocr_count += 1

            # Update history with detections
            self._history.update(detections, now)

            # Get best candidate
            best_entry = self._history.get_best_entry(self._scorer)
            if best_entry is not None:
                score = self._scorer.score(best_entry)
                original_text = best_entry.text.strip()
                
                # Translate the detected text
                try:
                    translated_results = self._translation_engine.translate([original_text], target_lang="en") # Target English for now
                    translated_text = translated_results[0]["translation"] if translated_results else "[Translation Error]"
                except RuntimeError as e:
                    translated_text = f"[Trans. Error: {e}]"
                
                overlay_text = (
                    f"Sprint 4 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms\n"
                    f"Original: \"{original_text}\"\n"
                    f"Translated: \"{translated_text}\""
                )
            else:
                overlay_text = (
                    f"Sprint 4 | Cap: {self._capture_fps} FPS | OCR: {self._ocr_latency_ms}ms\n"
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
