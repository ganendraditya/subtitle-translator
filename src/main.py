import sys
import time
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor, QFont
from PyQt6.QtCore import Qt

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine
from subtitle.history import SubtitleHistory
from subtitle.scorer import SubtitleScorer
from translate.engine import TranslationEngine
from config.settings import Settings
from ui.settings_dialog import SettingsDialog
import torch


class App:
    def __init__(self) -> None:
        self._settings = Settings.get_instance()
        self._capture = ScreenCapture()
        self._overlay: OverlayWindow | None = None
        self._translation: TranslationEngine | None = None
        self._history: SubtitleHistory | None = None
        self._scorer: SubtitleScorer | None = None
        self._frame_height: int | None = None
        self._tray: QSystemTrayIcon | None = None
        self._toggle_action: QAction | None = None
        self._enabled = False

        self._frame_count = 0
        self._ocr_count = 0
        self._last_fps_time = time.perf_counter()
        self._last_ocr_time = 0.0
        self._ocr_latency_ms = 0
        self._capture_fps = 0

    def run(self) -> None:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)
        app.aboutToQuit.connect(self._cleanup)

        self._overlay = OverlayWindow()
        self._overlay.set_position("bottom")
        self._overlay.hide()

        OCREngine.get_instance()
        use_gpu = torch.cuda.is_available()
        self._translation = TranslationEngine(use_gpu=use_gpu)

        self._create_tray()

        sys.exit(app.exec())

    def _create_tray(self) -> None:
        icon = self._make_icon()
        self._tray = QSystemTrayIcon(icon)
        self._tray.setToolTip("Subtitle Translator")

        menu = QMenu()
        self._toggle_action = QAction("Enable")
        self._toggle_action.triggered.connect(self._toggle)
        menu.addAction(self._toggle_action)

        menu.addSeparator()

        settings_action = QAction("Settings...")
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)

        quit_action = QAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.show()

        self._tray.activated.connect(self._on_tray_activated)

    def _make_icon(self) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 120, 212))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(2, 2, 28, 28, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "ST")
        painter.end()
        return QIcon(pixmap)

    def _on_tray_activated(self, reason: int) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle()

    def _toggle(self) -> None:
        if self._enabled:
            self._disable()
        else:
            self._enable()

    def _enable(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._toggle_action.setText("Disable")
        self._overlay.show()
        self._frame_height = None
        self._capture.register_callback(self._on_frame)
        self._capture.set_frame_interval(5)
        self._capture.start(target_fps=30)
        self._overlay.set_text("Subtitle Translator - Enabled")
        print("[App] Enabled")

    def _disable(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        self._toggle_action.setText("Enable")
        self._capture.stop()
        self._overlay.set_text("")
        self._overlay.hide()
        print("[App] Disabled")

    def _on_frame(self, frame) -> None:
        self._frame_count += 1
        now = time.perf_counter()

        if self._frame_height is None:
            self._frame_height = frame.shape[0]
            self._history = SubtitleHistory(max_history_seconds=10.0)
            self._scorer = SubtitleScorer(frame_height=self._frame_height)

        if now - self._last_ocr_time >= 1.0:
            self._last_ocr_time = now
            ocr_start = time.perf_counter()
            detections = OCREngine.run(frame, resize_scale=None, conf_thresh=self._settings.get("conf_thresh", 0.7))
            ocr_elapsed = time.perf_counter() - ocr_start
            self._ocr_latency_ms = round(ocr_elapsed * 1000)
            self._ocr_count += 1

            self._history.update(detections, now)

            best_entry = self._history.get_best_entry(self._scorer)
            if best_entry is not None:
                original_text = best_entry.text.strip()
                src = self._settings.source_lang
                tgt = self._settings.target_lang

                if src == tgt:
                    translated_text = original_text
                else:
                    try:
                        results = self._translation.translate(
                            [original_text], source_lang=src, target_lang=tgt
                        )
                        translated_text = results[0]["translation"] if results else "[Error]"
                    except RuntimeError as e:
                        translated_text = f"[Error: {e}]"

                overlay_text = (
                    f"OCR: {self._ocr_latency_ms}ms | FPS: {self._capture_fps}\n"
                    f"[{src}] {original_text}\n"
                    f"[{tgt}] {translated_text}"
                )
            else:
                overlay_text = (
                    f"OCR: {self._ocr_latency_ms}ms | FPS: {self._capture_fps}\n"
                    f"No subtitle detected"
                )
            self._overlay.set_text(overlay_text)

        elapsed = now - self._last_fps_time
        if elapsed >= 2.0:
            self._capture_fps = round(self._frame_count / elapsed)
            self._frame_count = 0
            self._ocr_count = 0
            self._last_fps_time = now

    def _show_settings(self) -> None:
        dialog = SettingsDialog()
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            print(f"[App] Settings: {self._settings.source_lang} \u2192 {self._settings.target_lang}")

    def _cleanup(self) -> None:
        if self._enabled:
            self._disable()
        if self._capture:
            self._capture.stop()


def main() -> None:
    app = App()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[App] Interrupted by user.")
    finally:
        app._cleanup()


if __name__ == "__main__":
    main()
