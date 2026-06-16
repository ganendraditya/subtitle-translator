import sys
import time
import re
import traceback
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor, QFont
from PyQt6.QtCore import Qt, QObject, pyqtSignal

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine
from subtitle.history import SubtitleHistory
from subtitle.scorer import SubtitleScorer
from subtitle.motion import filter_motion_detections
from subtitle.detector import SubtitleDetector
from translate.engine import TranslationEngine
from config.settings import Settings
from ui.settings_dialog import SettingsDialog
import torch


class OverlayBridge(QObject):
    text_ready = pyqtSignal(str)


def bbox_center_y(det: dict) -> float:
    bbox = det["bbox"]
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2


# Unicode ranges for script detection
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_HIRAGANA = re.compile(r"[\u3040-\u309f]")
_KATAKANA = re.compile(r"[\u30a0-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_script(text: str) -> str:
    """Detect language script using Unicode ranges. Returns ISO 639-1 code."""
    if _ARABIC.search(text):
        return "ar"
    if _HIRAGANA.search(text) or _KATAKANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _CJK.search(text):
        return "zh"
    return "en"  # Latin-based (en/id/fr/de/es handled by settings)


def filter_by_language(detections: list[dict], source_lang: str) -> list[dict]:
    """Discard detections whose text doesn't match the expected source script."""
    # If source is Latin-based (en/id/fr/de/es), keep all Latin detections
    latin_sources = {"en", "id", "fr", "de", "es"}
    if source_lang in latin_sources:
        return [d for d in detections if detect_script(d["text"]) != "ja"
                and detect_script(d["text"]) != "ko"
                and detect_script(d["text"]) != "zh"
                and detect_script(d["text"]) != "ar"]
    # For non-Latin sources, keep only the matching script
    return [d for d in detections if detect_script(d["text"]) == source_lang]


class App:
    def __init__(self) -> None:
        self._settings = Settings.get_instance()
        self._capture = ScreenCapture()
        self._overlay_top: OverlayWindow | None = None
        self._overlay_bot: OverlayWindow | None = None
        self._translation: TranslationEngine | None = None
        self._history_top: SubtitleHistory | None = None
        self._history_bot: SubtitleHistory | None = None
        self._scorer: SubtitleScorer | None = None
        self._frame_height: int | None = None
        self._prev_gray = None
        self._detector: SubtitleDetector | None = None
        self._tray: QSystemTrayIcon | None = None
        self._toggle_action: QAction | None = None
        self._settings_action: QAction | None = None
        self._quit_action: QAction | None = None
        self._bridge_top: OverlayBridge | None = None
        self._bridge_bot: OverlayBridge | None = None
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

        self._overlay_top = OverlayWindow()
        self._overlay_top.set_position("top")
        self._overlay_top.hide()

        self._overlay_bot = OverlayWindow()
        self._overlay_bot.set_position("bottom")
        self._overlay_bot.hide()

        self._bridge_top = OverlayBridge()
        self._bridge_top.text_ready.connect(self._overlay_top.set_text)
        self._bridge_bot = OverlayBridge()
        self._bridge_bot.text_ready.connect(self._overlay_bot.set_text)

        OCREngine.use_paddle(True)
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

        self._settings_action = QAction("Settings...")
        self._settings_action.triggered.connect(self._show_settings)
        menu.addAction(self._settings_action)

        self._quit_action = QAction("Quit")
        self._quit_action.triggered.connect(QApplication.quit)
        menu.addAction(self._quit_action)

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
        self._overlay_top.show()
        self._overlay_bot.show()
        self._frame_height = None
        self._detector = SubtitleDetector(
            self._settings.get("yolo_model", "models/yolov8s-subtitle.pt")
        )
        self._detector.load()
        self._capture.register_callback(self._on_frame)
        self._capture.set_frame_interval(5)
        self._capture.start(target_fps=30)
        print("[App] Enabled")

    def _disable(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        self._toggle_action.setText("Enable")
        self._capture.stop()
        self._bridge_top.text_ready.emit("")
        self._overlay_top.hide()
        self._bridge_bot.text_ready.emit("")
        self._overlay_bot.hide()
        print("[App] Disabled")

    def _on_frame(self, frame) -> None:
        try:
            self._frame_count += 1
            now = time.perf_counter()

            if self._frame_height is None:
                self._frame_height = frame.shape[0]
                self._history_top = SubtitleHistory(max_history_seconds=5.0)
                self._history_bot = SubtitleHistory(max_history_seconds=5.0)
                self._scorer = SubtitleScorer(frame_height=self._frame_height)

            if not self._enabled:
                return

            if now - self._last_ocr_time >= 0.5:
                self._last_ocr_time = now
                ocr_start = time.perf_counter()
                crop = self._settings.get("capture_crop", {"left": 0.0, "right": 1.0})

                # YOLO subtitle region detection (optional if model available)
                yolo_boxes = None
                if self._detector and self._detector.is_loaded():
                    yolo_boxes = self._detector.detect(frame)

                detections = OCREngine.run(
                    frame, resize_scale=None,
                    conf_thresh=self._settings.get("conf_thresh", 0.75),
                    h_crop=(crop["left"], crop["right"]),
                    yolo_bboxes=yolo_boxes,
                )
                ocr_elapsed = time.perf_counter() - ocr_start
                self._ocr_latency_ms = round(ocr_elapsed * 1000)
                self._ocr_count += 1

                # Language filter: discard detections not matching source language
                src = self._settings.source_lang
                tgt = self._settings.target_lang
                detections = filter_by_language(detections, src)

                # Motion filter: reject static UI overlays (notifications, etc.)
                detections, self._prev_gray = filter_motion_detections(
                    detections, frame, self._prev_gray
                )

                # Split detections by screen half
                h = self._frame_height
                mid = h // 2
                top_dets = [d for d in detections if bbox_center_y(d) < mid]
                bot_dets = [d for d in detections if bbox_center_y(d) >= mid]

                same_lang = src == tgt

                # Process top region
                self._history_top.update(top_dets, now)
                if top_dets:
                    texts = [d["text"] for d in top_dets]
                    text = self._history_top.get_stable_text(texts, now=now)
                    if text:
                        translated = text if same_lang else self._translate(text, src, tgt)
                        self._bridge_top.text_ready.emit(f"({src}) {text}\n({tgt}) {translated}")
                    else:
                        self._bridge_top.text_ready.emit("")
                else:
                    self._bridge_top.text_ready.emit("")

                # Process bottom region
                self._history_bot.update(bot_dets, now)
                if bot_dets:
                    texts = [d["text"] for d in bot_dets]
                    text = self._history_bot.get_stable_text(texts, now=now)
                    if text:
                        translated = text if same_lang else self._translate(text, src, tgt)
                        self._bridge_bot.text_ready.emit(f"({src}) {text}\n({tgt}) {translated}")
                    else:
                        self._bridge_bot.text_ready.emit("")
                else:
                    self._bridge_bot.text_ready.emit("")

            elapsed = now - self._last_fps_time
            if elapsed >= 2.0:
                self._capture_fps = round(self._frame_count / elapsed)
                self._frame_count = 0
                self._ocr_count = 0
                self._last_fps_time = now
        except Exception:
            self._bridge_bot.text_ready.emit(f"Error: {traceback.format_exc()}")

    def _translate(self, text: str, src: str, tgt: str) -> str:
        try:
            results = self._translation.translate([text], source_lang=src, target_lang=tgt)
            return results[0]["translation"] if results else "[Error]"
        except RuntimeError:
            return "[Model loading...]"

    def _show_settings(self) -> None:
        dialog = SettingsDialog()
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            print(f"[App] Settings: {self._settings.source_lang} \u2192 {self._settings.target_lang}")

    def _cleanup(self) -> None:
        try:
            if self._enabled:
                self._disable()
            if self._capture:
                self._capture.stop()
        except Exception:
            pass
        OCREngine.shutdown()


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
