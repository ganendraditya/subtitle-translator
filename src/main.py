import os
os.environ.setdefault("YOLO_AUTOINSTALL", "0")
import re
import sys
import time
import traceback
import threading
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap, QColor, QFont
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer

from capture.screen import ScreenCapture
from overlay.renderer import OverlayWindow
from ocr.engine import OCREngine
from subtitle.history import SubtitleHistory, _text_similarity
from subtitle.scorer import SubtitleScorer
from subtitle.motion import filter_motion_detections
from subtitle.detector import SubtitleDetector
from subtitle.capture_mode import can_enable_capture, mode_for_selected_hwnd, selected_hwnd_for_mode
from subtitle.text_filters import (
    clean_ocr_text,
    filter_by_language,
    is_feedback_text,
    is_overlay_echo,
    is_overlay_text,
    is_ui_noise_text,
    normalize_ocr,
    translation_key,
)
from translate.engine import TranslationEngine
from config.settings import Settings
from ui.settings_dialog import SettingsDialog
from ui.window_picker import WindowPickerDialog
import torch


def _resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to 'gpu' or 'cpu' based on CUDA availability."""
    if device_setting == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return device_setting


class OverlayBridge(QObject):
    text_ready = pyqtSignal(str)


class _WorkerSignal(QObject):
    result = pyqtSignal(object)


def bbox_center_y(det: dict) -> float:
    bbox = det["bbox"]
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2


HOLD_SECONDS = 1.0
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)")
_TERMINAL_PUNCT_RE = re.compile(r"([.!?]+)[\"')\]]*$")
_MONTH_ID = {
    "january": "Januari",
    "february": "Februari",
    "march": "Maret",
    "april": "April",
    "may": "Mei",
    "june": "Juni",
    "july": "Juli",
    "august": "Agustus",
    "september": "September",
    "october": "Oktober",
    "november": "November",
    "december": "Desember",
}
_MONTH_NAMES_RE = "|".join(_MONTH_ID)
_DATE_RE = re.compile(
    rf"^(?P<prefix>[\"'(\[]*)(?P<month>{_MONTH_NAMES_RE})\s*(?P<days>\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,\s*\d{{1,2}}(?:st|nd|rd|th)?)*)(?P<punct>[.!?]*)(?P<suffix>[\"')\]]*)$",
    re.IGNORECASE,
)


def _split_translation_segments(text: str) -> list[str]:
    segments = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    return [segment for segment in segments if segment]


def _postprocess_translation(source: str, translated: str, src: str, tgt: str) -> str:
    if src == "en" and tgt == "id" and not translated.startswith("["):
        translated = re.sub(r"\bChaos\b", "Kekacauan", translated)
        translated = re.sub(r"\bApakah I\b", "Apakah aku", translated)
        if translated.strip() == "I":
            translated = translated.replace("I", "Aku")

    source_punct = _TERMINAL_PUNCT_RE.search(source.strip())
    translated_punct = _TERMINAL_PUNCT_RE.search(translated.strip())
    if source_punct:
        punct = source_punct.group(1)
        if punct == "..." and (not translated_punct or translated_punct.group(1) != "..."):
            translated = re.sub(r"[.!?]+([\"')\]]*)$", r"\1", translated.rstrip())
            translated = f"{translated}..."
    return translated


def _translate_known_segment(segment: str, src: str, tgt: str) -> str | None:
    if src != "en" or tgt != "id":
        return None

    match = _DATE_RE.match(segment.strip())
    if match:
        month = _MONTH_ID[match.group("month").lower()]
        day_numbers = re.findall(r"\d{1,2}", match.group("days"))
        days = ", ".join(day_numbers)
        return f"{match.group('prefix')}{days} {month}{match.group('punct')}{match.group('suffix')}"

    return None


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
        self._hold_top = ""
        self._hold_top_time = 0.0
        self._hold_bot = ""
        self._hold_bot_time = 0.0
        self._last_shown_top = ""
        self._last_shown_bot = ""
        self._last_overlay_top = ""
        self._last_overlay_bot = ""
        self._translation_cache: dict[tuple[str, str, str], str] = {}
        self._detector: SubtitleDetector | None = None
        self._tray: QSystemTrayIcon | None = None
        self._toggle_action: QAction | None = None
        self._settings_action: QAction | None = None
        self._quit_action: QAction | None = None
        self._bridge_top: OverlayBridge | None = None
        self._bridge_bot: OverlayBridge | None = None
        self._ocr_timer: QTimer | None = None
        self._enabled = False
        self._worker_busy = False
        self._worker_signal = _WorkerSignal()
        self._worker_signal.result.connect(self._on_worker_result)

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

        device = _resolve_device(self._settings.get("device", "auto"))
        OCREngine.get_instance(device=device, lang=self._settings.source_lang)
        use_gpu = device != "cpu"
        self._translation = TranslationEngine(use_gpu=use_gpu)
        self._translation.preload_pair(self._settings.source_lang, self._settings.target_lang)

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

        self._window_action = QAction("Select Window...")
        self._window_action.triggered.connect(self._pick_window)
        menu.addAction(self._window_action)

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
        if not self._capture_target_ready():
            print("[App] Select a capture target first from the tray menu.")
            self._pick_window()
            if not self._capture_target_ready():
                print("[App] Enable cancelled; no capture target selected.")
                return
        self._enabled = True
        self._toggle_action.setText("Disable")
        self._overlay_top.show()
        self._overlay_bot.show()

        self._frame_height = None
        device = _resolve_device(self._settings.get("device", "auto"))
        self._detector = SubtitleDetector(
            self._settings.get("yolo_model", "models/yolov8s-subtitle.pt"),
            device=device,
        )
        self._detector.load()

        # Configure window capture if set
        hwnd = self._selected_capture_hwnd()
        self._capture.set_window(hwnd)

        self._capture.start()

        self._ocr_timer = QTimer()
        self._ocr_timer.timeout.connect(self._on_timer)
        self._ocr_timer.start(300)
        print(f"[App] Enabled (window={hwnd})")

    def _pick_window(self) -> None:
        current_hwnd = self._settings.get("window_hwnd")
        dialog = WindowPickerDialog(current_hwnd=current_hwnd)
        if dialog.exec() == WindowPickerDialog.DialogCode.Accepted:
            mode = mode_for_selected_hwnd(dialog.selected_hwnd)
            self._settings.set("capture_mode", mode)
            self._settings.set("window_hwnd", dialog.selected_hwnd)
            if self._enabled:
                self._capture.set_window(selected_hwnd_for_mode(mode, dialog.selected_hwnd))
            title = "Full Screen" if mode == "fullscreen" else f"hwnd={dialog.selected_hwnd}"
            print(f"[App] Capture target selected: {title}")

    def _capture_target_ready(self) -> bool:
        return can_enable_capture(self._settings.get("capture_mode"), self._settings.get("window_hwnd"))

    def _selected_capture_hwnd(self) -> int | None:
        return selected_hwnd_for_mode(self._settings.get("capture_mode"), self._settings.get("window_hwnd"))

    def _disable(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        self._toggle_action.setText("Enable")
        if self._ocr_timer:
            self._ocr_timer.stop()
            self._ocr_timer = None
        self._capture.stop()
        # Clear hold state so stale worker results don't re-show overlay
        self._hold_top = ""
        self._hold_top_time = 0.0
        self._hold_bot = ""
        self._hold_bot_time = 0.0
        self._last_shown_top = ""
        self._last_shown_bot = ""
        self._last_overlay_top = ""
        self._last_overlay_bot = ""
        self._bridge_top.text_ready.emit("")
        self._overlay_top.hide()
        self._bridge_bot.text_ready.emit("")
        self._overlay_bot.hide()
        print("[App] Disabled")

    def _on_timer(self) -> None:
        if not self._enabled or self._worker_busy:
            return
        frame = self._capture.grab()
        if frame is None:
            self._worker_signal.result.emit({
                "top": None,
                "bot": None,
                "_capture_bounds": self._capture.capture_bounds,
            })
            return
        capture_bounds = self._capture.capture_bounds
        self._worker_busy = True
        threading.Thread(target=self._worker_run, args=(frame, capture_bounds), daemon=True).start()

    def _worker_run(self, frame, capture_bounds=None) -> None:
        """Runs in background thread: YOLO + OCR + translate."""
        try:
            now = time.perf_counter()
            if self._frame_height != frame.shape[0]:
                self._frame_height = frame.shape[0]
                self._history_top = SubtitleHistory(max_history_seconds=5.0)
                self._history_bot = SubtitleHistory(max_history_seconds=5.0)
                self._scorer = SubtitleScorer(frame_height=self._frame_height)
                self._prev_gray = None

            crop = self._settings.get("capture_crop", {"left": 0.0, "right": 1.0})
            src = self._settings.source_lang
            tgt = self._settings.target_lang

            yolo_boxes = None
            if self._detector and self._detector.is_loaded():
                yolo_boxes = self._detector.detect(frame, conf_thresh=0.5)
                yolo_boxes = [b["bbox"] for b in yolo_boxes] if yolo_boxes else None

            ocr_groups = OCREngine.run(
                frame, resize_scale=None,
                conf_thresh=self._settings.get("conf_thresh", 0.85),
                h_crop=(crop["left"], crop["right"]),
                yolo_bboxes=yolo_boxes,
                device=_resolve_device(self._settings.get("device", "auto")),
                source_lang=src,
            )
            print(f"[Worker] yolo_boxes={len(yolo_boxes) if yolo_boxes else 0} ocr_groups={len(ocr_groups)} raw={[[d['text'] for d in g] for g in ocr_groups]}")
            # Reset prev_gray if frame size changed (window resize/move)
            if self._prev_gray is not None:
                curr_h, curr_w = frame.shape[:2]
                prev_h, prev_w = self._prev_gray.shape[:2]
                if curr_h != prev_h or curr_w != prev_w:
                    self._prev_gray = None
            # Flatten all groups for motion detection (single pass per frame)
            flat_dets = [d for group in ocr_groups for d in group]
            flat_dets, self._prev_gray = filter_motion_detections(flat_dets, frame, self._prev_gray)
            # Regroup: assign each detection back to its closest YOLO bbox
            if yolo_boxes:
                grouped = [[] for _ in yolo_boxes]
                for d in flat_dets:
                    dcy = bbox_center_y(d)
                    best_idx = 0
                    best_dist = float("inf")
                    for i, box in enumerate(yolo_boxes):
                        ys = [p[1] for p in box]
                        box_cy = (min(ys) + max(ys)) / 2
                        dist = abs(dcy - box_cy)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                    grouped[best_idx].append(d)
                ocr_groups = grouped
            else:
                # No YOLO — OCR already groups by fixed crop regions, just use filtered dets
                ocr_groups = [flat_dets] if flat_dets else []

            h = self._frame_height
            same_lang = src == tgt

            result = {"top": None, "bot": None, "_capture_bounds": capture_bounds}

            # Collect all detections per half (top/bottom), grouped by YOLO bbox
            top_texts = []
            top_sources = []
            bot_texts = []
            bot_sources = []
            top_history = self._history_top
            bot_history = self._history_bot
            recent_overlays = (self._last_overlay_top, self._last_overlay_bot)

            for group in ocr_groups:
                group = filter_by_language(group, src)

                cleaned = []
                for d in group:
                    t = d["text"].strip()
                    if is_overlay_text(t) or is_ui_noise_text(t) or is_overlay_echo(t, recent_overlays):
                        continue
                    t = clean_ocr_text(t)
                    t = normalize_ocr(t)
                    if is_feedback_text(t) or is_ui_noise_text(t) or is_overlay_echo(t, recent_overlays):
                        continue
                    if t and len(t) >= 3:
                        cleaned.append({**d, "text": t})
                if not cleaned:
                    continue

                avg_cy = sum(bbox_center_y(d) for d in cleaned) / len(cleaned)
                is_top = avg_cy < h * 0.5
                history = top_history if is_top else bot_history
                sorted_dets = sorted(cleaned, key=lambda d: bbox_center_y(d))
                history.update(sorted_dets, now)

                for d in sorted_dets:
                    text = d["text"].strip()
                    if not text or len(text) < 3:
                        continue
                    if is_feedback_text(text):
                        continue
                    norm = text.lower()
                    entry = history._entries.get(norm)
                    if entry is None:
                        entry = history._find_similar(norm)
                    if entry is None or now - entry.last_seen >= 1.0:
                        for key, e in history._entries.items():
                            if now - e.last_seen < 1.0 and len(key) > len(norm):
                                if norm in key or _text_similarity(norm, key) > 0.5:
                                    entry = e
                                    break
                    if entry and now - entry.last_seen < 1.0:
                        stable = entry.stable_text
                    else:
                        stable = text

                    if is_top:
                        top_texts.append(stable)
                        top_sources.append(text)
                    else:
                        bot_texts.append(stable)
                        bot_sources.append(text)

            # Build result: each half gets ONE overlay with ALL its texts joined
            print(f"[Worker] groups={len(ocr_groups)} top_texts={top_texts} bot_texts={bot_texts}")
            if top_texts:
                source = "\n".join(top_sources)
                joined = "\n".join(top_texts)
                translated = joined if same_lang else self._translate_lines(top_texts, src, tgt)
                overlay_text = f"({src}) {translated}" if same_lang else f"({tgt}) {translated}"
                result["top"] = {
                    "text": overlay_text,
                    "hold": source,
                    "hold_time": now,
                    "last_shown": joined,
                }
            if bot_texts:
                source = "\n".join(bot_sources)
                joined = "\n".join(bot_texts)
                translated = joined if same_lang else self._translate_lines(bot_texts, src, tgt)
                overlay_text = f"({src}) {translated}" if same_lang else f"({tgt}) {translated}"
                result["bot"] = {
                    "text": overlay_text,
                    "hold": source,
                    "hold_time": now,
                    "last_shown": joined,
                }

            self._worker_signal.result.emit(result)

        except Exception:
            err = traceback.format_exc()
            print(f"[Worker] Error: {err}")
        finally:
            self._worker_busy = False

    def _on_worker_result(self, result: dict) -> None:
        """Runs in main thread: update overlays from worker result."""
        # Ignore stale worker results if disabled
        if not self._enabled:
            return
        now = time.perf_counter()
        capture_bounds = result.get("_capture_bounds")

        for region_key, bridge, overlay, hold_key, hold_time_key, last_key, overlay_key in [
            ("top", self._bridge_top, self._overlay_top, "_hold_top", "_hold_top_time", "_last_shown_top", "_last_overlay_top"),
            ("bot", self._bridge_bot, self._overlay_bot, "_hold_bot", "_hold_bot_time", "_last_shown_bot", "_last_overlay_bot"),
        ]:
            if overlay:
                overlay.set_capture_bounds(capture_bounds)
            data = result.get(region_key)
            if data:
                new_text = data.get("last_shown", "")
                bridge.text_ready.emit(data["text"])
                overlay.set_bbox_cy(None)
                overlay.show()
                setattr(self, hold_key, data["hold"])
                setattr(self, hold_time_key, data["hold_time"])
                setattr(self, last_key, new_text)
                setattr(self, overlay_key, data["text"])
            else:
                held = getattr(self, hold_key)
                held_time = getattr(self, hold_time_key)
                if held and (now - held_time) < HOLD_SECONDS:
                    continue
                bridge.text_ready.emit("")
                overlay.hide()
                setattr(self, hold_key, "")
                setattr(self, hold_time_key, 0.0)
                setattr(self, last_key, "")
                setattr(self, overlay_key, "")

    def _translate(self, text: str, src: str, tgt: str) -> str:
        if self._translation is None:
            return "[Error]"
        key = (src, tgt, translation_key(text))
        cached = self._translation_cache.get(key)
        if cached is not None:
            return cached
        try:
            results = self._translation.translate([text], source_lang=src, target_lang=tgt)
            translated = results[0]["translation"] if results else "[Error]"
            if not translated.startswith("["):
                if len(self._translation_cache) > 256:
                    self._translation_cache.pop(next(iter(self._translation_cache)))
                self._translation_cache[key] = translated
            return translated
        except RuntimeError:
            return "[Model loading...]"

    def _translate_lines(self, lines: list[str], src: str, tgt: str) -> str:
        if self._translation is None:
            return "[Error]"

        line_segments = [_split_translation_segments(line) for line in lines]
        flat_segments = [segment for segments in line_segments for segment in segments]
        if not flat_segments:
            return ""

        translated_segments: list[str | None] = []
        missing: list[str] = []
        missing_indexes: list[int] = []
        for segment in flat_segments:
            known = _translate_known_segment(segment, src, tgt)
            if known is not None:
                translated_segments.append(known)
                print(f"[TranslateIO] {src}->{tgt} {segment!r} => {known!r}")
                continue

            key = (src, tgt, translation_key(segment))
            cached = self._translation_cache.get(key)
            if cached is None:
                translated_segments.append(None)
                missing.append(segment)
                missing_indexes.append(len(translated_segments) - 1)
            else:
                translated_segments.append(cached)

        if missing:
            try:
                results = self._translation.translate(missing, source_lang=src, target_lang=tgt)
            except RuntimeError:
                return "[Model loading...]"

            for idx, segment, result in zip(missing_indexes, missing, results):
                text = result.get("translation", "[Error]")
                text = _postprocess_translation(segment, text, src, tgt)
                print(f"[TranslateIO] {src}->{tgt} {segment!r} => {text!r}")
                translated_segments[idx] = text
                if not text.startswith("["):
                    if len(self._translation_cache) > 256:
                        self._translation_cache.pop(next(iter(self._translation_cache)))
                    self._translation_cache[(src, tgt, translation_key(segment))] = text

        translated_iter = iter(t or "[Error]" for t in translated_segments)
        translated_lines: list[str] = []
        for segments in line_segments:
            translated_lines.append(" ".join(next(translated_iter) for _ in segments))
        return "\n".join(translated_lines)

    def _show_settings(self) -> None:
        dialog = SettingsDialog()
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            print(f"[App] Settings: {self._settings.source_lang} \u2192 {self._settings.target_lang}")
            self._translation_cache.clear()
            self._hold_top = ""
            self._hold_top_time = 0.0
            self._hold_bot = ""
            self._hold_bot_time = 0.0
            self._last_shown_top = ""
            self._last_shown_bot = ""
            self._last_overlay_top = ""
            self._last_overlay_bot = ""
            self._prev_gray = None
            if self._translation:
                self._translation.preload_pair(self._settings.source_lang, self._settings.target_lang)
            if self._enabled:
                device = _resolve_device(self._settings.get("device", "auto"))
                self._detector = SubtitleDetector(
                    self._settings.get("yolo_model", "models/yolov8s-subtitle.pt"),
                    device=device,
                )
                self._detector.load()

    def _cleanup(self) -> None:
        try:
            self._disable()
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
