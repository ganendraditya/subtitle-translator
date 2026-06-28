import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from main import App, _postprocess_translation, _translate_known_segment


class FakeSignal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class FakeBridge:
    def __init__(self):
        self.text_ready = FakeSignal()


class FakeOverlay:
    def __init__(self):
        self.bounds = []
        self.bbox_cy = []
        self.shown = 0
        self.hidden = 0

    def set_capture_bounds(self, bounds):
        self.bounds.append(bounds)

    def set_bbox_cy(self, cy):
        self.bbox_cy.append(cy)

    def show(self):
        self.shown += 1

    def hide(self):
        self.hidden += 1


class FakeTranslation:
    def __init__(self):
        self.calls = []

    def translate(self, texts, source_lang, target_lang):
        self.calls.append((texts, source_lang, target_lang))
        return [{"translation": f"id:{text}"} for text in texts]


class OverlayUpdateTest(unittest.TestCase):
    def test_overlay_update_clears_dynamic_bbox_anchor(self):
        app = object.__new__(App)
        app._enabled = True
        app._bridge_top = FakeBridge()
        app._bridge_bot = FakeBridge()
        app._overlay_top = FakeOverlay()
        app._overlay_bot = FakeOverlay()
        app._hold_top = ""
        app._hold_top_time = 0.0
        app._hold_bot = ""
        app._hold_bot_time = 0.0
        app._last_shown_top = ""
        app._last_shown_bot = ""

        App._on_worker_result(app, {
            "_capture_bounds": (0, 0, 1920, 1080),
            "top": {
                "text": "(id) aku cinta kamu",
                "hold": "i love you",
                "hold_time": 1.0,
                "last_shown": "i love you",
            },
            "bot": None,
        })

        self.assertEqual(app._overlay_top.bbox_cy, [None])
        self.assertEqual(app._overlay_top.shown, 1)
        self.assertEqual(app._bridge_top.text_ready.values, ["(id) aku cinta kamu"])

    def test_empty_result_hides_after_hold_window_without_detection(self):
        app = object.__new__(App)
        app._enabled = True
        app._bridge_top = FakeBridge()
        app._bridge_bot = FakeBridge()
        app._overlay_top = FakeOverlay()
        app._overlay_bot = FakeOverlay()
        app._hold_top = ""
        app._hold_top_time = 0.0
        app._hold_bot = "i love you"
        app._hold_bot_time = time.perf_counter() - 0.8
        app._last_shown_top = ""
        app._last_shown_bot = "i love you"

        App._on_worker_result(app, {
            "_capture_bounds": (0, 0, 1920, 1080),
            "top": None,
            "bot": None,
        })

        self.assertEqual(app._overlay_bot.hidden, 0)
        self.assertEqual(app._bridge_bot.text_ready.values, [])

        app._hold_bot_time = time.perf_counter() - 1.1
        App._on_worker_result(app, {
            "_capture_bounds": (0, 0, 1920, 1080),
            "top": None,
            "bot": None,
        })

        self.assertEqual(app._overlay_bot.hidden, 1)
        self.assertEqual(app._bridge_bot.text_ready.values, [""])

    def test_translate_lines_keeps_multiline_subtitle_segments_separate(self):
        app = object.__new__(App)
        app._translation = FakeTranslation()
        app._translation_cache = {}

        translated = App._translate_lines(
            app,
            ["Did I have a brain hemorrhage?", "Was it overwork?"],
            "en",
            "id",
        )

        self.assertEqual(
            app._translation.calls,
            [(["Did I have a brain hemorrhage?", "Was it overwork?"], "en", "id")],
        )
        self.assertEqual(translated, "id:Did I have a brain hemorrhage?\nid:Was it overwork?")

    def test_translate_lines_splits_sentences_inside_one_ocr_line(self):
        app = object.__new__(App)
        app._translation = FakeTranslation()
        app._translation_cache = {}

        translated = App._translate_lines(
            app,
            ["Where am I? Is this a nightmare?"],
            "en",
            "id",
        )

        self.assertEqual(
            app._translation.calls,
            [(["Where am I?", "Is this a nightmare?"], "en", "id")],
        )
        self.assertEqual(translated, "id:Where am I? id:Is this a nightmare?")

    def test_postprocess_translation_restores_indonesian_terms_and_ellipsis(self):
        self.assertEqual(
            _postprocess_translation("Chaos stretches into eternity", "Chaos membentang ke keabadian", "en", "id"),
            "Kekacauan membentang ke keabadian",
        )
        self.assertEqual(_postprocess_translation("I", "I", "en", "id"), "Aku")
        self.assertEqual(_postprocess_translation("Did I.", "Apakah I.", "en", "id"), "Apakah aku.")
        self.assertEqual(
            _postprocess_translation("It hurts so much...", "Rasanya sangat sakit.", "en", "id"),
            "Rasanya sangat sakit...",
        )

    def test_translate_known_segment_handles_dates_without_marianmt(self):
        self.assertEqual(_translate_known_segment("January 1st.", "en", "id"), "1 Januari.")
        self.assertEqual(_translate_known_segment('"June 26th..."', "en", "id"), '"26 Juni..."')
        self.assertEqual(_translate_known_segment("June 21st, 22nd, 25th", "en", "id"), "21, 22, 25 Juni")


if __name__ == "__main__":
    unittest.main()
