import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from main import App


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

    def test_empty_result_hides_after_one_second_without_detection(self):
        app = object.__new__(App)
        app._enabled = True
        app._bridge_top = FakeBridge()
        app._bridge_bot = FakeBridge()
        app._overlay_top = FakeOverlay()
        app._overlay_bot = FakeOverlay()
        app._hold_top = ""
        app._hold_top_time = 0.0
        app._hold_bot = "i love you"
        app._hold_bot_time = time.perf_counter() - 1.1
        app._last_shown_top = ""
        app._last_shown_bot = "i love you"

        App._on_worker_result(app, {
            "_capture_bounds": (0, 0, 1920, 1080),
            "top": None,
            "bot": None,
        })

        self.assertEqual(app._overlay_bot.hidden, 1)
        self.assertEqual(app._bridge_bot.text_ready.values, [""])


if __name__ == "__main__":
    unittest.main()
