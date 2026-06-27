import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.capture_mode import can_enable_capture, mode_for_selected_hwnd, selected_hwnd_for_mode


class CaptureModeTest(unittest.TestCase):
    def test_unset_capture_target_cannot_enable(self):
        self.assertFalse(can_enable_capture("unset", None))
        self.assertFalse(can_enable_capture(None, None))

    def test_window_mode_requires_hwnd(self):
        self.assertTrue(can_enable_capture("window", 1234))
        self.assertFalse(can_enable_capture("window", None))

    def test_fullscreen_mode_is_explicit(self):
        self.assertTrue(can_enable_capture("fullscreen", None))
        self.assertEqual(mode_for_selected_hwnd(None), "fullscreen")
        self.assertIsNone(selected_hwnd_for_mode("fullscreen", 1234))

    def test_window_selection_sets_window_mode(self):
        self.assertEqual(mode_for_selected_hwnd(1234), "window")
        self.assertEqual(selected_hwnd_for_mode("window", 1234), 1234)


if __name__ == "__main__":
    unittest.main()
