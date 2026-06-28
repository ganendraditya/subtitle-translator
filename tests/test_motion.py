import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.motion import filter_motion_detections


class MotionFilterTest(unittest.TestCase):
    def test_static_title_card_text_in_subtitle_band_is_preserved(self):
        prev = np.zeros((100, 100), dtype=np.uint8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[45:55, 45:55] = 255
        dets = [{
            "text": "Episode One: The Fool",
            "bbox": [[30, 15], [70, 15], [70, 23], [30, 23]],
        }]

        filtered, _ = filter_motion_detections(dets, frame, prev)

        self.assertEqual(filtered, dets)

    def test_static_center_ui_text_can_still_be_filtered(self):
        prev = np.zeros((100, 100), dtype=np.uint8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = 255
        dets = [
            {"text": "UI", "bbox": [[40, 45], [60, 45], [60, 55], [40, 55]]},
            {"text": "Subtitle", "bbox": [[30, 80], [70, 80], [70, 88], [30, 88]]},
        ]

        filtered, _ = filter_motion_detections(dets, frame, prev)

        self.assertEqual(filtered, [dets[1]])


if __name__ == "__main__":
    unittest.main()
