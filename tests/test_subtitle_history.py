import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.history import SubtitleHistory


BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


class SubtitleHistoryTest(unittest.TestCase):
    def test_shorter_fragment_does_not_replace_complete_text(self):
        history = SubtitleHistory()
        history.update([{"text": "i love you", "bbox": BOX, "confidence": 0.80}], now=1.0)
        history.update([{"text": "i love", "bbox": BOX, "confidence": 0.95}], now=1.2)

        self.assertEqual(history.get_stable_text(["i love"], now=1.2), "i love you")

    def test_longer_text_replaces_fragment_even_with_slightly_lower_confidence(self):
        history = SubtitleHistory()
        history.update([{"text": "i love", "bbox": BOX, "confidence": 0.92}], now=1.0)
        history.update([{"text": "i love you", "bbox": BOX, "confidence": 0.86}], now=1.2)

        self.assertEqual(history.get_stable_text(["i love you"], now=1.2), "i love you")

    def test_single_word_fragment_uses_recent_complete_text(self):
        history = SubtitleHistory()
        history.update([{"text": "love", "bbox": BOX, "confidence": 0.95}], now=1.0)
        history.update([{"text": "i love you", "bbox": BOX, "confidence": 0.80}], now=1.2)

        self.assertEqual(history.get_stable_text(["love"], now=1.2), "i love you")


if __name__ == "__main__":
    unittest.main()
