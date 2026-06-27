import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translate.engine import TranslationEngine


class TranslationRoutesTest(unittest.TestCase):
    def setUp(self):
        self.engine = TranslationEngine(use_gpu=False)

    def test_japanese_uses_helsinki_jap_model_code(self):
        self.assertEqual(
            self.engine._model_name("en", "ja"),
            "Helsinki-NLP/opus-mt-en-jap",
        )
        self.assertEqual(
            self.engine._model_name("ja", "en"),
            "Helsinki-NLP/opus-mt-jap-en",
        )

    def test_korean_uses_tc_big_models(self):
        self.assertEqual(
            self.engine._model_name("en", "ko"),
            "Helsinki-NLP/opus-mt-tc-big-en-ko",
        )
        self.assertEqual(
            self.engine._model_name("ko", "en"),
            "Helsinki-NLP/opus-mt-tc-big-ko-en",
        )

    def test_non_english_targets_still_route_through_english(self):
        self.assertEqual(self.engine._route("id", "ko"), [("id", "en"), ("en", "ko")])
        self.assertEqual(self.engine._route("ko", "id"), [("ko", "en"), ("en", "id")])


if __name__ == "__main__":
    unittest.main()
