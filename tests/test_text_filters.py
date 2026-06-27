import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.text_filters import (
    clean_ocr_text,
    detect_script,
    filter_by_language,
    is_overlay_echo,
    is_feedback_text,
    is_overlay_text,
    normalize_ocr,
    translation_key,
)


class TextFiltersTest(unittest.TestCase):
    def test_detect_script_prefers_specific_non_latin_scripts(self):
        self.assertEqual(detect_script("مرحبا"), "ar")
        self.assertEqual(detect_script("こんにちは"), "ja")
        self.assertEqual(detect_script("안녕하세요"), "ko")
        self.assertEqual(detect_script("你好"), "zh")
        self.assertEqual(detect_script("hello"), "en")

    def test_filter_by_language_rejects_overlay_language_mismatch(self):
        detections = [
            {"text": "hello there"},
            {"text": "ini adalah teks overlay"},
            {"text": "こんにちは"},
        ]

        self.assertEqual(filter_by_language(detections, "en"), [{"text": "hello there"}])

    def test_ocr_cleanup_keeps_real_text_but_removes_overlay_markers(self):
        self.assertEqual(clean_ocr_text("(id) [0.99] Hello"), "Hello")
        self.assertEqual(normalize_ocr("ThosewithPower..."), "Thosewith Power")
        self.assertEqual(normalize_ocr("The SupremeOnebestows ultimate blessing."), "The Supreme One bestows ultimate blessing")
        self.assertEqual(normalize_ocr("Throughcorruption.."), "Through corruption")
        self.assertEqual(translation_key("  Hello\n world  "), "hello world")
        self.assertTrue(is_overlay_text("(en) translated text"))
        self.assertTrue(is_feedback_text("[0.91] translated text"))
        self.assertFalse(is_feedback_text("actual subtitle text"))

    def test_feedback_filter_catches_malformed_overlay_prefixes(self):
        self.assertTrue(is_overlay_text("id) Dunia hanyalah ilusi"))
        self.assertTrue(is_overlay_text("(id).https://aka.ms/Pswindows"))
        self.assertTrue(is_overlay_text("(id)thttps:/aka.ms/Pswindows"))

    def test_overlay_echo_filter_catches_prefixless_fragments(self):
        displayed = [
            "(id) Yang Agung memberikan berkah tertinggi id) Satubatestow tertinggi berkat tertingc."
        ]

        self.assertTrue(is_overlay_echo("tertinggi berkat tertingc.", displayed))
        self.assertFalse(is_overlay_echo("The world is but an illusion", displayed))


if __name__ == "__main__":
    unittest.main()
