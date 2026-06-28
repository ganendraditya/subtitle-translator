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
    is_ui_noise_text,
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
        self.assertEqual(normalize_ocr("ThosewithPower..."), "Thosewith Power...")
        self.assertEqual(normalize_ocr("The world js but an jllusjon"), "The world is but an illusion")
        self.assertEqual(normalize_ocr("The world is but an illusjon"), "The world is but an illusion")
        self.assertEqual(normalize_ocr("The SupremeOnebestows ultimate blessing."), "The Supreme One bestows ultimate blessing.")
        self.assertEqual(normalize_ocr("Throughcorruption.."), "Through corruption..")
        self.assertEqual(normalize_ocr("Was it overwork?."), "Was it overwork?")
        self.assertEqual(normalize_ocr("The Supreme@ne bestows ultimate blessing"), "The Supreme One bestows ultimate blessing")
        self.assertEqual(normalize_ocr("June 21st, 22nd, 25th"), "June 21st, 22nd, 25th")
        self.assertEqual(normalize_ocr("June26th..."), "June 26th...")
        self.assertEqual(normalize_ocr("June21st.22nd.25th"), "June 21st, 22nd, 25th")
        self.assertEqual(normalize_ocr("January1st."), "January 1st.")
        self.assertEqual(normalize_ocr("May29th."), "May 29th.")
        self.assertEqual(normalize_ocr("June 21st, 22nd. 25th."), "June 21st, 22nd, 25th.")
        self.assertEqual(normalize_ocr("Wesuccessfully deciphered"), "We successfully deciphered")
        self.assertEqual(normalize_ocr("TodayisJune28th. Whyisn't there"), "Today is June 28th. Why isn't there")
        self.assertEqual(normalize_ocr("The Shepherd kneels insolemn reverence"), "The Shepherd kneels in solemn reverence")
        self.assertEqual(normalize_ocr("offering everylambindevotior"), "offering every lamb in devotion")
        self.assertEqual(normalize_ocr("towelcomethedivine'sdescent"), "to welcome the divine's descent")
        self.assertEqual(translation_key("  Hello\n world  "), "hello world")
        self.assertTrue(is_overlay_text("(en) translated text"))
        self.assertTrue(is_feedback_text("[0.91] translated text"))
        self.assertFalse(is_feedback_text("actual subtitle text"))

    def test_ui_noise_filter_rejects_browser_and_player_text(self):
        self.assertTrue(is_ui_noise_text("youtube.com is now full screen"))
        self.assertTrue(is_ui_noise_text("Exit Full Screen (Esc"))
        self.assertTrue(is_ui_noise_text("See what others said about this video while it was live"))
        self.assertTrue(is_ui_noise_text("Lord of Mysteries: The Clown - Episode 01 [English Sub]"))
        self.assertTrue(is_ui_noise_text("ganendraditya/subtitle-translator"))
        self.assertFalse(is_ui_noise_text("Episode One: The Fool"))

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
