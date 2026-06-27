"""Text cleanup and language filtering for subtitle OCR results."""

from __future__ import annotations

import re


_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_HIRAGANA = re.compile(r"[\u3040-\u309f]")
_KATAKANA = re.compile(r"[\u30a0-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_LANG_CODES = "en|id|ja|zh|ko|fr|de|es|ar"
_OVERLAY_PREFIX_RE = re.compile(
    rf"^\s*(?:\(\s*(?:{_LANG_CODES})\s*\)|(?:{_LANG_CODES})\s*\)|\(\s*(?:{_LANG_CODES})\s*[).:])\s*",
    re.IGNORECASE,
)
_CONF_RE = re.compile(r"\[\d+\.\d+\]\s*")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ID_COMMON = re.compile(
    r"\b(untuk|dengan|adalah|dari|dalam|tidak|akan|juga|telah|atau|karena|sudah|"
    r"lebih|sangat|bahwa|mereka|kami|kita|anda|orang|waktu|tahun|sebelum|setelah|"
    r"ini|itu|yang|dan|di|ke|pada|se|per|bagi|tentang|antara|serta|hanya|atau|"
    r"punya|bisa|harus|mau|sini|sana|begitu|begini|kini|kemarin|besok|sedang|"
    r"telah|sedang|baru|lalu|lagi|cuma|saja|bahkan|sekali|sungguh|yakni|ayat|"
    r"salat|karena|perintah|diturunkan)\b",
    re.IGNORECASE,
)


def detect_script(text: str) -> str:
    if _ARABIC.search(text):
        return "ar"
    if _HIRAGANA.search(text) or _KATAKANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _CJK.search(text):
        return "zh"
    return "en"


def filter_by_language(detections: list[dict], source_lang: str) -> list[dict]:
    """Filter detections by expected source language."""
    if source_lang == "en":
        result = []
        for detection in detections:
            text = detection["text"]
            if _CJK.search(text) or _ARABIC.search(text) or _HANGUL.search(text):
                continue
            if _HIRAGANA.search(text) or _KATAKANA.search(text):
                continue
            if _ID_COMMON.search(text):
                continue
            result.append(detection)
        return result

    if source_lang in {"id", "fr", "de", "es"}:
        return [
            detection
            for detection in detections
            if detect_script(detection["text"]) not in ("ja", "ko", "zh", "ar")
        ]

    return [detection for detection in detections if detect_script(detection["text"]) == source_lang]


def clean_ocr_text(text: str) -> str:
    """Remove overlay prefix patterns from OCR text."""
    cleaned = _OVERLAY_PREFIX_RE.sub("", text.strip())
    cleaned = _CONF_RE.sub("", cleaned)
    return cleaned.strip()


def translation_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_ocr(text: str) -> str:
    """Fix common OCR noise: trailing dots, char mistakes, and missing spaces."""
    normalized = text.strip()
    normalized = re.sub(r"[.\u2026]{2,}$", ".", normalized)
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"(?<=\w)@(?=\w)", "a", normalized)
    normalized = re.sub(r"(?<=\w)0(?=\w)", "o", normalized)
    normalized = re.sub(r"(?<=\w)1(?=\w)", "l", normalized)
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"\bOne(?=bestows\b)", "One ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bThrough(?=(corruption|depravity|darkness)\b)", "Through ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"(?<=[a-z])(?=of|the|to|in|is|it|at|on|an|or|and|for|not|but|can|has|was|are|be|by|do|he|if|me|my|no|so|up|we|as|at|in|on)\b",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.strip()


def is_overlay_text(text: str) -> bool:
    stripped = text.strip()
    return bool(_OVERLAY_PREFIX_RE.match(stripped) or _CONF_RE.search(stripped))


def is_feedback_text(text: str) -> bool:
    """Return true only when OCR text clearly came from our overlay output."""
    return is_overlay_text(text)


def _echo_key(text: str) -> str:
    cleaned = clean_ocr_text(text).lower()
    words = _WORD_RE.findall(cleaned)
    return " ".join(words)


def _word_similarity(a: str, b: str) -> float:
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def is_overlay_echo(text: str, recent_overlay_texts: list[str] | tuple[str, ...]) -> bool:
    """Detect OCR text that is likely a prefixless fragment of our own overlay."""
    candidate = _echo_key(text)
    if len(candidate) < 8:
        return False

    candidate_words = candidate.split()
    if len(candidate_words) < 2:
        return False

    for overlay in recent_overlay_texts:
        overlay_key = _echo_key(overlay or "")
        if not overlay_key:
            continue
        if candidate in overlay_key:
            return True
        if len(candidate_words) >= 3 and _word_similarity(candidate, overlay_key) >= 0.60:
            return True
    return False
