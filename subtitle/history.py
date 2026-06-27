"""Subtitle history with fuzzy matching to stabilize OCR noise."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubtitleEntry:
    text: str
    best_text: str = ""
    best_conf: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 1
    bboxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def update(self, bbox: List[float], confidence: float, now: float, new_text: str = "") -> None:
        self.last_seen = now
        self.count += 1
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        if new_text and _should_replace_best_text(self.best_text, self.best_conf, new_text, confidence):
            self.best_text = new_text
            self.best_conf = confidence

    @property
    def lifetime(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def avg_bbox(self) -> List[float]:
        if not self.bboxes:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
        sum_x = [0.0, 0.0, 0.0, 0.0]
        sum_y = [0.0, 0.0, 0.0, 0.0]
        for bbox in self.bboxes:
            for i, (x, y) in enumerate(bbox):
                sum_x[i] += x
                sum_y[i] += y
        n = len(self.bboxes)
        return [[sum_x[i] / n, sum_y[i] / n] for i in range(4)]

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def stable_text(self) -> str:
        return self.best_text if self.best_text else self.text


def _text_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity — prevents long subtitles with shared chars from matching."""
    if not a or not b:
        return 0.0
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    intersection = len(a_words & b_words)
    union = len(a_words | b_words)
    return intersection / union if union > 0 else 0.0


def _word_count(text: str) -> int:
    return len(text.lower().split())


def _is_text_fragment(a: str, b: str) -> bool:
    """Return True when one text is a likely subtitle fragment of the other."""
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b or a == b:
        return False

    if _word_count(a) <= _word_count(b):
        short, long = a, b
    else:
        short, long = b, a

    padded_short = f" {short} "
    padded_long = f" {long} "
    if padded_short not in padded_long:
        return False

    short_words = _word_count(short)
    long_words = _word_count(long)
    return short_words >= 2 or (len(short) >= 4 and long_words <= 6)


def _should_replace_best_text(current: str, current_conf: float, new_text: str, new_conf: float) -> bool:
    if not current:
        return True

    current_words = _word_count(current)
    new_words = _word_count(new_text)
    if new_words > current_words:
        return new_conf >= current_conf - 0.20
    if new_words == current_words:
        return new_conf > current_conf
    return False


def _entry_y_center(e: SubtitleEntry) -> float:
    bbox = e.avg_bbox
    if bbox and len(bbox) >= 2:
        ys = [pt[1] for pt in bbox]
        return (min(ys) + max(ys)) / 2
    return 0.0


class SubtitleHistory:
    def __init__(self, max_history_seconds: float = 10.0):
        self.max_history = max_history_seconds
        self._entries: Dict[str, SubtitleEntry] = {}
        self._last_cleanup = time.time()

    def _find_similar(self, norm_text: str, threshold: float = 0.6) -> Optional[SubtitleEntry]:
        """Find existing entry with similar text (fuzzy match)."""
        for key, entry in self._entries.items():
            if _is_text_fragment(norm_text, key):
                return entry
            if _text_similarity(norm_text, key) >= threshold:
                return entry
        return None

    def update(self, detections: List[dict], now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()

        for det in detections:
            text = det["text"].strip()
            if not text:
                continue
            norm_text = text.lower()
            bbox = det["bbox"]
            conf = det["confidence"]

            entry = self._entries.get(norm_text)
            if entry is not None:
                entry.update(bbox, conf, now, new_text=text)
            else:
                similar = self._find_similar(norm_text)
                if similar is not None:
                    similar.update(bbox, conf, now, new_text=text)
                else:
                    self._entries[norm_text] = SubtitleEntry(
                        text=text,
                        best_text=text,
                        best_conf=conf,
                        first_seen=now,
                        last_seen=now,
                        count=1,
                        bboxes=[bbox],
                        confidences=[conf],
                    )

        if now - self._last_cleanup > 2.0:
            self._prune(now)
            self._last_cleanup = now

    def _prune(self, now: float) -> None:
        to_delete = [
            norm_text
            for norm_text, entry in self._entries.items()
            if now - entry.last_seen > self.max_history
        ]
        for norm_text in to_delete:
            del self._entries[norm_text]

    def get_entries(self) -> List[SubtitleEntry]:
        return list(self._entries.values())

    def get_recent_entries(self, now: Optional[float] = None, max_age: float = 1.0, min_count: int = 1) -> List[SubtitleEntry]:
        if now is None:
            now = time.time()
        entries = [
            e for e in self._entries.values()
            if now - e.last_seen < max_age and e.count >= min_count
        ]
        # Fall back to min_count=1 if nothing passes threshold (avoids blank overlay)
        if not entries:
            entries = [
                e for e in self._entries.values()
                if now - e.last_seen < max_age
            ]
        entries.sort(key=_entry_y_center)
        return entries

    def get_stable_text(self, detected_texts: list[str], now: Optional[float] = None, max_age: float = 1.0) -> str:
        """Get stable display text for current detections (uses best_text from history)."""
        if now is None:
            now = time.time()
        entries = []
        for text in detected_texts:
            norm = text.lower().strip()
            if not norm:
                continue
            entry = self._entries.get(norm)
            if entry is None:
                entry = self._find_similar(norm)
            if entry and now - entry.last_seen < max_age:
                entries.append(entry)
        entries.sort(key=_entry_y_center)
        return " ".join(e.stable_text.strip() for e in entries)

    def get_best_entry(self, scorer, now: Optional[float] = None) -> Optional[SubtitleEntry]:
        if now is None:
            now = time.time()
        entries = self.get_recent_entries(now=now, max_age=1.0)
        if not entries:
            return None
        return max(entries, key=scorer.score)
