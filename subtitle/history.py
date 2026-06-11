"""Subtitle history tracking.

Stores information about detected text over time to determine stability
and lifetime.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubtitleEntry:
    """Information about a specific subtitle candidate."""
    text: str
    first_seen: float
    last_seen: float
    count: int = 1
    bboxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def update(self, bbox: List[float], confidence: float, now: float) -> None:
        """Update entry with a new detection."""
        self.last_seen = now
        self.count += 1
        self.bboxes.append(bbox)
        self.confidences.append(confidence)

    @property
    def lifetime(self) -> float:
        """Duration (seconds) the subtitle has been present."""
        return self.last_seen - self.first_seen

    @property
    def stability(self) -> float:
        """Fraction of frames where this text was detected (approx)."""
        # We don't have total frames, so use count as proxy; caller can compute.
        return float(self.count)

    @property
    def avg_bbox(self) -> List[float]:
        """Average bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."""
        if not self.bboxes:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
        # Convert to numpy for easy averaging? Avoid dependency.
        # Simple per-point average.
        # Each bbox is list of 4 points, each point is [x, y].
        # We'll sum across detections.
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


class SubtitleHistory:
    """Keeps track of subtitle candidates over time."""

    def __init__(self, max_history_seconds: float = 10.0):
        """
        Args:
            max_history_seconds: how long to keep inactive entries before dropping.
        """
        self.max_history = max_history_seconds
        self._entries: Dict[str, SubtitleEntry] = {}
        self._last_cleanup = time.time()

    def update(self, detections: List[dict], now: Optional[float] = None) -> None:
        """Update history with new detections from a frame.

        Args:
            detections: list of dicts from OCREngine.run, each containing
                'text', 'bbox', 'confidence'.
            now: current timestamp (defaults to time.time()).
        """
        if now is None:
            now = time.time()

        # Normalize text for grouping (lowercase, strip)
        for det in detections:
            text = det["text"].strip()
            if not text:
                continue
            norm_text = text.lower()
            bbox = det["bbox"]
            conf = det["confidence"]

            if norm_text in self._entries:
                entry = self._entries[norm_text]
                entry.update(bbox, conf, now)
            else:
                self._entries[norm_text] = SubtitleEntry(
                    text=text,  # keep original casing for display
                    first_seen=now,
                    last_seen=now,
                    count=1,
                    bboxes=[bbox],
                    confidences=[conf],
                )

        # Prune old entries
        if now - self._last_cleanup > 2.0:  # cleanup every 2 seconds
            self._prune(now)
            self._last_cleanup = now

    def _prune(self, now: float) -> None:
        """Remove entries that have not been seen for max_history seconds."""
        to_delete = [
            norm_text
            for norm_text, entry in self._entries.items()
            if now - entry.last_seen > self.max_history
        ]
        for norm_text in to_delete:
            del self._entries[norm_text]

    def get_entries(self) -> List[SubtitleEntry]:
        """Return a list of all current entries."""
        return list(self._entries.values())

    def get_best_entry(self, scorer) -> Optional[SubtitleEntry]:
        """Return the entry with the highest score according to scorer."""
        entries = self.get_entries()
        if not entries:
            return None
        return max(entries, key=scorer.score)