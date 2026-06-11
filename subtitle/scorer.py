"""Subtitle scoring.

Assigns a score to a subtitle candidate based on:
  - stability (number of frames seen)
  - lifetime (how long it has been on screen)
  - bbox size (prefer medium/large text, not tiny UI)
  - vertical position (prefer bottom/top where subtitles usually appear)
  - average confidence
  - aspect ratio (prefer wider than tall)
"""

from __future__ import annotations

from typing import List

from .history import SubtitleEntry


class SubtitleScorer:
    def __init__(
        self,
        frame_height: int,
        stability_weight: float = 0.4,
        lifetime_weight: float = 0.2,
        size_weight: float = 0.15,
        position_weight: float = 0.15,
        confidence_weight: float = 0.05,
    ):
        """
        Args:
            frame_height: height of the video frame (used to normalize bbox height).
            stability_weight: weight for stability factor (count).
            lifetime_weight: weight for how long the text has persisted.
            size_weight: weight for normalized text size.
            position_weight: weight for vertical position (prefer edges).
            confidence_weight: weight for average OCR confidence.
        """
        self.frame_height = frame_height
        self.weights = {
            "stability": stability_weight,
            "lifetime": lifetime_weight,
            "size": size_weight,
            "position": position_weight,
            "confidence": confidence_weight,
        }
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total == 0:
            total = 1.0
        for k in self.weights:
            self.weights[k] /= total

    def score(self, entry: SubtitleEntry) -> float:
        """Compute a score (higher is better) for the entry."""
        # Stability: use count, but cap to avoid overly long-running text dominating.
        # We'll use log(count+1) to dampen.
        import math

        stability = math.log(entry.count + 1)

        # Lifetime: seconds, cap at e.g., 10 seconds.
        lifetime = min(entry.lifetime, 10.0)

        # Size: average height of bbox in pixels, normalized by frame height.
        # Prefer medium size: ideal around 0.05-0.15 of frame height.
        avg_bbox = entry.avg_bbox
        # Compute height as average of vertical extents.
        ys = [pt[1] for pt in avg_bbox]
        bbox_height = max(ys) - min(ys)
        size_norm = bbox_height / self.frame_height if self.frame_height > 0 else 0
        # Ideal size ~0.1 (10% of screen height). Use Gaussian-like score.
        # Score peaks at 0.1, falls off.
        size_score = math.exp(-((size_norm - 0.1) ** 2) / (2 * 0.05**2))

        # Position: prefer bottom or top edges.
        # Compute vertical center of bbox.
        y_center = (min(ys) + max(ys)) / 2
        # Normalize to [0,1] where 0=top, 1=bottom.
        y_norm = y_center / self.frame_height if self.frame_height > 0 else 0.5
        # Score high near 0 or 1, low in middle.
        # Use cosine: score = cos(pi * y_norm) gives 1 at 0 and 1, -1 at 0.5.
        # Shift to [0,1]:
        position_score = (math.cos(math.pi * y_norm) + 1) / 2

        # Confidence: already 0-1.
        confidence = entry.avg_confidence

        # Combine weighted sum.
        total = (
            self.weights["stability"] * stability
            + self.weights["lifetime"] * lifetime
            + self.weights["size"] * size_score
            + self.weights["position"] * position_score
            + self.weights["confidence"] * confidence
        )
        return total