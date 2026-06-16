"""OCREngine with auto GPU/CPU, multi-region, and subtitle filtering."""

import re
import easyocr
import numpy as np
from typing import Any
import torch


# Patterns that look like UI noise rather than subtitles
_NOISE_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}\s*/\s*\d{1,2}:\d{2}$"),  # 10:00 / 30:00
    re.compile(r"^\d{1,2}:\d{2}:\d{2}$"),                  # 01:23:45
    re.compile(r"^\d{3,4}p$", re.IGNORECASE),               # 720p, 1080p
    re.compile(r"^\d+x\d+$"),                               # 1920x1080
]


class OCREngine:
    """Wrapper around EasyOCR for screen text detection with subtitle filtering."""

    _ocr = None
    _use_gpu = False

    @classmethod
    def get_instance(cls) -> "easyocr.Reader":
        if cls._ocr is None:
            cls._use_gpu = torch.cuda.is_available()
            print(f"[OCREngine] CUDA: {cls._use_gpu}")
            cls._ocr = easyocr.Reader(["en"], gpu=cls._use_gpu)
        return cls._ocr

    @classmethod
    def get_scale(cls) -> float:
        cls.get_instance()
        return 0.7 if cls._use_gpu else 0.5

    @staticmethod
    def is_subtitle_candidate(bbox: list, text: str, confidence: float, frame_h: int, frame_w: int) -> bool:
        """Light filter: only reject obvious UI noise (timecodes, edge garbage)."""
        if any(p.match(text.strip()) for p in _NOISE_PATTERNS):
            return False

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        bh = max(ys) - min(ys)
        h_ratio = bh / frame_h if frame_h > 0 else 0

        # Reject only truly tiny or truly huge text
        if h_ratio < 0.008 or h_ratio > 0.15:
            return False

        # Reject text at extreme horizontal edges (UI elements like volume, time)
        cx = (min(xs) + max(xs)) / 2
        if cx < frame_w * 0.03 or cx > frame_w * 0.97:
            return False

        return True

    @staticmethod
    def run(
        frame: np.ndarray,
        resize_scale: float | None = None,
        conf_thresh: float = 0.7,
        crop_regions: list[tuple[float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run OCR with subtitle filtering.

        Args:
            frame: Full screen frame.
            resize_scale: Resize factor (auto GPU=0.7 / CPU=0.5).
            conf_thresh: Minimum confidence.
            crop_regions: (top_frac, bottom_frac) tuples.
                Default: top 27% + bottom 27%.
        """
        import cv2

        if crop_regions is None:
            crop_regions = [(0, 0.27), (0.73, 1.0)]

        if resize_scale is None:
            resize_scale = OCREngine.get_scale()

        ocr = OCREngine.get_instance()
        h, w = frame.shape[:2]
        detections = []

        for top_frac, bottom_frac in crop_regions:
            y1 = int(h * top_frac)
            y2 = int(h * bottom_frac)
            if y1 >= y2:
                continue
            region = frame[y1:y2, :]

            if resize_scale != 1.0:
                region = cv2.resize(region, (0, 0), fx=resize_scale, fy=resize_scale)

            result = ocr.readtext(region)

            for bbox, text, confidence in result:
                if confidence < conf_thresh:
                    continue

                adjusted_bbox = [
                    [x / resize_scale, y / resize_scale + y1]
                    for x, y in bbox
                ]

                if not OCREngine.is_subtitle_candidate(adjusted_bbox, text, confidence, h, w):
                    continue

                detections.append({
                    "bbox": adjusted_bbox,
                    "text": text,
                    "confidence": confidence,
                })

        return detections
