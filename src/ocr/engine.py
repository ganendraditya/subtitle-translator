"""OCREngine with auto GPU/CPU detection and optimized defaults."""

import easyocr
import numpy as np
from typing import Any
import torch


class OCREngine:
    """Wrapper around EasyOCR for screen text detection.

    Auto-detects GPU availability and adjusts inference parameters for CPU.
    """

    _ocr = None
    _use_gpu = False  # Will be detected on first use

    @classmethod
    def get_instance(cls) -> "easyocr.Reader":
        if cls._ocr is None:
            cls._use_gpu = torch.cuda.is_available()
            print(f"[OCREngine] CUDA: {cls._use_gpu}")
            cls._ocr = easyocr.Reader(["en"], gpu=cls._use_gpu)
        return cls._ocr

    @classmethod
    def get_scale(cls) -> float:
        """Return resize scale. GPU gets higher scale for accuracy, CPU lower for speed."""
        cls.get_instance()
        return 0.7 if cls._use_gpu else 0.5

    @staticmethod
    def run(
        frame: np.ndarray,
        resize_scale: float | None = None,
        conf_thresh: float = 0.7,
        crop_bottom: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Run OCR with EasyOCR.

        Args:
            frame: Full screen frame.
            resize_scale: Resize factor. Auto-chooses 0.7 (GPU) / 0.5 (CPU) if None.
            conf_thresh: Minimum confidence threshold.
            crop_bottom: Fraction of bottom to crop (0.15 = bottom 15%).
        """
        import cv2

        h = frame.shape[0]
        y1 = int(h * (1.0 - crop_bottom))
        frame = frame[y1:h, :]

        if resize_scale is None:
            resize_scale = OCREngine.get_scale()

        if resize_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

        ocr = OCREngine.get_instance()
        result = ocr.readtext(frame)

        detections = []
        for bbox, text, confidence in result:
            if confidence >= conf_thresh:
                detections.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence,
                })

        return detections
