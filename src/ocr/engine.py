from paddleocr import PaddleOCR
import numpy as np
from typing import Any


class OCREngine:
    """Wrapper around PaddleOCR for screen text detection."""

    _ocr: PaddleOCR | None = None

    @classmethod
    def get_instance(cls) -> "PaddleOCR":
        """Get or create the singleton PaddleOCR instance."""
        if cls._ocr is None:
            # Try GPU first, fallback to CPU
            try:
                cls._ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=True,
                    show_log=False,
                )
                # Quick test to ensure GPU works
                dummy = np.zeros((32, 32, 3), dtype=np.uint8)
                cls._ocr.ocr(dummy, cls=False)
            except Exception as e:
                # Fallback to CPU
                print(f"[OCR] GPU initialization failed ({e}). Falling back to CPU.")
                cls._ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                )
        return cls._ocr

    @staticmethod
    def run(frame: np.ndarray, resize_scale: float = 0.5, conf_thresh: float = 0.7) -> list[dict[str, Any]]:
        """Run OCR with resize and confidence filtering."""
        import cv2

        # Resize frame
        if resize_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

        ocr = OCREngine.get_instance()
        result = ocr.ocr(frame, cls=False)

        detections = []
        for line in result[0] if result else []:
            bbox, (text, confidence) = line
            if confidence >= conf_thresh:
                detections.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence,
                })

        return detections
