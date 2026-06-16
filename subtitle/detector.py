"""YOLOv8 subtitle region detector.

Detects subtitle text regions using a trained YOLOv8 model.
User trains with 1 class "subtitle" using their own labeled frames.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Any


class SubtitleDetector:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self.model_path and os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                print(f"[Detector] Loaded: {self.model_path}")
                return True
            except Exception as e:
                print(f"[Detector] Load failed: {e}")
                return False
        return False

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.35) -> list[dict[str, Any]]:
        if self._model is None:
            return []

        h, w = frame.shape[:2]
        # Resize for inference speed (640px)
        scale = 640 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        results = self._model(small, conf=conf_thresh, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            # Scale back to original frame coordinates
            x1 = int(x1 / scale) if scale < 1.0 else int(x1)
            y1 = int(y1 / scale) if scale < 1.0 else int(y1)
            x2 = int(x2 / scale) if scale < 1.0 else int(x2)
            y2 = int(y2 / scale) if scale < 1.0 else int(y2)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "confidence": float(b.conf[0]),
            })
        return detections

    def is_loaded(self) -> bool:
        return self._model is not None
