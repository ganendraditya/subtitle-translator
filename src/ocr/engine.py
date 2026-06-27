"""OCREngine with auto GPU/CPU, multi-region, and subtitle filtering."""
from __future__ import annotations

import os
import re
import struct
import json
import base64
import subprocess
import sys

import numpy as np
from typing import Any

_NOISE_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}\s*/\s*\d{1,2}:\d{2}$"),
    re.compile(r"^\d{1,2}:\d{2}:\d{2}$"),
    re.compile(r"^\d{3,4}p$", re.IGNORECASE),
    re.compile(r"^\d+x\d+$"),
]


def is_subtitle_candidate(bbox: list, text: str, confidence: float, frame_h: int, frame_w: int) -> bool:
    if any(p.match(text.strip()) for p in _NOISE_PATTERNS):
        return False
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    bh = max(ys) - min(ys)
    bw = max(xs) - min(xs)
    h_ratio = bh / frame_h if frame_h > 0 else 0
    if h_ratio < 0.008 or h_ratio > 0.15:
        return False
    # Reject narrow/squarish text (UI buttons, labels) vs wide subtitle lines
    w_ratio = bw / frame_w if frame_w > 0 else 0
    if w_ratio < 0.04:
        return False
    aspect = bw / bh if bh > 0 else 0
    if aspect < 2.0:
        return False
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    if cx < frame_w * 0.03 or cx > frame_w * 0.97:
        return False
    # Reject Windows notification area (bottom-right ~20% × bottom ~15%)
    if cx > frame_w * 0.80 and cy > frame_h * 0.85:
        return False
    return True


class _PaddleOCRClient:
    """Manages PaddleOCR subprocess (GPU) - avoids import conflict with PyTorch."""

    def __init__(self, device: str = "gpu", lang: str = "en"):
        worker_path = os.path.join(os.path.dirname(__file__), "paddleocr_worker.py")
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--device", device, "--lang", lang],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def run(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w, c = frame.shape
        msg = json.dumps({
            "h": h, "w": w, "c": c,
            "data": base64.b64encode(frame.tobytes()).decode("ascii"),
        }).encode("utf-8")
        proc = self._proc
        proc.stdin.write(struct.pack("<I", len(msg)))
        proc.stdin.write(msg)
        proc.stdin.flush()

        line = proc.stderr.readline()
        if not line:
            return []
        result = json.loads(line.decode("utf-8"))
        return result.get("detections", [])

    def close(self):
        if self._proc:
            try:
                msg = json.dumps({"type": "shutdown"}).encode("utf-8")
                self._proc.stdin.write(struct.pack("<I", len(msg)))
                self._proc.stdin.write(msg)
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


class OCREngine:
    """OCR engine. Uses PaddleOCR (GPU subprocess) or EasyOCR (CPU/GPU in-process)."""

    _easyocr = None
    _use_gpu = False
    _paddle: _PaddleOCRClient | None = None
    _paddle_mode = False
    _paddle_lang: str = "en"
    _paddle_device: str = "gpu"

    @classmethod
    def use_paddle(cls, enabled: bool = True):
        cls._paddle_mode = enabled

    @classmethod
    def get_instance(cls, device: str = "gpu", lang: str = "en"):
        if cls._paddle_mode:
            if cls._paddle is None:
                cls._paddle = _PaddleOCRClient(device=device, lang=lang)
                cls._paddle_lang = lang
                cls._paddle_device = device
            elif cls._paddle_lang != lang or cls._paddle_device != device:
                # Language changed — restart worker
                print(f"[OCREngine] Restarting PaddleOCR worker: lang={lang}, device={device}")
                cls._paddle.close()
                cls._paddle = _PaddleOCRClient(device=device, lang=lang)
                cls._paddle_lang = lang
                cls._paddle_device = device
        else:
            if cls._easyocr is None:
                import torch
                cls._use_gpu = torch.cuda.is_available()
                print(f"[OCREngine] EasyOCR CUDA: {cls._use_gpu}")
                import easyocr
                cls._easyocr = easyocr.Reader(["en"], gpu=cls._use_gpu)
        return cls

    @classmethod
    def get_scale(cls) -> float:
        if cls._paddle_mode:
            return 0.7
        cls.get_instance()
        return 0.7 if cls._use_gpu else 0.5

    @classmethod
    def run(
        cls,
        frame: np.ndarray,
        resize_scale: float | None = None,
        conf_thresh: float = 0.75,
        crop_regions: list[tuple[float, float]] | None = None,
        h_crop: tuple[float, float] = (0.0, 1.0),
        yolo_bboxes: list[list] | None = None,
        device: str = "gpu",
        source_lang: str = "en",
    ) -> list[list[dict[str, Any]]]:
        """Return detections grouped by YOLO bbox crop. Each group = one subtitle region."""
        import cv2

        if resize_scale is None:
            resize_scale = cls.get_scale()

        cls.get_instance(device=device, lang=source_lang)
        h, w = frame.shape[:2]
        x_l = int(w * h_crop[0])
        x_r = int(w * h_crop[1])
        if x_l >= x_r:
            x_l, x_r = 0, w
        groups = []

        if yolo_bboxes:
            crops = []
            for b in yolo_bboxes:
                xs = [p[0] for p in b]
                ys = [p[1] for p in b]
                y1 = max(0, int(min(ys)))
                y2 = min(h, int(max(ys)))
                if y2 - y1 < 10:
                    continue
                bx1 = max(x_l, int(min(xs)))
                bx2 = min(x_r, int(max(xs)))
                if bx2 - bx1 < 10:
                    continue
                crops.append((y1, y2, bx1, bx2))
        else:
            if crop_regions is None:
                crop_regions = [(0, 0.27), (0.73, 1.0)]
            crops = [(int(h * t), int(h * b), x_l, x_r) for t, b in crop_regions]

        for y1, y2, cx1, cx2 in crops:
            if y1 >= y2 or cx1 >= cx2:
                continue
            region = frame[y1:y2, cx1:cx2]
            if resize_scale != 1.0:
                region = cv2.resize(region, (0, 0), fx=resize_scale, fy=resize_scale)

            if cls._paddle_mode:
                raw_dets = cls._paddle.run(region)
            else:
                raw_dets = cls._run_easyocr_region(region)

            group = []
            for det in raw_dets:
                if det.get("confidence", 0) < conf_thresh:
                    continue
                bbox = det["bbox"]
                adjusted_bbox = [
                    [x / resize_scale + cx1, y / resize_scale + y1]
                    for x, y in bbox
                ]
                if not is_subtitle_candidate(adjusted_bbox, det["text"], det["confidence"], h, w):
                    continue
                group.append({
                    "bbox": adjusted_bbox,
                    "text": det["text"],
                    "confidence": det["confidence"],
                })
            if group:
                print(f"[OCR] crop({y1},{y2},{cx1},{cx2}) dets: {[d['text'] for d in group]}")
                groups.append(group)

        return groups

    @classmethod
    def _run_easyocr_region(cls, region: np.ndarray) -> list[dict[str, Any]]:
        result = cls._easyocr.readtext(region)
        detections = []
        for bbox, text, confidence in result:
            detections.append({"bbox": bbox, "text": text, "confidence": confidence})
        return detections

    @classmethod
    def shutdown(cls):
        if cls._paddle:
            cls._paddle.close()
            cls._paddle = None
