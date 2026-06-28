from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_ultralytics_config_dir(base_dir: Path | None = None) -> Path:
    """Keep Ultralytics settings inside the project instead of user AppData."""
    env_value = os.environ.get("YOLO_CONFIG_DIR")
    if env_value:
        path = Path(env_value).expanduser()
    else:
        path = (base_dir or _project_root()) / ".ultralytics"
        os.environ["YOLO_CONFIG_DIR"] = str(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_model_paths(model_base: str) -> list[str]:
    base, ext = os.path.splitext(model_base)
    if ext.lower() == ".onnx":
        return [model_base]
    return [model_base + ".onnx"]


class SubtitleDetector:
    def __init__(self, model_base: str | None = None, device: str = "gpu"):
        os.environ.setdefault("YOLO_AUTOINSTALL", "0")
        _ensure_ultralytics_config_dir()
        # ONNX Runtime GPU needs CUDA/cuDNN DLLs that are bundled with PyTorch wheels.
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        self.model_base = model_base
        self.device = device
        self._model = None
        self._loaded_path = None

    def _try_load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            _ensure_ultralytics_config_dir()
            from ultralytics import YOLO
            self._model = YOLO(path, task="detect")
            self._loaded_path = path
            print(f"[Detector] Loaded ONNX Runtime: {path}")
            return True
        except Exception as e:
            print(f"[Detector] Failed to load {path}: {e}")
            self._model = None
            return False

    def load(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_base:
            return False

        for path in _candidate_model_paths(self.model_base):
            if self._try_load(path):
                return True

        base = os.path.splitext(self.model_base)[0]
        print(f"[Detector] No usable model found at {base}.onnx")
        return False

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.35) -> list[dict[str, Any]]:
        if self._model is None:
            return []

        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        infer_device = "cpu" if self.device == "cpu" else 0
        try:
            results = self._model(small, conf=conf_thresh, verbose=False, device=infer_device)
        except Exception as e:
            print(f"[Detector] Inference failed, disabling model: {e}")
            self._model = None
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if scale < 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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
