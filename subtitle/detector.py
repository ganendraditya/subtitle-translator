from __future__ import annotations

import os
import importlib.util
import cv2
import numpy as np
from pathlib import Path
from typing import Any


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


def _inject_tensorrt_runtime_paths(base_dir: Path | None = None) -> list[str]:
    """Add TensorRT runtime folders to PATH when a local install is configured."""
    root = base_dir or _project_root()
    roots = []
    for env_name in ("TENSORRT_HOME", "TRT_HOME"):
        env_value = os.environ.get(env_name)
        if env_value:
            roots.append(Path(env_value).expanduser())
    roots.extend([root / "tensorrt", root / "TensorRT"])

    added: list[str] = []
    for trt_root in roots:
        for candidate in (trt_root / "lib", trt_root / "bin", trt_root):
            if not candidate.exists():
                continue
            path = str(candidate)
            if path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(path)
            except (AttributeError, FileNotFoundError, OSError):
                pass
            added.append(path)
    return added


def _candidate_model_paths(model_base: str, device: str, backend: str = "onnx") -> list[str]:
    base = model_base
    if base.lower().endswith((".engine", ".onnx")):
        base = os.path.splitext(base)[0]

    engine_path = base + ".engine"
    onnx_path = base + ".onnx"
    backend = (backend or "onnx").lower()
    wants_tensorrt = backend in ("tensorrt", "trt", "engine")
    if device == "cpu" or not wants_tensorrt:
        return [onnx_path]
    return [engine_path, onnx_path]


class SubtitleDetector:
    def __init__(self, model_base: str | None = None, device: str = "gpu", backend: str = "onnx"):
        os.environ.setdefault("YOLO_AUTOINSTALL", "0")
        _ensure_ultralytics_config_dir()
        _inject_tensorrt_runtime_paths()
        # Add PyTorch's bundled CUDA/cuDNN DLLs to PATH so onnxruntime can find cudnn64_9.dll
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        self.model_base = model_base
        self.device = device
        self.backend = (backend or "onnx").lower()
        self._model = None
        self._loaded_path = None
        self._loaded_ext = None

    def _tensorrt_available(self) -> tuple[bool, str]:
        try:
            import tensorrt
            return True, getattr(tensorrt, "__version__", "unknown")
        except ImportError as exc:
            if importlib.util.find_spec("tensorrt_bindings") is not None:
                try:
                    import tensorrt_bindings  # noqa: F401
                except Exception as binding_exc:
                    return (
                        False,
                        "TensorRT Python bindings are installed, but native TensorRT DLLs are missing or unusable: "
                        f"{binding_exc}",
                    )
                return (
                    False,
                    "TensorRT bindings are installed, but the top-level `tensorrt` package is missing.",
                )
            return (
                False,
                "Python package `tensorrt` is not importable. On Windows, install NVIDIA TensorRT runtime "
                "and ensure its `lib` directory with `nvinfer_*.dll` is on PATH.",
            )

    def _warmup_engine(self) -> None:
        if self._model is None:
            return
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(image, conf=0.5, verbose=False, device=0)

    def _try_load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext == ".engine":
            ok, detail = self._tensorrt_available()
            if not ok:
                print(f"[Detector] Skipping TensorRT engine: {detail}")
                return False
        try:
            _ensure_ultralytics_config_dir()
            from ultralytics import YOLO
            self._model = YOLO(path, task="detect")
            if ext == ".engine":
                self._warmup_engine()
            self._loaded_path = path
            self._loaded_ext = ext
            backend = "TensorRT" if ext == ".engine" else "ONNX Runtime"
            print(f"[Detector] Loaded {backend}: {path}")
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

        paths = _candidate_model_paths(self.model_base, self.device, self.backend)
        engine_path = _candidate_model_paths(self.model_base, "gpu", "tensorrt")[0]
        if self.device == "cpu" and self.backend in ("tensorrt", "trt", "engine") and os.path.exists(engine_path):
            print("[Detector] CPU mode selected; skipping TensorRT .engine")

        for path in paths:
            if self._try_load(path):
                return True

        base = os.path.splitext(self.model_base)[0]
        print(f"[Detector] No usable model found at {base}.(engine|onnx)")
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
                x1 = int(x1 / scale); y1 = int(y1 / scale)
                x2 = int(x2 / scale); y2 = int(y2 / scale)
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
