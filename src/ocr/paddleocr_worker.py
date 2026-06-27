"""PaddleOCR worker process. Runs in a subprocess to avoid PaddlePaddle vs PyTorch conflict."""
from __future__ import annotations

import os
import sys
import json
import base64
import struct
import time

import numpy as np

# Signal startup
start = time.time()


def log(msg: str):
    sys.stdout.write(f"[worker {time.time()-start:.1f}s] {msg}\n")
    sys.stdout.flush()


log("Setting up PATH")
nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for p in [
    os.path.join(nvidia_base, "cudnn", "bin"),
    os.path.join(nvidia_base, "cublas", "bin"),
]:
    if os.path.exists(p):
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")

# Parse args
_device = "gpu"
_lang = "en"
for i, arg in enumerate(sys.argv):
    if arg == "--device" and i + 1 < len(sys.argv):
        _device = sys.argv[i + 1].lower()
    if arg == "--lang" and i + 1 < len(sys.argv):
        _lang = sys.argv[i + 1].lower()

use_gpu = _device != "cpu"
log(f"Device: {_device} (use_gpu={use_gpu}), Lang: {_lang}")

# Map source language codes to PaddleOCR lang codes
_LANG_MAP = {
    "en": "en", "id": "en", "fr": "fr", "de": "de", "es": "es",
    "ja": "japan", "zh": "ch", "ko": "korean", "ar": "arabic",
}
ocr_lang = _LANG_MAP.get(_lang, "en")
log(f"PaddleOCR lang: {ocr_lang}")

log("Importing PaddleOCR")
from paddleocr import PaddleOCR

_ocr: PaddleOCR | None = None


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        log("Creating PaddleOCR instance")
        _ocr = PaddleOCR(use_angle_cls=False, lang=ocr_lang, use_gpu=use_gpu, show_log=False)
        log("PaddleOCR ready")
    return _ocr


def _decode_image(msg: dict) -> np.ndarray:
    data = base64.b64decode(msg["data"])
    h, w, c = msg["h"], msg["w"], msg["c"]
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))


def _format_detections(raw: list) -> list[dict]:
    if not raw or raw[0] is None:
        return []
    detections = []
    for line in raw[0]:
        bbox, (text, conf) = line
        detections.append({
            "bbox": bbox,
            "text": text,
            "confidence": round(conf, 4),
        })
    return detections


def main():
    log("Entering main loop")
    ocr = get_ocr()
    stdin_buf = sys.stdin.buffer

    while True:
        header = stdin_buf.read(4)
        if not header or len(header) < 4:
            log("stdin closed, exiting")
            break
        msg_len = struct.unpack("<I", header)[0]
        if msg_len == 0:
            break
        msg_bytes = stdin_buf.read(msg_len)
        if not msg_bytes or len(msg_bytes) < msg_len:
            break

        msg = json.loads(msg_bytes.decode("utf-8"))
        if msg.get("type") == "shutdown":
            log("Shutdown received")
            break

        img = _decode_image(msg)
        result = ocr.ocr(img, cls=False)
        detections = _format_detections(result)

        line = json.dumps({"detections": detections}, ensure_ascii=False) + "\n"
        sys.stderr.write(line)
        sys.stderr.flush()


if __name__ == "__main__":
    main()
