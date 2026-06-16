# Subtitle Translator

Real-time screen subtitle detection, OCR, and translation with GPU acceleration.

> **Status:** Active development — pre-alpha.

## Pipeline

```
Screen capture (dxcam)
    → YOLOv8 subtitle region detection (optional, fallback to fixed crop)
    → PaddleOCR GPU (multilingual, subprocess worker)
    → Language filter (Unicode script detection)
    → MarianMT 2-hop translation (src → en → target)
    → PyQt6 transparent overlay (dual: top + bottom)
```

## Architecture

- **Screen capture**: dxcam (Windows Desktop Duplication API, ~30 FPS)
- **Subtitle detection**: YOLOv8 (user-trained, single class "subtitle") or fixed crop regions
- **OCR**: PaddleOCR 2.10 GPU (subprocess worker to avoid pybind11 conflict with PyTorch)
- **Language filter**: Unicode range detection (CJK, Hangul, Hiragana/Katakana, Arabic, Latin)
- **Translation**: MarianMT (Helsinki-NLP opus-mt), 2-hop via English for unsupported direct pairs
- **Overlay**: PyQt6 transparent windows, positioned above/below OCR crop regions

## Requirements

- Windows 10/11
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 3060)
- CUDA 12.5
- Python 3.12

## Quick Start

```bash
# Clone
git clone https://github.com/ganendraditya/subtitle-translator.git
cd subtitle-translator

# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Additional GPU dependencies
pip install paddlepaddle-gpu==2.6.2 paddleocr==2.10.0
pip install nvidia-cudnn-cu12==8.9.7.29 nvidia-cublas-cu11==11.11.3.6

# Run
python run.py
# Or silent (no console):
pythonw run.pyw
```

## YOLO Training (Optional)

The app falls back to fixed crop regions if no YOLO model is found at `models/yolov8s-subtitle.pt`.

To train:

1. Capture ~200-500 screen frames with subtitles
2. Label ground-truth bounding boxes — 1 class `subtitle`
3. Train:

```bash
yolo train model=yolov8s.pt data=dataset.yaml epochs=100 imgsz=640
```

Expected: P ≥ 0.85, R ≥ 0.80, mAP50 ≥ 0.85.

## Languages

| Language | OCR | Translation |
|----------|:---:|:-----------:|
| English | ✅ | ✅ |
| Indonesian | ✅ | ✅ |
| Japanese | ✅ | ✅ 2-hop |
| Chinese (Simplified + Traditional) | ✅ | ✅ 2-hop |
| Korean | ✅ | ✅ 2-hop |
| French | ✅ | ✅ 2-hop |
| German | ✅ | ✅ 2-hop |
| Spanish | ✅ | ✅ 2-hop |
| Arabic | ✅ | ✅ 2-hop |

## Performance

| Stage | Latency |
|:------|:-------:|
| Screen capture | ~1ms |
| YOLOv8s (640px) | ~15ms |
| PaddleOCR GPU (2 regions) | ~70ms |
| MarianMT 1-hop | ~50ms |
| MarianMT 2-hop | ~100-200ms |
| **Total 1-hop** | **~136ms** |
| **Total 2-hop** | **~186-286ms** |

OCR runs at ~2 Hz (every 500ms). Translation is per frame.

## Sprint History

| Sprint | What |
|:------:|------|
| 1 | Screen capture + PyQt6 overlay |
| 2 | PaddleOCR/EasyOCR integration |
| 3 | Subtitle filtering, history, fuzzy matching |
| 4 | Translation engine (MarianMT + NLLB), GPU/CPU fallback |
| 5 | System tray, settings dialog, startup shortcut |
| 6 | YOLOv8 detection, MarianMT 2-hop, PaddleOCR multilingual, language filter |

**Current:** Sprint 6 — training YOLO for subtitle region detection.

## Known Issues

- PaddlePaddle GPU and PyTorch cannot coexist in the same process (pybind11 `_gpuDeviceProperties` conflict) — PaddleOCR runs in a subprocess
- First launch ~10s for model loading (CUDA compilation)
- Windows toast notifications in bottom-right corner may be detected as subtitles (motion filter tries to reject)

## License

MIT
