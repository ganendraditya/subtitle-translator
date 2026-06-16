# Subtitle Translator

Real-time screen subtitle translator
utilizing object detection for subtitle localization, OCR to identify text, and translation with GPU acceleration.

> **Status:** Active development — pre-alpha.

## Pipeline

```
Screen capture (dxcam)
    → YOLOv26 subtitle region detection (optional, fallback to fixed crop)
    → PaddleOCR GPU (multilingual, subprocess worker)
    → Language filter (Unicode script detection)
    → MarianMT 2-hop translation (src → en → target)
    → PyQt6 transparent overlay (dual: top + bottom)
```

## Architecture

- **Screen capture**: dxcam
- **Subtitle detection**: YOLOv26 (user-trained, single class "subtitle")
- **OCR**: PaddleOCR 2.10 GPU
- **Translation**: MarianMT (Helsinki-NLP opus-mt), 2-hop via English for unsupported direct pairs
- **Overlay**: PyQt6 transparent windows

## Languages

| Language | OCR | Translation |
|----------|:---:|:-----------:|
| English | ✅ | ✅ |
| Indonesian | ✅ | ✅ 2-hop |
| Japanese | ✅ | ✅ 2-hop |
| Chinese (Simplified + Traditional) | ✅ | ✅ 2-hop |
| Korean | ✅ | ✅ 2-hop |
| French | ✅ | ✅ 2-hop |
| German | ✅ | ✅ 2-hop |
| Spanish | ✅ | ✅ 2-hop |
| Arabic | ✅ | ✅ 2-hop |

## Sprint History

| Sprint | What |
|:------:|------|
| 1 | Screen capture + PyQt6 overlay |
| 2 | PaddleOCR/EasyOCR integration |
| 3 | Subtitle filtering, history, fuzzy matching |
| 4 | Translation engine (MarianMT + NLLB), GPU/CPU fallback |
| 5 | System tray, settings dialog, startup shortcut |
| 6 | YOLOv26 detection, MarianMT 2-hop, PaddleOCR multilingual, language filter |

**Currently on:** Sprint 6 — training YOLO for subtitle region detection.
