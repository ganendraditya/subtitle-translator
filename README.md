# Subtitle Translator

Real-time screen captured subtitle translator. Utilizing object detection model for subtitle localization, OCR to identify the text, and neural machine translation.

> **Status:** Active development — pre-alpha.

## Sprints

1. Screen capture + PyQt6 overlay
2. PaddleOCR/EasyOCR integration
3. Subtitle filtering, history, fuzzy matching
4. Translation engine (MarianMT + NLLB), GPU/CPU fallback
5. System tray, settings dialog, startup shortcut

**Currently:** gathering and annotating data to train YOLO for subtitle region detection.

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
