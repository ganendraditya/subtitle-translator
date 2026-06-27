# Subtitle Translator

Real-time screen subtitle translator for Windows. It captures the screen or a selected window, detects subtitle regions, runs OCR, translates the text, and displays a click-through PyQt overlay.

> Status: active development, pre-alpha.

## What Is Included

- Application source code
- Lightweight unit tests

## What Is Not Included

- Training datasets
- YOLO model binaries (`.onnx`, `.engine`, `.pt`)
- Local `config.json`
- Internal notes, agent files, and training assets

## Stack

- Screen/window capture: dxcam and Windows `PrintWindow`
- Subtitle detection: YOLO via Ultralytics, using `.onnx` and optional TensorRT `.engine`
- OCR: PaddleOCR in a subprocess, with EasyOCR fallback code still present
- Translation: MarianMT / Helsinki-NLP, with 2-hop routing through English
- Overlay/UI: PyQt6 tray app and transparent click-through overlay windows

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place your subtitle detector model at:

```text
yolo/epoch51.onnx
```

TensorRT is optional. If you have a compatible TensorRT runtime and engine, place it at:

```text
yolo/epoch51.engine
```

The default backend is ONNX.

## Run

```powershell
.\venv\Scripts\python.exe run.py
.\venv\Scripts\python.exe run.pyw
```

Use `run.py` for debugging with terminal logs. Use `run.pyw` for no-console startup.

## Test

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

## Languages

Supported UI language codes:

| Code | Language |
| --- | --- |
| `en` | English |
| `id` | Indonesian |
| `ja` | Japanese |
| `zh` | Chinese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ar` | Arabic |

Translation caveat: Japanese and Korean use non-standard Helsinki model IDs internally, so the app maps UI codes `ja` and `ko` before loading MarianMT models.
