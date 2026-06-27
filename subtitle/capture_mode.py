"""Capture target helpers."""

from __future__ import annotations


CAPTURE_UNSET = "unset"
CAPTURE_WINDOW = "window"
CAPTURE_FULLSCREEN = "fullscreen"


def normalize_capture_mode(mode: str | None) -> str:
    if mode in {CAPTURE_WINDOW, CAPTURE_FULLSCREEN}:
        return mode
    return CAPTURE_UNSET


def can_enable_capture(mode: str | None, hwnd: int | None) -> bool:
    normalized = normalize_capture_mode(mode)
    if normalized == CAPTURE_FULLSCREEN:
        return True
    if normalized == CAPTURE_WINDOW:
        return hwnd is not None
    return False


def mode_for_selected_hwnd(hwnd: int | None) -> str:
    return CAPTURE_FULLSCREEN if hwnd is None else CAPTURE_WINDOW


def selected_hwnd_for_mode(mode: str | None, hwnd: int | None) -> int | None:
    return hwnd if normalize_capture_mode(mode) == CAPTURE_WINDOW else None
