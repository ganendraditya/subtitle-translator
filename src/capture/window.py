"""Enumerate visible windows using ctypes (no pywin32 needed)."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


_user32 = ctypes.windll.user32
_dwmapi = ctypes.windll.dwmapi

# DWMWA_EXTENDED_FRAME_BOUNDS = 9 — gives visible frame without shadow
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def list_windows() -> list[dict]:
    """Return list of visible windows with title, position and size."""
    windows = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if not title or title in ("Program Manager", "Windows Input Experience"):
            return True
        rect = get_window_rect(hwnd)
        if rect is None:
            return True
        left, top, right, bottom = rect
        w = right - left
        h = bottom - top
        if w < 100 or h < 100:
            return True
        windows.append({
            "hwnd": hwnd,
            "title": title,
            "left": left,
            "top": top,
            "width": w,
            "height": h,
        })
        return True

    _user32.EnumWindows(_callback, 0)
    return windows


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Get visible window bounds (without shadow): (left, top, right, bottom)."""
    rect = RECT()
    # Try DWM extended frame bounds first (excludes shadow)
    hr = _dwmapi.DwmGetWindowAttribute(
        hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    )
    if hr == 0 and rect.right > rect.left and rect.bottom > rect.top:
        return (rect.left, rect.top, rect.right, rect.bottom)
    # Fallback to GetWindowRect
    if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        if rect.right > rect.left and rect.bottom > rect.top:
            return (rect.left, rect.top, rect.right, rect.bottom)
    return None
