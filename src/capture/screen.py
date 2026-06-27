"""Screen capture with optional strict window support."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time
import numpy as np
import dxcam


_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32
_dwmapi = ctypes.windll.dwmapi

DWMWA_EXTENDED_FRAME_BOUNDS = 9
PW_RENDERFULLCONTENT = 0x00000002


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class ScreenCapture:
    def __init__(self):
        self._camera = None
        self._window_hwnd: int | None = None
        self._last_bounds: tuple[int, int, int, int] | None = None
        self._last_window_fail_log = 0.0

    def start(self):
        if self._camera is None:
            try:
                self._camera = dxcam.create(output_color="BGR")
            except TypeError:
                self._camera = dxcam.create()

    def set_window(self, hwnd: int | None) -> None:
        """Set target window for capture. None = full screen."""
        self._window_hwnd = hwnd
        if hwnd:
            print(f"[Capture] Window set: hwnd={hwnd}")
        else:
            print("[Capture] Full screen mode")

    @property
    def capture_bounds(self) -> tuple[int, int, int, int] | None:
        """Last captured screen bounds as (left, top, right, bottom)."""
        return self._last_bounds

    def grab(self):
        if self._camera is None:
            return None

        if self._window_hwnd:
            frame = self._grab_window_printwindow(self._window_hwnd)
            if frame is not None:
                return frame

            frame = self._grab_foreground_window_region(self._window_hwnd)
            if frame is not None:
                return frame

            self._log_window_failure(self._window_hwnd)
            return None

        frame = self._camera.grab()
        if frame is not None:
            h, w = frame.shape[:2]
            self._last_bounds = (0, 0, w, h)
        return frame

    def _get_window_bounds(self, hwnd: int) -> tuple[int, int, int, int] | None:
        rect = RECT()
        hr = _dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if hr == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return (rect.left, rect.top, rect.right, rect.bottom)
        if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            if rect.right > rect.left and rect.bottom > rect.top:
                return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def _grab_window_printwindow(self, hwnd: int) -> np.ndarray | None:
        """Capture window content using PrintWindow (captures regardless of z-order)."""
        try:
            if not _user32.IsWindow(hwnd) or _user32.IsIconic(hwnd):
                return None

            bounds = self._get_window_bounds(hwnd)
            if bounds is None:
                return None
            left, top, right, bottom = bounds
            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                return None

            hdc_screen = _user32.GetDC(0)
            hdc_mem = _gdi32.CreateCompatibleDC(hdc_screen)
            hbitmap = _gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            old_obj = _gdi32.SelectObject(hdc_mem, hbitmap)

            result = _user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
            if result == 0:
                result = _user32.PrintWindow(hwnd, hdc_mem, 0)
                if result == 0:
                    _gdi32.SelectObject(hdc_mem, old_obj)
                    _gdi32.DeleteObject(hbitmap)
                    _gdi32.DeleteDC(hdc_mem)
                    _user32.ReleaseDC(0, hdc_screen)
                    return None

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h  # negative = top-down
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0  # BI_RGB

            buf = ctypes.create_string_buffer(w * h * 4)
            _gdi32.GetDIBits(hdc_mem, hbitmap, 0, h, buf, ctypes.byref(bmi), 0)

            _gdi32.SelectObject(hdc_mem, old_obj)
            _gdi32.DeleteObject(hbitmap)
            _gdi32.DeleteDC(hdc_mem)
            _user32.ReleaseDC(0, hdc_screen)

            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            frame = frame[:, :, :3].copy()
            if self._is_blank_frame(frame):
                return None
            self._last_bounds = bounds
            return frame

        except Exception as e:
            print(f"[Capture] PrintWindow error: {e}")
            return None

    def _grab_foreground_window_region(self, hwnd: int) -> np.ndarray | None:
        """Fallback to desktop-duplication crop only while target window is foreground."""
        if _user32.GetForegroundWindow() != hwnd:
            return None
        bounds = self._get_window_bounds(hwnd)
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        if right <= left or bottom <= top:
            return None
        try:
            frame = self._camera.grab(region=(left, top, right, bottom))
        except Exception as e:
            print(f"[Capture] Window region grab error: {e}")
            return None
        if frame is None:
            return None
        self._last_bounds = bounds
        return frame

    def _is_blank_frame(self, frame: np.ndarray) -> bool:
        return frame.size == 0 or (float(frame.mean()) < 2.0 and float(frame.std()) < 2.0)

    def _log_window_failure(self, hwnd: int) -> None:
        now = time.perf_counter()
        if now - self._last_window_fail_log < 2.0:
            return
        self._last_window_fail_log = now
        print(
            f"[Capture] Window hwnd={hwnd} unavailable; strict window mode skips frame "
            "instead of falling back to full screen"
        )

    def stop(self):
        if self._camera:
            self._camera.release()
            self._camera = None
        self._last_bounds = None
