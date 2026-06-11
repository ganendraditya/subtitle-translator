import time
import threading
from typing import Optional, Callable
import numpy as np
import dxcam


class ScreenCapture:
    """Wrapper around dxcam for screen capture."""

    def __init__(self) -> None:
        self._camera: Optional[dxcam.DXCamera] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._frame_interval = 1  # Process every Nth frame
        self._frame_counter = 0

    def start(self, target_fps: int = 30) -> None:
        """Start capturing screen at target_fps."""
        if self._running:
            return

        self._camera = dxcam.create()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(target_fps,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._camera:
            self._camera.release()
            self._camera = None

    def register_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback to receive frames."""
        self._callbacks.append(callback)

    def set_frame_interval(self, interval: int) -> None:
        """Process every Nth frame (interval=3 means 33% of frames processed)."""
        self._frame_interval = max(1, interval)

    def _capture_loop(self, target_fps: int) -> None:
        """Internal capture loop running in a thread."""
        interval = 1.0 / target_fps
        assert self._camera is not None
        while self._running:
            start = time.perf_counter()
            frame = self._camera.grab()
            if frame is not None:
                self._frame_counter += 1
                if self._frame_counter % self._frame_interval == 0:
                    self._notify(frame)
            elapsed = time.perf_counter() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _notify(self, frame: np.ndarray) -> None:
        """Notify all registered callbacks with a new frame."""
        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception as e:
                print(f"[ScreenCapture] Callback error: {e}")
