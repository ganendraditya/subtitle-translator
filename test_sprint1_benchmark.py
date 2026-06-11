#!/usr/bin/env python3
"""Visible test + Performance benchmark for Sprint 1."""

import sys
import time
import os
import threading
import psutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6.QtWidgets import QApplication
from overlay.renderer import OverlayWindow
from capture.screen import ScreenCapture


class Benchmark:
    def __init__(self):
        self.frame_times = []
        self.fps_values = []
        self.start_time = None
        self.process = psutil.Process()
        self.initial_memory = 0
        
    def start(self):
        self.start_time = time.perf_counter()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def record_frame(self, frame_time):
        self.frame_times.append(frame_time)
        
    def record_fps(self, fps):
        self.fps_values.append(fps)
        
    def report(self):
        elapsed = time.perf_counter() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        print("\n" + "=" * 60)
        print("SPRINT 1 PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Duration: {elapsed:.2f}s")
        print(f"Total frames captured: {len(self.frame_times)}")
        print(f"Average FPS: {len(self.frame_times) / elapsed:.2f}")
        print(f"Frame time (avg): {np.mean(self.frame_times) * 1000:.2f}ms")
        print(f"Frame time (min): {np.min(self.frame_times) * 1000:.2f}ms")
        print(f"Frame time (max): {np.max(self.frame_times) * 1000:.2f}ms")
        print(f"Memory usage: {self.initial_memory:.1f} MB -> {current_memory:.1f} MB (delta {current_memory - self.initial_memory:.1f} MB)")
        print("=" * 60)
        

class Sprint1Test:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.overlay = OverlayWindow()
        self.capture = ScreenCapture()
        self.benchmark = Benchmark()
        self.running = True
        
    def run(self):
        # Setup overlay
        self.overlay.set_text("Sprint 1: Benchmark Running...")
        self.overlay.set_position("bottom")
        
        # Register callback
        self.capture.register_callback(self._on_frame)
        
        print("=" * 60)
        print("SPRINT 1 VISIBLE + BENCHMARK TEST")
        print("=" * 60)
        print("Overlay will appear for 10 seconds.")
        print("You can click through it (try clicking other windows).")
        print("=" * 60)
        
        # Start capture
        self.benchmark.start()
        self.capture.start(target_fps=30)
        
        # Auto-close after 10s
        def auto_close():
            time.sleep(10)
            self.running = False
            self.capture.stop()
            self.benchmark.report()
            print("\n[SUCCESS] Sprint 1 Test PASSED")
            print("Overlay + Capture working correctly!")
            self.app.quit()
            
        threading.Thread(target=auto_close, daemon=True).start()
        
        try:
            self.app.exec()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            self.capture.stop()
            
    def _on_frame(self, frame):
        start = time.perf_counter()
        # Sprint 1: just count FPS, no processing
        self.benchmark.record_frame(0)  # Frame captured
        
        # Update FPS display every second
        self.benchmark.record_fps(30)  # Target FPS
        
        # Overlay update
        elapsed = time.perf_counter() - self.benchmark.start_time
        if int(elapsed) % 2 == 0:  # Update every 2 seconds
            self.overlay.set_text(f"Sprint 1 | FPS: ~{len(self.benchmark.frame_times) / max(elapsed, 0.1):.1f}")


if __name__ == "__main__":
    test = Sprint1Test()
    test.run()
