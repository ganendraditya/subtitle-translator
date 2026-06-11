#!/usr/bin/env python3
"""Pure performance benchmark - no GUI, writes result to file."""

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from capture.screen import ScreenCapture


def benchmark():
    capture = ScreenCapture()
    
    frame_count = 0
    start_time = time.perf_counter()
    
    def on_frame(frame):
        nonlocal frame_count
        frame_count += 1
    
    capture.register_callback(on_frame)
    capture.start(target_fps=30)
    
    # Run for 10 seconds
    time.sleep(10)
    
    capture.stop()
    
    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed
    
    report = f"""
{'=' * 60}
SPRINT 1 PERFORMANCE BENCHMARK (No GUI)
{'=' * 60}
Duration: {elapsed:.2f}s
Total frames captured: {frame_count}
Average FPS: {avg_fps:.2f}
Target FPS: 30
Frame time avg: {(elapsed / frame_count) * 1000:.2f}ms if > 0 else N/A
{'=' * 60}
RESULT: {'PASS' if avg_fps > 25 else 'MARGINAL' if avg_fps > 15 else 'FAIL'}
{'=' * 60}
"""
    
    print(report)
    
    # Write to file
    with open("sprint1_benchmark_result.txt", "w") as f:
        f.write(report)
    
    print("[INFO] Report saved to sprint1_benchmark_result.txt")


if __name__ == "__main__":
    benchmark()
