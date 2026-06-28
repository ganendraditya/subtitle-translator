"""Motion-based static overlay rejection.

Subtitles are embedded in the video frame — their pixel region changes
frame-to-frame because the background video content changes behind the text.

Notifications/UI overlays are rendered ON TOP of the video with a solid
background — their pixel region has near-zero motion between frames.

Motion filter exploits this difference to reject static overlays.
"""

import numpy as np
import cv2


def _motion_ratio(mask: np.ndarray, bbox: list) -> float:
    """Fraction of 'changed' pixels inside the bbox region."""
    xs = [int(pt[0]) for pt in bbox]
    ys = [int(pt[1]) for pt in bbox]
    x1, x2 = max(0, min(xs)), min(mask.shape[1], max(xs))
    y1, y2 = max(0, min(ys)), min(mask.shape[0], max(ys))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    total = region.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(region)) / total


def _is_subtitle_band(bbox: list, frame_h: int) -> bool:
    ys = [float(pt[1]) for pt in bbox]
    cy = (min(ys) + max(ys)) / 2
    return cy < frame_h * 0.28 or cy > frame_h * 0.62


def filter_motion_detections(
    detections: list[dict],
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    diff_thresh: int = 15,
    motion_threshold: float = 0.03,
    global_motion_floor: float = 0.01,
) -> tuple[list[dict], np.ndarray]:
    """Remove detections in static regions (UI overlays, notifications).

    Returns (filtered_detections, curr_gray_for_next_frame).
    If prev_gray is None (first frame), returns all detections unchanged.

    Falls back to unfiltered list if removal would zero out all detections
    (handles paused-video edge case).
    """
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or not detections:
        return detections, curr_gray

    diff = cv2.absdiff(curr_gray, prev_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # If whole frame is nearly static (paused video), skip filter
    global_motion = float(np.count_nonzero(mask)) / mask.size
    if global_motion < global_motion_floor:
        return detections, curr_gray

    filtered = [
        d
        for d in detections
        if _is_subtitle_band(d["bbox"], frame.shape[0])
        or _motion_ratio(mask, d["bbox"]) >= motion_threshold
    ]

    return filtered if filtered else detections, curr_gray
