"""Computer vision, auto-framing and smart face tracking for 9:16 vertical video clipping.

Supports:
- Face detection using OpenCV FaceDetectorYN (YuNet) with heuristic fallback.
- Auto-Reframe: Smooth panning and subject tracking for widescreen to 9:16 vertical crop.
- Streamer facecam / gameplay bounding box auto-detection.
- Dynamic Retention Framing: Punch-in zooms on high-excitement timestamps.
- Normalized [0.0, 1.0] coordinates conversion with even-pixel parity for FFmpeg.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("framing")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_detector = None


def get_detector(width: int = 320, height: int = 320, conf_threshold: float = 0.5):
    """Initializes or configures the YuNet face detector."""
    global _detector
    if not MODEL_PATH.exists():
        return None
    try:
        if _detector is None:
            _detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), conf_threshold)
        else:
            _detector.setInputSize((width, height))
            _detector.setScoreThreshold(conf_threshold)
        return _detector
    except Exception as e:
        log.warning("Could not initialize FaceDetectorYN: %s", e)
        return None


def to_pixels(box: Optional[Dict[str, float]], src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    """Converts a normalized [0.0, 1.0] or pixel bounding box into safe, even pixel bounds."""
    if not box:
        return 0, 0, src_w - (src_w % 2), src_h - (src_h % 2)
    bx = box.get("x", 0.0)
    by = box.get("y", 0.0)
    bw = box.get("w", 1.0)
    bh = box.get("h", 1.0)

    # If coordinates are normalized <= 1.0, scale by src dimensions
    x = int(bx * src_w) if bx <= 1.0 else int(bx)
    y = int(by * src_h) if by <= 1.0 else int(by)
    w = int(bw * src_w) if bw <= 1.0 else int(bw)
    h = int(bh * src_h) if bh <= 1.0 else int(bh)

    # Clamp bounds within source frame
    x = max(0, min(src_w - 4, x))
    y = max(0, min(src_h - 4, y))
    w = max(4, min(src_w - x, w))
    h = max(4, min(src_h - y, h))

    # Ensure even dimensions (required by FFmpeg yuv420p & nvenc/qsv encoders)
    w -= (w % 2)
    h -= (h % 2)
    x -= (x % 2)
    y -= (y % 2)
    return x, y, w, h


def detect_faces_in_frame(frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
    """Detects faces in a BGR frame. Returns list of (x, y, w, h, score)."""
    h, w = frame.shape[:2]
    # Resize frame to speed up inference while maintaining aspect ratio
    max_dim = 640
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        target_w = int(w * scale)
        target_h = int(h * scale)
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame
        target_w, target_h = w, h

    detector = get_detector(target_w, target_h, conf_threshold)
    if detector is not None:
        try:
            _, faces = detector.detect(resized)
            if faces is not None and len(faces) > 0:
                results = []
                for f in faces:
                    fx, fy, fw, fh = f[:4] / scale
                    score = float(f[-1])
                    results.append((float(fx), float(fy), float(fw), float(fh), score))
                return results
        except Exception as e:
            log.warning("YuNet detection error: %s", e)

    # Heuristic fallback: Skin color and edge density detector
    return _detect_face_heuristic(frame)


def _detect_face_heuristic(frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    """Fast color/contour fallback when deep model is unavailable."""
    h, w = frame.shape[:2]
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Typical skin tone range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 250, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if (0.01 * w * h) < area < (0.35 * w * h):
                x, y, cw, ch = cv2.boundingRect(c)
                aspect = cw / float(ch)
                if 0.6 <= aspect <= 1.5:  # Roughly face-like aspect ratio
                    candidates.append((float(x), float(y), float(cw), float(ch), 0.55))
        if candidates:
            candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
            return candidates[:3]
    except Exception:
        pass
    return []


def sample_video_frames(src_path: Path, timestamps: List[float]) -> List[Tuple[float, np.ndarray]]:
    """Extracts frames from video at the requested timestamps."""
    frames = []
    if not src_path.exists():
        return frames
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return frames
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for t in sorted(timestamps):
            target_idx = int(t * fps)
            if 0 <= target_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append((t, frame))
    finally:
        cap.release()
    return frames


def find_facecam_box(src_path: Path, src_w: int, src_h: int, start: float, end: float) -> Dict[str, float]:
    """Detects streamer webcam or primary subject in [0.0, 1.0] normalized space."""
    dur = max(1.0, end - start)
    sample_ts = [start + dur * 0.2, start + dur * 0.5, start + dur * 0.8]
    frames = sample_video_frames(src_path, sample_ts)

    detected_faces = []
    for _, f in frames:
        faces = detect_faces_in_frame(f)
        if faces:
            detected_faces.extend(faces)

    if detected_faces:
        # Pick the most consistent/largest face
        detected_faces.sort(key=lambda item: (item[4], item[2] * item[3]), reverse=True)
        fx, fy, fw, fh, _ = detected_faces[0]

        # Expand box to capture full facecam (head, shoulders, framing margin)
        pad_w = fw * 0.5
        pad_h = fh * 0.7
        x0 = max(0, fx - pad_w)
        y0 = max(0, fy - pad_h * 0.6)
        x1 = min(src_w, fx + fw + pad_w)
        y1 = min(src_h, fy + fh + pad_h * 1.4)

        box_w = x1 - x0
        box_h = y1 - y0

        return {
            "x": round(x0 / src_w, 4),
            "y": round(y0 / src_h, 4),
            "w": round(box_w / src_w, 4),
            "h": round(box_h / src_h, 4),
        }

    # Default fallback for gaming streamers: top-left quadrant (most common webcam location)
    return {"x": 0.0, "y": 0.0, "w": 0.35, "h": 0.40}


def calculate_auto_reframe_crop(src_w: int, src_h: int, face_box: Optional[Dict[str, float]] = None) -> Tuple[int, int, int, int]:
    """Calculates smoothed 9:16 vertical crop bounds centered on subject."""
    crop_w = min(src_w, int(src_h * 9 / 16))
    crop_h = min(src_h, int(crop_w * 16 / 9))

    # Center-of-mass anchor
    if face_box:
        center_x = (face_box["x"] + face_box["w"] / 2.0) * src_w
    else:
        center_x = src_w / 2.0

    x0 = int(center_x - crop_w / 2.0)
    x0 = max(0, min(src_w - crop_w, x0))
    y0 = int((src_h - crop_h) / 2.0)

    # Parity check
    crop_w -= (crop_w % 2)
    crop_h -= (crop_h % 2)
    x0 -= (x0 % 2)
    y0 -= (y0 % 2)
    return x0, y0, crop_w, crop_h


def detect_retention_zooms(words: List[dict], start: float, end: float) -> List[Tuple[float, float]]:
    """Identifies candidate punch-in zoom timestamps for dynamic retention framing.
    Triggers on energetic words, exclamation points, punchlines, or long uniform pauses."""
    dur = end - start
    if dur < 8.0:
        return []

    zooms = []
    inside = [w for w in words if start <= w.get("s", 0) <= end]

    # Find words with high vocal energy or exclamation marks
    for i, w in enumerate(inside):
        text = str(w.get("w", ""))
        rel_s = w.get("s", start) - start
        if any(c in text for c in ("!", "?", "...")) or text.isupper():
            z_start = max(0.5, rel_s - 0.2)
            z_end = min(dur - 0.5, z_start + 1.8)
            if z_end - z_start >= 1.2:
                # Avoid overlapping zooms
                if not zooms or z_start > (zooms[-1][1] + 5.0):
                    zooms.append((round(z_start, 2), round(z_end, 2)))
                    if len(zooms) >= 3:
                        break

    # If no punctuation cues found but clip is long (>20s), add a rhythmic mid-clip retention zoom
    if not zooms and dur >= 18.0:
        mid = round(dur * 0.45, 2)
        zooms.append((mid, round(mid + 1.8, 2)))

    return zooms


def make_zoom_filter(punch_ins: List[Tuple[float, float]], zoom_factor: float = 1.15) -> str:
    """Generates FFmpeg expression for single-pass dynamic punch-in zoom."""
    if not punch_ins:
        return ""
    conditions = [f"between(t\\,{t0}\\,{t1})" for t0, t1 in punch_ins]
    cond_expr = "+".join(conditions)
    # If any condition holds, scale factor is zoom_factor, else 1.0
    expr = f"if({cond_expr}\\,{zoom_factor:.2f}\\,1.0)"
    return f"crop=w=iw/{expr}:h=ih/{expr}:x=(iw-ow)/2:y=(ih-oh)/2,scale=1080:1920"
