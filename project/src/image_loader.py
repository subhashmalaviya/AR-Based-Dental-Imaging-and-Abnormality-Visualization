"""
image_loader.py
----------------
Thin I/O layer: load images (with alpha support), logos, and videos.
Keeping this separate means the cup in later modules can be swapped for
a dental photo/video without touching any loading code.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image as BGR uint8 (3-channel), dropping alpha if present."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_image_bgra(path: str) -> np.ndarray:
    """Load an image preserving alpha if present; synthesizes an opaque
    alpha channel if the source has none."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def video_properties(cap: cv2.VideoCapture) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def make_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
