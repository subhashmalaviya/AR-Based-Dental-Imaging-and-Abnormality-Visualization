"""
logo_processing.py
--------------------
Prepares the IIT Delhi logo for cylindrical mapping: builds a clean alpha
mask (the supplied PNG has a flat white background and no alpha channel),
trims to the logo's bounding box, and hands back a float32 RGBA image.

The logo's own pixels (red emblem, Hindi + English ring text) are never
redrawn or altered -- we only classify background vs. foreground and feather
the resulting edge for anti-aliasing.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_alpha_from_white_background(
    bgr: np.ndarray, white_thresh: int = 245, feather: int = 3
) -> np.ndarray:
    """Foreground = anything not close to pure white. Works well for logos
    scanned/exported on a clean white canvas (true for iitd_logo.png)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg = (gray < white_thresh).astype(np.uint8) * 255

    # fill small internal specks / smooth edges
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if feather > 0:
        fg = cv2.GaussianBlur(fg, (feather * 2 + 1, feather * 2 + 1), 0)
    return fg


def load_logo_rgba(path: str, feather: int = 3) -> np.ndarray:
    """Returns an HxWx4 uint8 RGBA image: (B, G, R, A) to match OpenCV
    channel order, trimmed to the tight bounding box of the logo."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)

    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr = raw[..., :3]
        alpha = raw[..., 3]
        # If the supplied alpha is trivially all-opaque, it doesn't encode
        # real transparency -- derive it from the white background instead.
        if alpha.min() > 250:
            alpha = build_alpha_from_white_background(bgr, feather=feather)
    else:
        bgr = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        alpha = build_alpha_from_white_background(bgr, feather=feather)

    rgba = np.dstack([bgr, alpha]).astype(np.uint8)

    ys, xs = np.where(alpha > 8)
    if len(ys) == 0:
        return rgba
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgba[y0:y1, x0:x1]


def resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgba, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)
