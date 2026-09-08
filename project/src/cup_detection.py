"""
cup_detection.py
-----------------
Segments the cup (or, later, a dental object) away from the background.

Method: GrabCut initialised from a generous central rectangle. This was
chosen after testing plain HSV/saturation thresholding, which failed
because the wooden table and the cup body have overlapping saturation/
value ranges under the photo's warm lighting (see project README for the
comparison). GrabCut instead builds foreground/background colour models
from the whole image and is robust to that overlap.

For video frames, the previous frame's mask can be supplied as a prior
(``prev_mask``) to warm-start GrabCut with GC_INIT_WITH_MASK, which is both
faster (fewer iterations needed) and more temporally consistent than
re-running GC_INIT_WITH_RECT from scratch every frame.
"""
from __future__ import annotations

import cv2
import numpy as np


def _largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def segment_object(
    image_bgr: np.ndarray,
    rect_margin: float = 0.03,
    iterations: int = 6,
    prev_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the dominant foreground object (the cup).

    Returns
    -------
    mask : uint8 HxW, 0/255, filled silhouette of the largest foreground blob.
    contour : Nx1x2 int32 contour points of that silhouette.
    """
    h, w = image_bgr.shape[:2]
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if prev_mask is not None and prev_mask.any():
        # Warm start: dilate previous mask for "probable FG", erode for
        # "sure FG", everything else "probable BG". Keeps GrabCut cheap and
        # temporally stable across video frames.
        sure_fg = cv2.erode(prev_mask, np.ones((15, 15), np.uint8))
        probable_fg = cv2.dilate(prev_mask, np.ones((25, 25), np.uint8))
        gc_mask[:] = cv2.GC_PR_BGD
        gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD
        cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model,
                    max(2, iterations - 3), cv2.GC_INIT_WITH_MASK)
    else:
        rect = (
            int(w * rect_margin), int(h * rect_margin),
            int(w * (1 - 2 * rect_margin)), int(h * (1 - 2 * rect_margin)),
        )
        cv2.grabCut(image_bgr, gc_mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_RECT)

    binary = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    contour = _largest_external_contour(binary)
    if contour is None:
        return np.zeros((h, w), np.uint8), np.zeros((0, 1, 2), np.int32)

    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled, contour


def mask_quality_ok(mask: np.ndarray, min_area_frac: float = 0.03) -> bool:
    """Sanity check used by the video tracker to detect segmentation failure."""
    h, w = mask.shape[:2]
    area = int((mask > 0).sum())
    return area > min_area_frac * h * w
