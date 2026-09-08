"""
surface_estimation.py
----------------------
Turns a binary cup silhouette into a lightweight geometric model of the
cup's cylindrical body: for every image row y within the visible body,
a left/right silhouette boundary, hence a center cx(y) and a radius
(half-width) a(y).

Why this design
----------------
* The cup is imaged at an angle, so its 2D silhouette *converges*
  (perspective foreshortening) even though the physical object is a
  true cylinder. Fitting a straight line to cx(y) and a(y) over the
  clean "wall" rows captures exactly that convergence, which is what we
  need for image-space texture mapping -- we deliberately do not try to
  recover a full 3D pose/camera calibration, which would be overkill for
  a single hand-held photo/video with no calibration target.
* The handle is attached to the left side of the body and, at some
  rows, is merged into the same connected silhouette run as the body
  (no background gap between them). A naive "leftmost pixel" measurement
  would then report the handle's extent instead of the body's. We
  correct for this with a robust (sigma-clipped) line fit that treats
  handle-affected rows as outliers and ignores them.
* The top of the mask is the cup's (flat, disc-shaped) base, seen at an
  angle -- its silhouette is not part of the cylindrical wall and would
  bias the radius fit if included. We auto-detect the row at which the
  width profile stabilises (the base-to-wall transition) and only fit
  the wall region below it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import medfilt


@dataclass
class CylinderModel:
    y_top: int          # first row of the usable cylindrical wall
    y_bottom: int        # last row of the usable cylindrical wall
    left_coef: np.ndarray   # polyfit coeffs, left(y) = polyval(left_coef, y)
    right_coef: np.ndarray  # polyfit coeffs, right(y) = polyval(right_coef, y)
    body_mask: np.ndarray   # handle-excluded silhouette (uint8 0/255)
    full_mask: np.ndarray   # original cup silhouette incl. handle (uint8 0/255)

    def left(self, y):
        return np.polyval(self.left_coef, y)

    def right(self, y):
        return np.polyval(self.right_coef, y)

    def center(self, y):
        return (self.left(y) + self.right(y)) / 2.0

    def radius(self, y):
        return (self.right(y) - self.left(y)) / 2.0

    def anchors(self):
        """(top_left, top_right, bottom_left, bottom_right) points -- a
        compact 4-point summary of the model, convenient for propagating
        the geometry frame-to-frame with a tracked 2D transform."""
        yt, yb = self.y_top, self.y_bottom
        return (
            np.array([self.left(yt), yt]), np.array([self.right(yt), yt]),
            np.array([self.left(yb), yb]), np.array([self.right(yb), yb]),
        )

    @classmethod
    def from_anchors(cls, top_left, top_right, bottom_left, bottom_right,
                      shape: tuple[int, int]) -> "CylinderModel":
        """Rebuilds a model from 4 tracked anchor points (see `anchors()`).

        Deliberately rebuilds body_mask/full_mask as a fresh polygon fill
        from the anchors rather than warping the previous frame's raster
        mask forward with cv2.warpAffine: repeatedly warping a binary mask
        over many frames accumulates interpolation/rounding error (the
        silhouette slowly erodes), which was observed to progressively
        clip the logo. A fresh polygon fill from the current (tracked)
        boundary lines has no such history-dependent drift.
        """
        y_top = int(round((top_left[1] + top_right[1]) / 2))
        y_bottom = int(round((bottom_left[1] + bottom_right[1]) / 2))
        if y_bottom <= y_top:
            y_bottom = y_top + 1
        left_coef = np.polyfit([top_left[1], bottom_left[1]], [top_left[0], bottom_left[0]], 1)
        right_coef = np.polyfit([top_right[1], bottom_right[1]], [top_right[0], bottom_right[0]], 1)
        mask = polygon_body_mask(left_coef, right_coef, y_top, y_bottom, shape)
        return cls(y_top, y_bottom, left_coef, right_coef, mask, mask)


def polygon_body_mask(left_coef, right_coef, y_top: int, y_bottom: int,
                       shape: tuple[int, int], pad_frac: float = 0.05) -> np.ndarray:
    """Fresh (no history dependence) fill of the region between the fitted
    left/right boundary lines, over [y_top, y_bottom] plus a little slack."""
    h, w = shape[:2]
    pad = max(2, int(pad_frac * (y_bottom - y_top)))
    y0 = max(0, y_top - pad)
    y1 = min(h - 1, y_bottom + pad)
    mask = np.zeros((h, w), np.uint8)
    if y1 <= y0:
        return mask
    poly_left, poly_right = [], []
    for y in range(y0, y1 + 1):
        lx = int(round(np.clip(np.polyval(left_coef, y), 0, w - 1)))
        rx = int(round(np.clip(np.polyval(right_coef, y), 0, w - 1)))
        poly_left.append((lx, y))
        poly_right.append((rx, y))
    poly = np.array(poly_left + poly_right[::-1], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _runs_in_row(row: np.ndarray):
    d = np.diff(np.concatenate([[0], (row > 0).astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def _dominant_run_boundaries(mask: np.ndarray, y0: int, y1: int):
    h = mask.shape[0]
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    for y in range(y0, y1 + 1):
        runs = _runs_in_row(mask[y])
        if not runs:
            continue
        s, e = max(runs, key=lambda r: r[1] - r[0])
        left[y], right[y] = s, e
    return left, right


def _robust_fit(rows, vals, deg=1, iters=3, k=1.5, reject="none"):
    m = ~np.isnan(vals)
    r, v = rows[m].astype(float), vals[m]
    coef = np.polyfit(r, v, deg)
    for _ in range(iters):
        fit = np.polyval(coef, r)
        resid = v - fit
        std = resid.std() + 1e-6
        if reject == "low":
            keep = resid > -k * std
        elif reject == "high":
            keep = resid < k * std
        else:
            keep = np.abs(resid) < k * std
        if keep.sum() < max(10, deg + 2):
            break
        r, v = r[keep], v[keep]
        coef = np.polyfit(r, v, deg)
    return coef


def _find_wall_start(width: np.ndarray, search_start_frac=0.15, win=25, thresh=1.0):
    n = len(width)
    wsm = medfilt(width, 15 if n > 15 else (n // 2) * 2 + 1)
    dW = np.gradient(wsm)
    roll_std = np.array([
        np.std(dW[max(0, i - win):i + win]) for i in range(n)
    ])
    start = int(search_start_frac * n)
    for i in range(start, n - 40):
        if np.all(roll_std[i:i + 40] < thresh):
            return i
    return start  # fallback


def estimate_cylinder(
    mask: np.ndarray,
    handle_side: str = "auto",
    top_exclude_frac: float | None = None,
    bottom_margin: int = 10,
) -> CylinderModel:
    """Fit a CylinderModel to a cup silhouette mask.

    Parameters
    ----------
    mask : uint8 HxW binary silhouette (cup + handle).
    top_exclude_frac : if given, skip auto-detection and just exclude this
        fraction of the mask's vertical extent from the top (useful as a
        manual override / speed-up for video where geometry is stable).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("Empty mask passed to estimate_cylinder")
    y0, y1 = int(ys.min()), int(ys.max())

    left, right = _dominant_run_boundaries(mask, y0, y1)
    rows = np.arange(y0, y1 + 1)
    L = left[y0:y1 + 1]
    R = right[y0:y1 + 1]
    width = R - L
    # guard against NaN gaps for medfilt/gradient
    nan_fix = np.where(np.isnan(width), np.nanmedian(width), width)

    if top_exclude_frac is None:
        wall_start_idx = _find_wall_start(nan_fix)
    else:
        wall_start_idx = int(top_exclude_frac * len(rows))
    wall_end_idx = max(wall_start_idx + 20, len(rows) - bottom_margin)

    seg = slice(wall_start_idx, wall_end_idx)
    r_rows, r_L, r_R = rows[seg], L[seg], R[seg]

    right_coef = _robust_fit(r_rows, r_R, deg=1, reject="high")
    left_coef = _robust_fit(r_rows, r_L, deg=1, reject="low")

    body_y_top = int(rows[wall_start_idx])
    body_y_bottom = int(rows[min(wall_end_idx, len(rows) - 1)])

    # Body-only mask (handle excluded) for occlusion handling: fill the
    # polygon defined by the fitted left/right boundaries. Deliberately
    # restricted to [body_y_top, body_y_bottom] -- the only range the
    # linear fit was actually validated on. Extrapolating the line all the
    # way up to y0 (the base-disc region) can blow up or cross over on a
    # noisy fit, producing a self-intersecting polygon that wrongly
    # clips content; a little slack top/bottom is harmless since the logo
    # band is always placed inside this range anyway.
    body_mask = polygon_body_mask(left_coef, right_coef, body_y_top, body_y_bottom, mask.shape)
    # intersect with a mild dilation of the true silhouette so we never
    # draw outside the *actual* cup pixels (handles frame-edge cases)
    dil = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    body_mask = cv2.bitwise_and(body_mask, dil)

    return CylinderModel(
        y_top=body_y_top,
        y_bottom=body_y_bottom,
        left_coef=left_coef,
        right_coef=right_coef,
        body_mask=body_mask,
        full_mask=mask,
    )
