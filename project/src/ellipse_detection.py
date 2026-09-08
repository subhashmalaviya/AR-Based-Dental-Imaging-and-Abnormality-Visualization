"""
ellipse_detection.py
---------------------
Locates the image ellipse of the cup's end disc. This is the measurement
that drives 3D pose estimation (see pose_from_ellipse.py): the disc is a
true circle on the rigid object, so its image ellipse determines 5 of the
6 pose DoF in closed form.

Design: detect-once-then-track
------------------------------
Single-frame detection of this rim turned out to be genuinely hard, and it
is worth recording why, because the failure modes drove the design:

  * Brightness alone confuses the disc with the table (measured frame 0:
    disc V=226, table V=156, wall V=105).
  * Brightness + low saturation separates disc from table and handle, but
    the *white ceramic wall* is also bright and unsaturated on many
    frames; a global threshold tuned on frame 0 flooded to 20-40% of the
    image on frames 80/120/160.
  * Fitting an ellipse to the thresholded blob's outline fails once that
    blob merges disc and wall (measured fill 0.71, ellipse score 0.12 on
    frame 80).
  * Global RANSAC over Canny edges recovers frame 0 well (score 0.98) but
    without strong geometric gates it happily returns an enormous
    near-straight ellipse threaded through unrelated edges (observed:
    centres several hundred px off-screen, axes > 3000 px).

The physical rim is, however, always a strong *intensity edge*, and
between adjacent frames it barely moves. So detection is only used to
bootstrap: `detect_disc_ellipse` runs RANSAC with geometric gates and
reports a quality score, the caller anchors on the best-scoring frame, and
`refine_ellipse` then tracks the rim by re-fitting inside a narrow band
around the previous estimate -- a far better conditioned problem, because
the band excludes nearly all the wall and flower edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EllipseObs:
    center: tuple[float, float]
    axes: tuple[float, float]     # full axis lengths, OpenCV convention
    angle: float                  # degrees
    score: float                  # 0..1 support quality
    inliers: int
    coverage: float               # fraction of the rim actually observed
    clipped: bool                 # disc runs off the image border

    def as_cv(self):
        return (self.center, self.axes, self.angle)

    @property
    def mean_radius(self) -> float:
        return 0.25 * (self.axes[0] + self.axes[1])


# --------------------------------------------------------------- helpers
def _residual(ellipse, pts: np.ndarray) -> np.ndarray:
    """Normalised radial distance of points from the ellipse (0 = on it)."""
    (xc, yc), (MA, ma), ang = ellipse
    a, b = MA / 2.0, ma / 2.0
    if a < 1e-6 or b < 1e-6:
        return np.full(len(pts), 1e6)
    p = np.deg2rad(ang)
    c, s = np.cos(p), np.sin(p)
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    return np.abs(np.sqrt((xr / a) ** 2 + (yr / b) ** 2) - 1.0)


def _coverage(ellipse, pts: np.ndarray, n_bins: int = 36) -> float:
    """Fraction of angular bins around the ellipse carrying support.

    A real rim is sampled all the way round; a spurious giant ellipse only
    grazes edges along one short arc. This is the single most effective
    guard against the degenerate RANSAC solutions described above.
    """
    if len(pts) == 0:
        return 0.0
    (xc, yc), (MA, ma), ang = ellipse
    a, b = max(MA / 2.0, 1e-6), max(ma / 2.0, 1e-6)
    p = np.deg2rad(ang)
    c, s = np.cos(p), np.sin(p)
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    xr = (dx * c + dy * s) / a
    yr = (-dx * s + dy * c) / b
    bins = ((np.arctan2(yr, xr) + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(np.unique(bins)) / n_bins


def _edge_points(frame_bgr: np.ndarray, mask: np.ndarray | None = None,
                  border_margin: int = 3) -> np.ndarray:
    edges = cv2.Canny(cv2.GaussianBlur(frame_bgr, (5, 5), 0), 40, 120)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    ys, xs = np.where(edges > 0)
    h, w = frame_bgr.shape[:2]
    keep = ((xs > border_margin) & (xs < w - 1 - border_margin) &
            (ys > border_margin) & (ys < h - 1 - border_margin))
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)


def _bright_roi(frame_bgr: np.ndarray, percentile: float = 98.5) -> np.ndarray:
    """Generous bright/unsaturated region used only to bound the search."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.int32)
    V = hsv[..., 2].astype(np.int32)
    low_s = S < 90
    if low_s.sum() < 500:
        return np.zeros(frame_bgr.shape[:2], np.uint8)
    thr = max(float(np.percentile(V[low_s], percentile)) * 0.82, 120.0)
    m = ((low_s) & (V >= thr)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == k).astype(np.uint8)) * 255


def _quality(ellipse, pts: np.ndarray, tol: float, clipped: bool):
    res = _residual(ellipse, pts)
    inl_pts = pts[res < tol]
    inl = len(inl_pts)
    near = int((res < 0.15).sum())
    cov = _coverage(ellipse, inl_pts)
    # A clipped disc legitimately shows less rim, so normalise against a
    # lower expectation rather than penalising it.
    target = 0.55 if clipped else 0.80
    score = (inl / max(near, 1)) * min(1.0, cov / target)
    return float(score), inl, float(cov)


# ------------------------------------------------------------- detection
def detect_disc_ellipse(frame_bgr: np.ndarray,
                         ransac_iters: int = 1200,
                         tol: float = 0.02,
                         rng: np.random.Generator | None = None,
                         ) -> EllipseObs | None:
    """Bootstrap detection: robust global fit with geometric gates."""
    rng = rng or np.random.default_rng(12345)
    h, w = frame_bgr.shape[:2]

    roi = _bright_roi(frame_bgr)
    if (roi > 0).sum() < 0.004 * h * w:
        return None

    ys, xs = np.where(roi > 0)
    roi_cx, roi_cy = xs.mean(), ys.mean()
    roi_w = xs.max() - xs.min() + 1
    roi_h = ys.max() - ys.min() + 1
    roi_diag = float(np.hypot(roi_w, roi_h))
    clipped = bool(xs.min() <= 3 or ys.min() <= 3 or
                   xs.max() >= w - 4 or ys.max() >= h - 4)

    pts = _edge_points(frame_bgr, cv2.dilate(roi, np.ones((21, 21), np.uint8)))
    if len(pts) < 80:
        return None
    if len(pts) > 4000:
        pts = pts[rng.choice(len(pts), 4000, replace=False)]

    min_axis = 0.30 * min(roi_w, roi_h)
    max_axis = 1.7 * roi_diag
    max_center_off = 0.60 * roi_diag

    best = None
    for _ in range(ransac_iters):
        sample = pts[rng.choice(len(pts), 5, replace=False)].astype(np.float32)
        try:
            cand = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        # cv2.fitEllipse returns (width, height) in the ellipse's own rotated
        # frame -- NOT sorted major-first -- so sort before testing.
        a_min, a_max = sorted(cand[1])
        if a_min < min_axis or a_max > max_axis or a_max > 3.0 * a_min:
            continue
        if np.hypot(cand[0][0] - roi_cx, cand[0][1] - roi_cy) > max_center_off:
            continue
        res = _residual(cand, pts)
        keep = res < tol
        if keep.sum() < 60:
            continue
        cov = _coverage(cand, pts[keep])
        if cov < 0.45:
            continue
        rank = int(keep.sum()) * cov
        if best is None or rank > best[0]:
            best = (rank, cand)

    if best is None:
        return None

    ell = _irls(best[1], pts, tol, (h, w))
    if ell is None:
        return None
    score, inl, cov = _quality(ell, pts, tol * 1.5, clipped)
    if inl < 60 or cov < (0.30 if clipped else 0.45):
        return None
    return EllipseObs(ell[0], ell[1], ell[2], score, inl, cov, clipped)


def _irls(ellipse, pts: np.ndarray, tol: float, shape, iters: int = 6,
           anchor=None, max_axis_drift: float = 1.35,
           min_cov: float = 0.0):
    """Iteratively reweighted ellipse refit.

    `anchor` (default: the starting ellipse) bounds how far the solution may
    drift. Without this, plain IRLS diverges: on motion-blurred frames the
    rim edge weakens and the fit collapses onto an unrelated sub-cluster
    inside the search band (measured: a 523x880 prior collapsing to 86x227
    in a single iteration on frame 66). Each iteration is accepted only if
    it stays within `max_axis_drift` of the anchor on both axes and keeps
    enough angular support.
    """
    h, w = shape
    anchor = anchor if anchor is not None else ellipse
    a_min0, a_max0 = sorted(anchor[1])
    cur = ellipse
    for _ in range(iters):
        keep = _residual(cur, pts) < tol * 1.5
        if keep.sum() < 40:
            break
        try:
            nxt = cv2.fitEllipse(pts[keep].astype(np.float32))
        except cv2.error:
            break
        n_min, n_max = sorted(nxt[1])
        if n_min < 0.04 * min(h, w) or n_max > 3.0 * max(h, w):
            break
        if a_min0 > 1e-6 and a_max0 > 1e-6:
            lo, hi = 1.0 / max_axis_drift, max_axis_drift
            if not (lo <= n_min / a_min0 <= hi and lo <= n_max / a_max0 <= hi):
                break
        if min_cov > 0.0:
            inl = pts[_residual(nxt, pts) < tol * 1.5]
            if _coverage(nxt, inl) < min_cov:
                break
        cur = nxt
    return cur


# -------------------------------------------------------------- tracking
def refine_ellipse(frame_bgr: np.ndarray, prior, band_px: float = 30.0,
                    tol: float = 0.02) -> EllipseObs | None:
    """Track the rim by re-fitting inside a narrow band around `prior`.

    `prior` is a (center, axes, angle) tuple, typically the previous
    frame's result (optionally motion-predicted). The band is what makes
    this robust: it geometrically excludes the wall texture and table
    edges that defeat global fitting.
    """
    h, w = frame_bgr.shape[:2]
    band = np.zeros((h, w), np.uint8)
    # Use the RotatedRect overload: the (center, axes, angle, ...) form
    # requires integer center/axes and silently rejects float input.
    box = ((float(prior[0][0]), float(prior[0][1])),
           (float(prior[1][0]), float(prior[1][1])), float(prior[2]))
    try:
        cv2.ellipse(band, box, 255, thickness=int(max(6, round(band_px * 2))))
    except (cv2.error, ValueError):
        return None

    pts = _edge_points(frame_bgr, band)
    if len(pts) < 50:
        return None

    ell = _irls(prior, pts, tol, (h, w), anchor=prior,
                max_axis_drift=1.30, min_cov=0.35)
    if ell is None:
        return None

    # Final guard: both axes must stay close to the prior's, which a
    # collapsed fit cannot satisfy.
    p_min, p_max = sorted(prior[1])
    e_min, e_max = sorted(ell[1])
    if p_min > 1e-6 and p_max > 1e-6:
        if not (0.70 <= e_min / p_min <= 1.45 and 0.70 <= e_max / p_max <= 1.45):
            return None

    ys, xs = np.where(band > 0)
    clipped = bool(xs.min() <= 3 or ys.min() <= 3 or
                   xs.max() >= w - 4 or ys.max() >= h - 4)
    score, inl, cov = _quality(ell, pts, tol * 1.5, clipped)
    if inl < 40:
        return None
    return EllipseObs(ell[0], ell[1], ell[2], score, inl, cov, clipped)


def plausible_step(prev: EllipseObs, cur: EllipseObs,
                    max_center_frac: float = 0.35,
                    scale_range: tuple[float, float] = (0.75, 1.33)) -> bool:
    """Reject a tracked ellipse that jumped implausibly between frames.

    Checks position, size, *and* shape. Mean radius alone is not enough: a
    degenerate slit can have a mean radius close to the true disc's while
    being completely wrong, which is exactly how frame 160 first slipped
    through.
    """
    r = max(prev.mean_radius, 1e-6)
    d = np.hypot(cur.center[0] - prev.center[0], cur.center[1] - prev.center[1])
    if d > max_center_frac * r:
        return False
    if not (scale_range[0] <= cur.mean_radius / r <= scale_range[1]):
        return False

    p_min, p_max = sorted(prev.axes)
    c_min, c_max = sorted(cur.axes)
    if p_min < 1e-6 or c_min < 1e-6:
        return False
    # area and elongation must both evolve smoothly
    if not (0.60 <= (c_min * c_max) / (p_min * p_max) <= 1.65):
        return False
    elong_ratio = (c_max / c_min) / (p_max / p_min)
    return 0.65 <= elong_ratio <= 1.55


def draw_ellipse(frame_bgr: np.ndarray, obs: EllipseObs,
                  color=(0, 255, 255), thickness: int = 3) -> np.ndarray:
    out = frame_bgr.copy()
    cv2.ellipse(out, obs.as_cv(), color, thickness, cv2.LINE_AA)
    cv2.circle(out, (int(obs.center[0]), int(obs.center[1])), 6, (0, 0, 255), -1)
    return out
