"""
calibrate3d.py
---------------
Estimates what a single uncalibrated monocular video can actually tell us
about the camera and the object:

  * focal length f (pixels),
  * the cup's shape ratios  height/r_top  and  r_bottom/r_top.

`r_top` itself is *not* estimated. Absolute scale is unobservable from one
monocular camera -- doubling the cup and doubling its distance produce
identical images -- so r_top is fixed as the scale gauge and everything
else is expressed relative to it. This is stated as a modelling choice,
not hidden.

Method
------
The disc ellipse alone constrains nothing about f: for any f there is a
circle pose reproducing that ellipse exactly (we measure 0.00 px residual
across the clip). What *does* constrain f is the rest of the object: given
a pose, the cup's body must project onto the observed cup silhouette, and
how strongly the body foreshortens with distance depends on f. So the
objective is silhouette agreement (IoU) between the projected 3D model and
a GrabCut segmentation of the cup, summed over keyframes spread across the
clip, with one shared (f, shape) explaining all of them.

The same objective resolves the classical two-fold ambiguity of
circle-pose-from-ellipse: both solutions reproduce the ellipse, but only
one puts the cup body where the silhouette actually is.

`focal_landscape` reports the objective as a function of f so the
conditioning can be inspected rather than assumed -- a flat curve means f
is not identifiable from this footage and the prior should be kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from . import cup_detection
from .camera3d import Intrinsics
from .cup_model_3d import CupGeometry
from .ellipse_detection import EllipseObs
from .pose_estimation3d import render_silhouette, cup_pose_candidates


@dataclass
class Keyframe:
    index: int
    ellipse: EllipseObs      # in FULL-resolution image coordinates
    body_mask: np.ndarray    # handle-free cup silhouette at working scale
    scale: float             # working_width / full_width


@dataclass
class CalibrationResult:
    focal_full_res: float
    height_ratio: float
    r_bottom_ratio: float
    mean_iou: float
    identifiable: bool
    landscape: list[tuple[float, float]]

    def geometry(self, r_top: float = 0.040) -> CupGeometry:
        return CupGeometry(r_top=r_top,
                           r_bottom=r_top * self.r_bottom_ratio,
                           height=r_top * self.height_ratio)


def body_mask_from_frame(frame_bgr: np.ndarray, work_width: int = 320
                          ) -> tuple[np.ndarray, float]:
    """GrabCut cup silhouette with the handle removed.

    The handle is a thin appendage attached to the body; a morphological
    opening whose kernel is a sizeable fraction of the body width removes
    it while leaving the body intact. Keeping it would penalise the
    (handle-less) 3D model for pixels it can never explain.
    """
    h, w = frame_bgr.shape[:2]
    s = work_width / float(w)
    small = cv2.resize(frame_bgr, (int(round(w * s)), int(round(h * s))))
    mask, _ = cup_detection.segment_object(small, iterations=5)
    if not mask.any():
        return mask, s

    xs = np.where(mask.any(axis=0))[0]
    body_w = max(int(xs.max() - xs.min()), 8)
    k = max(9, int(body_w * 0.22)) | 1
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        opened = ((lab == big).astype(np.uint8)) * 255
    return (opened if opened.any() else mask), s


def collect_keyframes(frames: list[np.ndarray], ellipses, count: int = 10,
                       work_width: int = 320) -> list[Keyframe]:
    """Pick well-spread frames with confident ellipses and segment them."""
    usable = [i for i, e in enumerate(ellipses)
              if e is not None and e.confident]
    if not usable:
        usable = [i for i, e in enumerate(ellipses) if e is not None]
    if not usable:
        return []
    picks = [usable[int(round(t))] for t in
             np.linspace(0, len(usable) - 1, min(count, len(usable)))]

    out = []
    for i in sorted(set(picks)):
        m, s = body_mask_from_frame(frames[i], work_width)
        if m.any():
            out.append(Keyframe(i, ellipses[i].obs, m, s))
    return out


def _mean_iou(params: np.ndarray, kfs: list[Keyframe], base: Intrinsics,
               r_top: float) -> float:
    f, hr, rbr = params
    if not (200.0 < f < 12000.0 and 0.6 < hr < 6.0 and 0.5 < rbr < 2.0):
        return 0.0
    geom = CupGeometry(r_top=r_top, r_bottom=r_top * rbr, height=r_top * hr)

    total = 0.0
    for kf in kfs:
        intr = Intrinsics(f, f, base.cx, base.cy, base.width, base.height).scaled(kf.scale)
        cands = cup_pose_candidates(kf.ellipse, intr, geom, 0.0)
        if not cands:
            continue
        best = 0.0
        for c in cands:
            m = render_silhouette(c.pose, geom, intr)
            if not m.any():
                continue
            inter = np.count_nonzero(m & kf.body_mask)
            union = np.count_nonzero(m | kf.body_mask)
            if union:
                best = max(best, inter / union)
        total += best
    return total / max(len(kfs), 1)


def calibrate(frames: list[np.ndarray], ellipses, base_intr: Intrinsics,
               r_top: float = 0.040, n_keyframes: int = 10,
               verbose: bool = True) -> CalibrationResult | None:
    """Estimate (f, height/r_top, r_bottom/r_top) from silhouette agreement."""
    kfs = collect_keyframes(frames, ellipses, count=n_keyframes)
    if not kfs:
        return None
    if verbose:
        print(f"  calibration keyframes: {[k.index for k in kfs]}")

    # Coarse sweep over focal length, optimising shape at each f. This both
    # initialises the joint fit and yields the identifiability landscape.
    f_grid = np.linspace(0.5 * base_intr.fx, 2.6 * base_intr.fx, 13)
    landscape: list[tuple[float, float]] = []
    best = (-1.0, None)
    for f in f_grid:
        r = minimize(lambda p: -_mean_iou(np.array([f, p[0], p[1]]), kfs, base_intr, r_top),
                     x0=np.array([2.4, 1.05]), method="Nelder-Mead",
                     options=dict(maxiter=60, xatol=1e-2, fatol=1e-4))
        iou = -r.fun
        landscape.append((float(f), float(iou)))
        if iou > best[0]:
            best = (iou, np.array([f, r.x[0], r.x[1]]))
        if verbose:
            print(f"    f={f:7.1f}px  IoU={iou:.4f}  h/r={r.x[0]:.2f} rb/rt={r.x[1]:.2f}")

    # Joint refinement from the best grid point.
    res = minimize(lambda p: -_mean_iou(p, kfs, base_intr, r_top),
                   x0=best[1], method="Nelder-Mead",
                   options=dict(maxiter=300, xatol=1e-2, fatol=1e-5))
    f, hr, rbr = res.x
    iou = -res.fun

    # Is f actually identifiable? Compare the peak against the best IoU
    # achievable far away from it; if the objective barely degrades, the
    # data does not pin f down and we say so instead of pretending.
    ious = np.array([v for _, v in landscape])
    fs = np.array([k for k, _ in landscape])
    far = np.abs(fs - f) > 0.45 * f
    identifiable = bool(far.any() and (iou - ious[far].max()) > 0.02)

    if verbose:
        print(f"  -> f={f:.1f}px  height/r_top={hr:.2f}  r_bottom/r_top={rbr:.2f}"
              f"  IoU={iou:.4f}  identifiable={identifiable}")
    return CalibrationResult(float(f), float(hr), float(rbr), float(iou),
                             identifiable, landscape)
