"""
disc_tracker.py
----------------
Sequence-level tracking of the cup's end-disc ellipse, using the
detect-once-then-track strategy documented in ellipse_detection.py.

Per frame the tracker tries, in order of cost:

  1. local refit inside a narrow band around a constant-velocity
     prediction of the previous rim (cheap, and geometrically immune to
     the wall/table edges that defeat global fitting),
  2. the same refit with a widened band, for fast camera motion,
  3. a full global RANSAC re-detection, accepted on its own merit when the
     tracker has lost lock (so a stale prior cannot veto recovery),
  4. coasting on the previous estimate, flagged low-confidence.

Step 3 matters: an earlier version validated re-detections against the
last tracked ellipse, which meant that once the prior went stale during a
motion-blurred passage the tracker could never re-acquire and 203 of 293
frames coasted. Recovery must not be gated on the thing that failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ellipse_detection import (EllipseObs, detect_disc_ellipse, plausible_step,
                                 refine_ellipse)


@dataclass
class TrackedEllipse:
    obs: EllipseObs
    frame_index: int
    source: str          # "detect" | "refine" | "refine_wide" | "redetect" | "coast"
    confident: bool


def _predict(prev: EllipseObs, pprev: EllipseObs | None):
    """Constant-velocity prediction of the rim centre for the next frame.

    Only the centre is extrapolated. Extrapolating the *axes* as well
    creates a positive feedback loop -- a slightly-too-large prediction
    pushes the search band outward, the refit lands on the outer edge, and
    the next prediction grows again. That produced a perfectly linear
    runaway from frame 234 (mean radius climbing 703 -> 1297 px, i.e.
    larger than the 1080 px frame width). Size changes slowly enough that
    the band alone tracks it.
    """
    if pprev is None:
        return (prev.center, prev.axes, prev.angle)
    dx = prev.center[0] - pprev.center[0]
    dy = prev.center[1] - pprev.center[1]
    return ((prev.center[0] + dx, prev.center[1] + dy), prev.axes, prev.angle)


def _size_ok(obs: EllipseObs, shape) -> bool:
    """Absolute sanity bound: the disc cannot dwarf the image."""
    h, w = shape[:2]
    return obs.mean_radius < 0.75 * max(h, w)


def _quality(obs: EllipseObs) -> float:
    return obs.inliers * obs.coverage


def _band_for(obs_or_prior) -> float:
    """Search band scales with the rim's size (bigger disc -> bigger motion)."""
    axes = obs_or_prior[1] if isinstance(obs_or_prior, tuple) else obs_or_prior.axes
    mean_r = 0.25 * (axes[0] + axes[1])
    return float(np.clip(0.075 * mean_r, 24.0, 80.0))


class DiscEllipseTracker:
    """Stateful, streaming-friendly tracker for the end-disc ellipse."""

    def __init__(self, min_anchor_score: float = 0.55,
                  redetect_score: float = 0.55, max_coast: int = 5,
                  min_inliers: int = 200, min_coverage: float = 0.50):
        self.min_anchor_score = min_anchor_score
        self.redetect_score = redetect_score
        self.max_coast = max_coast
        # Minimum evidence for calling a frame tracked. Without this the
        # tracker happily "tracks" noise: the constant-velocity prediction
        # is accepted every frame on a handful of stray edge inliers, so
        # the rim estimate marches linearly across the image while still
        # reporting success (observed on frames 89-128, where the radius
        # froze at 369 px and support fell to 1-8 inliers). Good frames
        # carry 1300-2200 inliers at coverage 0.75-1.0, so this gate is
        # far below any genuine measurement.
        self.min_inliers = min_inliers
        self.min_coverage = min_coverage
        self.prev: EllipseObs | None = None
        self.pprev: EllipseObs | None = None
        self.coast_run = 0

    def _supported(self, obs: EllipseObs) -> bool:
        return obs.inliers >= self.min_inliers and obs.coverage >= self.min_coverage

    def update(self, frame_bgr: np.ndarray, frame_index: int = -1) -> TrackedEllipse | None:
        if self.prev is None:
            obs = detect_disc_ellipse(frame_bgr)
            if obs is None or obs.score < self.min_anchor_score \
                    or not _size_ok(obs, frame_bgr.shape) or not self._supported(obs):
                return None
            self.prev, self.pprev, self.coast_run = obs, None, 0
            return TrackedEllipse(obs, frame_index, "detect", True)

        prior = _predict(self.prev, self.pprev)
        band = _band_for(prior)

        for tag, b in (("refine", band), ("refine_wide", band * 2.2)):
            cand = refine_ellipse(frame_bgr, prior, band_px=b)
            if cand is not None and plausible_step(self.prev, cand) \
                    and _size_ok(cand, frame_bgr.shape) and self._supported(cand):
                self._accept(cand)
                return TrackedEllipse(cand, frame_index, tag, True)

        # Lost lock: re-detect on the frame's own merit, not against a
        # possibly-stale prior.
        cand = detect_disc_ellipse(frame_bgr)
        if cand is not None and _size_ok(cand, frame_bgr.shape) and self._supported(cand):
            trust_prior = self.coast_run < 3
            if (not trust_prior and cand.score >= self.redetect_score) or \
                    (trust_prior and plausible_step(self.prev, cand)):
                self._accept(cand, reset_velocity=True)
                return TrackedEllipse(cand, frame_index, "redetect", True)

        self.coast_run += 1
        if self.coast_run > self.max_coast:
            self.prev = self.pprev = None      # give up; re-anchor next frame
            self.coast_run = 0
            return None
        return TrackedEllipse(self.prev, frame_index, "coast", False)

    def _accept(self, obs: EllipseObs, reset_velocity: bool = False):
        self.pprev = None if reset_velocity else self.prev
        self.prev = obs
        self.coast_run = 0


def _interpolate_gaps(results: list[TrackedEllipse | None], n: int
                       ) -> list[TrackedEllipse | None]:
    """Fill frames with no confident estimate by interpolating between the
    nearest confident neighbours.

    Legitimate here because the camera motion is smooth and continuous: a
    gap means the *measurement* failed (motion blur), not that the rim
    moved discontinuously. Interpolated frames stay flagged
    (`confident=False`) so downstream stages can down-weight them.
    """
    good = [i for i in range(n) if results[i] is not None and results[i].confident]
    if not good:
        return results
    for i in range(n):
        if results[i] is not None and results[i].confident:
            continue
        prev_g = max((g for g in good if g < i), default=None)
        next_g = min((g for g in good if g > i), default=None)
        if prev_g is None and next_g is None:
            continue
        if prev_g is None:
            src = results[next_g].obs
        elif next_g is None:
            src = results[prev_g].obs
        else:
            a, b = results[prev_g].obs, results[next_g].obs
            t = (i - prev_g) / float(next_g - prev_g)
            # unwrap the angle so interpolation does not take the long way
            da = (b.angle - a.angle + 90.0) % 180.0 - 90.0
            src = EllipseObs(
                center=(a.center[0] + t * (b.center[0] - a.center[0]),
                        a.center[1] + t * (b.center[1] - a.center[1])),
                axes=(a.axes[0] + t * (b.axes[0] - a.axes[0]),
                      a.axes[1] + t * (b.axes[1] - a.axes[1])),
                angle=a.angle + t * da,
                score=0.0, inliers=0,
                coverage=min(a.coverage, b.coverage), clipped=a.clipped or b.clipped,
            )
        results[i] = TrackedEllipse(src, i, "interp", False)
    return results


def _params(obs: EllipseObs) -> np.ndarray:
    """Ellipse as a smoothing-friendly vector.

    Orientation is carried as (cos 2t, sin 2t) because an ellipse's angle
    is only defined modulo 180 degrees -- filtering the raw angle would
    corrupt every wrap through 0/180.
    """
    a_min, a_max = sorted(obs.axes)
    t = np.deg2rad(obs.angle if obs.axes[1] >= obs.axes[0] else obs.angle + 90.0)
    return np.array([obs.center[0], obs.center[1], a_min, a_max,
                     np.cos(2 * t), np.sin(2 * t)])


def _unparams(p: np.ndarray, ref: EllipseObs, source: str, i: int) -> TrackedEllipse:
    t = 0.5 * np.arctan2(p[5], p[4])
    a_min, a_max = max(2.0, p[2]), max(2.0, p[3])
    obs = EllipseObs(center=(float(p[0]), float(p[1])),
                      axes=(float(a_min), float(a_max)),
                      angle=float(np.rad2deg(t)),
                      score=ref.score, inliers=ref.inliers,
                      coverage=ref.coverage, clipped=ref.clipped)
    return TrackedEllipse(obs, i, source, ref.score > 0)


def clean_sequence(results: list[TrackedEllipse | None], med_win: int = 7,
                    center_frac: float = 0.30, axis_frac: float = 0.30
                    ) -> list[TrackedEllipse | None]:
    """Reject temporal outliers in the tracked ellipse trajectory.

    The forward/backward merge picks the better-supported estimate per
    frame independently, so it can switch between two locally-plausible
    tracks and introduce a step discontinuity. Comparing each frame
    against a running median and replacing robust outliers removes those
    switches without smoothing away genuine fast camera motion.
    """
    idx = [i for i, r in enumerate(results) if r is not None]
    if len(idx) < med_win:
        return results
    P = np.stack([_params(results[i].obs) for i in idx])

    k = med_win | 1
    half = k // 2
    med = np.empty_like(P)
    for j in range(len(P)):
        lo, hi = max(0, j - half), min(len(P), j + half + 1)
        med[j] = np.median(P[lo:hi], axis=0)

    # Scale-relative thresholds rather than a global MAD: the ellipse grows
    # substantially over the clip, so a single MAD across the whole
    # sequence is dominated by genuine motion and flags almost everything
    # (an earlier version marked 238 of 293 frames as outliers).
    med_r = 0.5 * (med[:, 2] + med[:, 3])
    center_dev = np.hypot(P[:, 0] - med[:, 0], P[:, 1] - med[:, 1])
    axis_dev = np.maximum(np.abs(P[:, 2] - med[:, 2]) / np.maximum(med[:, 2], 1e-6),
                          np.abs(P[:, 3] - med[:, 3]) / np.maximum(med[:, 3], 1e-6))
    bad = (center_dev > center_frac * med_r) | (axis_dev > axis_frac)

    for j, i in enumerate(idx):
        if bad[j]:
            results[i] = _unparams(med[j], results[i].obs, "outlier_fixed", i)
    return results


def track_sequence(frames: list[np.ndarray], verbose: bool = False
                    ) -> list[TrackedEllipse | None]:
    """Track the disc across a whole (in-memory) sequence.

    Runs the tracker forward over the entire clip and then backward over
    it, and keeps the better-supported estimate per frame. Two passes
    matter because a passage of motion blur blocks the tracker travelling
    one way but is usually traversable from the other side -- a
    single forward pass left frames 110-143 unrecoverable. Any frames
    still missing are interpolated.
    """
    n = len(frames)

    def _run(order):
        out: list[TrackedEllipse | None] = [None] * n
        tr = DiscEllipseTracker()
        for i in order:
            r = tr.update(frames[i], i)
            if r is None:                        # dropped lock -> try to re-anchor
                r = tr.update(frames[i], i)
            out[i] = r
        return out

    fwd = _run(range(n))
    bwd = _run(range(n - 1, -1, -1))

    merged: list[TrackedEllipse | None] = [None] * n
    for i in range(n):
        a, b = fwd[i], bwd[i]
        cands = [c for c in (a, b) if c is not None and c.confident]
        if cands:
            merged[i] = max(cands, key=lambda c: _quality(c.obs))
        else:
            merged[i] = a or b

    if verbose:
        nf = sum(1 for r in fwd if r and r.confident)
        nb = sum(1 for r in bwd if r and r.confident)
        nm = sum(1 for r in merged if r and r.confident)
        print(f"  forward {nf}/{n} confident, backward {nb}/{n}, merged {nm}/{n}")

    merged = _interpolate_gaps(merged, n)
    return clean_sequence(merged)
