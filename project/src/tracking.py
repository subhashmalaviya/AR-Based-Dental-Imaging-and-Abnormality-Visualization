"""
tracking.py
-------------
Frame-to-frame cup tracking for video AR.

Why this design (see README for the full write-up)
----------------------------------------------------
Sampling tv1.mp4 showed a hand-held camera with small pan/tilt/zoom and a
static cup -- i.e. inter-frame motion is small and mostly similarity-like
(translation + slight rotation/scale), but running full GrabCut
segmentation on every 1920x1080 frame is unnecessarily slow and, because
GrabCut has no memory of the previous frame, would re-introduce
independent per-frame segmentation noise -> visible jitter. A pure
optical-flow tracker, on the other hand, would slowly drift over 293
frames with nothing to correct it.

So this module is a hybrid, "detect-and-track" design:

1. Every `redetect_interval` frames (and whenever tracking confidence
   drops), run full cup segmentation + cylinder fitting from scratch
   (cup_detection + surface_estimation) on a downscaled frame for speed,
   and reseed a fresh set of good-features-to-track inside the cup body.
2. On the other frames, track those feature points with pyramidal
   Lucas-Kanade optical flow, robustly fit a similarity transform between
   the previous and current point sets with RANSAC (rejecting outlier
   matches -- e.g. points that drifted onto the moving handle boundary or
   were lost to specular highlights), and apply that transform to the
   cylinder model's 4 corner anchors to propagate the geometry forward.
3. The resulting anchor points are exponentially smoothed (EMA) frame to
   frame to remove residual jitter without introducing noticeable lag.
4. Tracking failure (too few inlier points, or a degenerate transform) is
   explicitly detected and triggers an immediate re-detection instead of
   propagating garbage geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import cup_detection, surface_estimation
from .surface_estimation import CylinderModel

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass
class TrackerState:
    model: CylinderModel | None = None
    prev_gray: np.ndarray | None = None
    prev_points: np.ndarray | None = None
    smoothed_anchors: np.ndarray | None = None  # 4x2
    frames_since_redetect: int = 10_000
    lost_streak: int = 0


class CupTracker:
    def __init__(
        self,
        redetect_interval: int = 30,
        ema_alpha: float = 0.35,
        min_inliers: int = 12,
        detect_scale: float = 0.4,
    ):
        self.redetect_interval = redetect_interval
        self.ema_alpha = ema_alpha
        self.min_inliers = min_inliers
        self.detect_scale = detect_scale
        self.state = TrackerState()

    # ------------------------------------------------------------------
    def _full_redetect(self, frame_bgr: np.ndarray) -> CylinderModel | None:
        h, w = frame_bgr.shape[:2]
        s = self.detect_scale
        small = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
        mask_small, _ = cup_detection.segment_object(small, iterations=6)
        if not cup_detection.mask_quality_ok(mask_small):
            return None
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        try:
            model = surface_estimation.estimate_cylinder(mask)
        except ValueError:
            return None
        return model

    def _seed_features(self, gray: np.ndarray, model: CylinderModel):
        roi = model.body_mask
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=150, qualityLevel=0.01, minDistance=8, mask=roi,
        )
        return pts

    def _propagate_with_flow(self, gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        st = self.state
        if st.prev_points is None or len(st.prev_points) < 6 or st.prev_gray is None:
            return None, 0

        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            st.prev_gray, gray, st.prev_points, None, **LK_PARAMS
        )
        # forward-backward consistency check to drop unreliable matches
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, st.prev_gray, new_pts, None, **LK_PARAMS
        )
        fb_err = np.linalg.norm(st.prev_points - back_pts, axis=2).reshape(-1)
        good = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (fb_err < 3.0)

        if good.sum() < 6:
            return None, int(good.sum())

        src = st.prev_points[good].reshape(-1, 2)
        dst = new_pts[good].reshape(-1, 2)
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000
        )
        if transform is None:
            return None, 0
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        # keep only inlier points as the seed for next frame's flow
        if inlier_mask is not None:
            self.state.prev_points_for_next = dst[inlier_mask.reshape(-1).astype(bool)].reshape(-1, 1, 2).astype(np.float32)
        else:
            self.state.prev_points_for_next = dst.reshape(-1, 1, 2).astype(np.float32)

        return transform, n_inliers

    @staticmethod
    def _apply_affine(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.hstack([points, np.ones((len(points), 1))])
        return (transform @ pts.T).T

    def _plausible(self, candidate: CylinderModel) -> bool:
        """Sanity-check a fresh full-redetect against the last known-good
        smoothed geometry. Guards against a single motion-blurred /
        occluded frame confusing GrabCut and corrupting the tracked
        sequence with an implausible jump in size or position."""
        prev = self.state.smoothed_anchors
        if prev is None:
            return True
        prev_bl, prev_br = prev[2], prev[3]
        prev_radius = np.linalg.norm(prev_br - prev_bl) / 2.0
        prev_center = (prev_bl + prev_br) / 2.0

        _, _, bl, br = candidate.anchors()
        new_radius = np.linalg.norm(br - bl) / 2.0
        new_center = (bl + br) / 2.0

        if prev_radius < 1e-3:
            return True
        radius_ratio = new_radius / prev_radius
        center_shift = np.linalg.norm(new_center - prev_center) / prev_radius
        return (0.6 <= radius_ratio <= 1.6) and (center_shift <= 1.5)

    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> CylinderModel | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        st = self.state
        need_redetect = (
            st.model is None
            or st.frames_since_redetect >= self.redetect_interval
            or st.lost_streak >= 3
        )

        model = None
        if not need_redetect:
            transform, n_inliers = self._propagate_with_flow(gray)
            if transform is not None and n_inliers >= self.min_inliers:
                tl, tr, bl, br = st.model.anchors()
                anchors = np.stack([tl, tr, bl, br])
                new_anchors = self._apply_affine(transform, anchors)
                model = CylinderModel.from_anchors(
                    new_anchors[0], new_anchors[1], new_anchors[2], new_anchors[3],
                    shape=frame_bgr.shape,
                )
                st.prev_points = getattr(st, "prev_points_for_next", None)
                st.lost_streak = 0
                st.frames_since_redetect += 1
            else:
                st.lost_streak += 1

        if model is None:
            candidate = self._full_redetect(frame_bgr)
            if candidate is not None and self._plausible(candidate):
                model = candidate
                st.frames_since_redetect = 0
                st.lost_streak = 0
                st.prev_points = self._seed_features(gray, model)
            else:
                # Redetection failed or produced an implausible jump (e.g.
                # motion-blurred frame confusing GrabCut) -- freeze on the
                # last good, smoothed geometry rather than snapping to a
                # bad estimate, and try again next frame.
                st.lost_streak += 1
                if st.model is not None and st.lost_streak < 8:
                    model = st.model

        if model is not None:
            model = self._smooth(model, frame_bgr.shape)
            st.model = model
            if st.prev_points is None or len(st.prev_points) < self.min_inliers:
                st.prev_points = self._seed_features(gray, model)

        st.prev_gray = gray
        return st.model

    def _smooth(self, model: CylinderModel, shape: tuple[int, int]) -> CylinderModel:
        st = self.state
        tl, tr, bl, br = model.anchors()
        anchors = np.stack([tl, tr, bl, br])
        if st.smoothed_anchors is None:
            st.smoothed_anchors = anchors
        else:
            a = self.ema_alpha
            st.smoothed_anchors = a * anchors + (1 - a) * st.smoothed_anchors
        sm = st.smoothed_anchors
        return CylinderModel.from_anchors(sm[0], sm[1], sm[2], sm[3], shape=shape)
