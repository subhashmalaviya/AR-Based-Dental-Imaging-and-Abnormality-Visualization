"""
azimuth.py
-----------
Recovers the one pose degree of freedom that the disc ellipse cannot give:
rotation of the cup about its own axis.

Why it is missing: a surface of revolution is invariant to rotation about
its axis, so its silhouette -- and the base disc's ellipse -- are
completely unchanged by it. Five DoF come from the conic (pose_from_
ellipse.py); this sixth must come from the *texture* on the surface.

Why it matters: the logo lives at a fixed theta in cup coordinates. If the
azimuth reference wanders frame to frame, the logo slides around the mug
even though the pose "looks" right. Pinning it is what makes the overlay
attached to a physical patch of the object rather than merely to its
silhouette.

Method: back-project each frame onto the cup's (theta, v) surface -- an
"unwrap" of the visible ceramic, computed from the already-known 5-DoF
pose -- and align it to a reference unwrap. Because a change of azimuth is
*exactly* a circular shift along the theta axis of that unwrap, the
alignment is a 1-D circular cross-correlation, which is cheap and has no
local minima to get stuck in. The mug's printed flowers and text give it
plenty of signal to lock onto.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera3d import Intrinsics, Pose, project_points, transform_points
from .cup_model_3d import CupGeometry


def unwrap_surface(frame_bgr: np.ndarray, pose: Pose, geom: CupGeometry,
                    intr: Intrinsics, n_theta: int = 360, n_v: int = 64,
                    v_range: tuple[float, float] = (0.15, 0.95)
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample the frame over the cup's surface parameters.

    Returns (texture, valid): texture is (n_v, n_theta) grayscale float32,
    valid is a boolean mask of samples that are front-facing and inside the
    image. Rows span `v_range`, avoiding the very ends of the cup where the
    silhouette and the table edge contaminate the sampling.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    vv = np.linspace(v_range[0], v_range[1], n_v)
    TH, VV = np.meshgrid(th, vv)

    pts = geom.surface_point(TH, VV).reshape(-1, 3)
    nrm = geom.surface_normal(TH, VV).reshape(-1, 3)

    uv, z = project_points(pts, pose, intr)
    pc = transform_points(pts, pose)
    nc = nrm @ pose.R.T

    # front-facing: surface normal must point back towards the camera
    facing = np.einsum("ij,ij->i", nc, pc) < 0.0
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < intr.width - 1) &
              (uv[:, 1] >= 0) & (uv[:, 1] < intr.height - 1) & (z > 0))
    valid = facing & inside

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = np.clip(uv[:, 0], 0, intr.width - 1).astype(np.int32)
    y = np.clip(uv[:, 1], 0, intr.height - 1).astype(np.int32)
    tex = gray[y, x]
    tex[~valid] = 0.0

    return tex.reshape(n_v, n_theta), valid.reshape(n_v, n_theta)


def _column_profile(tex: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Collapse the unwrap to a 1-D signal over theta, zero-mean where valid."""
    w = valid.astype(np.float32)
    s = (tex * w).sum(axis=0)
    n = w.sum(axis=0)
    prof = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    seen = n > 0
    if seen.sum() > 4:
        prof[seen] -= prof[seen].mean()
    prof[~seen] = 0.0
    return prof


def circular_shift_align(prof: np.ndarray, ref: np.ndarray,
                          max_shift_frac: float = 0.5) -> tuple[int, float]:
    """Best circular shift aligning `prof` to `ref`, by FFT correlation.

    Returns (shift_bins, normalised_score). A positive shift means `prof`
    must be rolled by that many bins to match `ref`.
    """
    n = len(ref)
    if not np.any(prof) or not np.any(ref):
        return 0, 0.0
    F = np.fft.rfft(ref) * np.conj(np.fft.rfft(prof))
    corr = np.fft.irfft(F, n)
    lim = int(max_shift_frac * n)
    idx = np.concatenate([np.arange(0, lim), np.arange(n - lim, n)])
    k = idx[int(np.argmax(corr[idx]))]
    denom = np.linalg.norm(prof) * np.linalg.norm(ref) + 1e-9
    return int(k), float(corr[k] / denom)


def estimate_azimuth(frame_bgr: np.ndarray, pose_zero: Pose, geom: CupGeometry,
                      intr: Intrinsics, ref_profile: np.ndarray,
                      n_theta: int = 360) -> tuple[float, float, np.ndarray]:
    """Azimuth (radians) of this frame relative to the reference unwrap.

    `pose_zero` must be the pose built with azimuth = 0 so the measured
    shift is the azimuth itself.
    """
    tex, valid = unwrap_surface(frame_bgr, pose_zero, geom, intr, n_theta=n_theta)
    prof = _column_profile(tex, valid)
    shift, score = circular_shift_align(prof, ref_profile)
    # rolling the profile by +shift aligns it to the reference, so the cup's
    # texture sits `shift` bins further round than the reference azimuth
    az = -2.0 * np.pi * shift / n_theta
    return float(np.arctan2(np.sin(az), np.cos(az))), score, prof


def reference_profile(frame_bgr: np.ndarray, pose_zero: Pose, geom: CupGeometry,
                       intr: Intrinsics, n_theta: int = 360) -> np.ndarray:
    tex, valid = unwrap_surface(frame_bgr, pose_zero, geom, intr, n_theta=n_theta)
    return _column_profile(tex, valid)


def unwrap_visualisation(frame_bgr: np.ndarray, pose: Pose, geom: CupGeometry,
                          intr: Intrinsics, n_theta: int = 360, n_v: int = 64
                          ) -> np.ndarray:
    """Colour version of the unwrap, for the presentation figures."""
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    vv = np.linspace(0.05, 0.98, n_v)
    TH, VV = np.meshgrid(th, vv)
    pts = geom.surface_point(TH, VV).reshape(-1, 3)
    nrm = geom.surface_normal(TH, VV).reshape(-1, 3)
    uv, z = project_points(pts, pose, intr)
    pc = transform_points(pts, pose)
    nc = nrm @ pose.R.T
    valid = (np.einsum("ij,ij->i", nc, pc) < 0) & (z > 0) & \
            (uv[:, 0] >= 0) & (uv[:, 0] < intr.width - 1) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < intr.height - 1)
    x = np.clip(uv[:, 0], 0, intr.width - 1).astype(np.int32)
    y = np.clip(uv[:, 1], 0, intr.height - 1).astype(np.int32)
    out = frame_bgr[y, x].astype(np.uint8)
    out[~valid] = 0
    return out.reshape(n_v, n_theta, 3)
