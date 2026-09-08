"""
video_ar3d.py
--------------
The full markerless 3D AR pipeline, per frame:

    frame
      -> disc ellipse            (disc_tracker.py)
      -> 6-DoF cup pose          (pose_from_ellipse.py + pose_estimation3d.py)
      -> azimuth about the axis  (azimuth.py)
      -> temporal smoothing      (this module)
      -> rasterise textured mesh (renderer3d.py + texture_atlas.py)
      -> composite as print      (this module)

The logo never exists in 2D. It is texture at a fixed (theta, v) on the
cup's surface; what changes per frame is only the camera pose used to
project that surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .azimuth import estimate_azimuth, reference_profile
from .camera3d import Intrinsics, Pose
from .cup_model_3d import CupGeometry, Mesh, build_cup_mesh
from .disc_tracker import track_sequence
from .pose_estimation3d import cup_pose_candidates, reprojection_error
from .renderer3d import render_mesh, view_directions
from .texture_atlas import (LogoPlacement, build_atlas, face_has_texture,
                             load_logo_rgba, sample_atlas)


# ------------------------------------------------------------------ poses
def _quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        vals = [1 + R[0, 0] - R[1, 1] - R[2, 2],
                1 - R[0, 0] + R[1, 1] - R[2, 2],
                1 - R[0, 0] - R[1, 1] + R[2, 2]]
        i = int(np.argmax(vals))
        q = np.zeros(4)
        q[i + 1] = np.sqrt(max(0.0, vals[i])) / 2.0
        d = 4 * q[i + 1] + 1e-12
        if i == 0:
            q[0] = (R[2, 1] - R[1, 2]) / d
            q[2] = (R[0, 1] + R[1, 0]) / d
            q[3] = (R[0, 2] + R[2, 0]) / d
        elif i == 1:
            q[0] = (R[0, 2] - R[2, 0]) / d
            q[1] = (R[0, 1] + R[1, 0]) / d
            q[3] = (R[1, 2] + R[2, 1]) / d
        else:
            q[0] = (R[1, 0] - R[0, 1]) / d
            q[1] = (R[0, 2] + R[2, 0]) / d
            q[2] = (R[1, 2] + R[2, 1]) / d
        return q / (np.linalg.norm(q) + 1e-12)
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def _rot(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def smooth_poses(poses: list[Pose | None], alpha: float = 0.45) -> list[Pose | None]:
    """Causal EMA on rotation (via quaternions) and translation.

    Rotations are averaged as quaternions with hemisphere alignment rather
    than by filtering Rodrigues vectors, which would break at the +/-pi
    wrap and is not a linear space anyway.
    """
    out: list[Pose | None] = [None] * len(poses)
    q_s = None
    t_s = None
    for i, p in enumerate(poses):
        if p is None:
            out[i] = None
            continue
        q = _quat(p.R)
        t = p.tvec
        if q_s is None:
            q_s, t_s = q, t
        else:
            if np.dot(q, q_s) < 0:            # same rotation, opposite sign
                q = -q
            q_s = alpha * q + (1 - alpha) * q_s
            q_s /= np.linalg.norm(q_s) + 1e-12
            t_s = alpha * t + (1 - alpha) * t_s
        out[i] = Pose.from_Rt(_rot(q_s), t_s)
    return out


def _unwrap_series(a: np.ndarray) -> np.ndarray:
    """Unwrap a sequence of angles and median-filter out jumps."""
    a = np.unwrap(a)
    if len(a) >= 5:
        k = 5
        pad = np.pad(a, k // 2, mode="edge")
        med = np.array([np.median(pad[i:i + k]) for i in range(len(a))])
        bad = np.abs(a - med) > np.deg2rad(25)
        a = np.where(bad, med, a)
    return a


# ------------------------------------------------------------ compositing
@dataclass
class ARConfig:
    geometry: CupGeometry = field(
        default_factory=lambda: CupGeometry(0.035, 0.035, 0.0858, 1.0))
    focal: float = 1500.0
    placement: LogoPlacement = field(default_factory=LogoPlacement)
    opacity: float = 0.95
    blend_strength: float = 0.55
    texture_strength: float = 0.08
    pose_smoothing: float = 0.45
    mesh_theta: int = 96
    mesh_v: int = 24


def composite_print(frame_bgr: np.ndarray, fb, atlas: np.ndarray,
                     intr: Intrinsics, cfg: ARConfig) -> np.ndarray:
    """Blend the rendered texture into the frame as ink on ceramic.

    Shading uses the *3D surface normal* from the rasteriser (n . v), so
    the falloff toward the silhouette comes from the geometry rather than
    from any 2D approximation, and it is modulated by the frame's own
    luminance so existing highlights and shadows still read through.
    """
    if not fb.mask.any():
        return frame_bgr

    ys, xs = np.where(fb.mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    sub = np.s_[y0:y1, x0:x1]
    region = frame_bgr[sub].astype(np.float32)

    tex = sample_atlas(atlas, fb.uv[sub])
    a = (tex[..., 3:4] / 255.0) * cfg.opacity
    a = a * fb.mask[sub][..., None]
    if a.max() <= 0:
        return frame_bgr

    v = view_directions(fb, intr)[sub]
    facing = np.clip(np.einsum("ijk,ijk->ij", fb.normal[sub], v), 0.0, 1.0)
    shading = (0.35 + 0.65 * facing)[..., None]

    base_lum = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2GRAY
                             ).astype(np.float32)[..., None] / 255.0
    ink = tex[..., :3] * shading * (0.5 + 0.5 * base_lum)

    multiplied = (ink / 255.0) * (region / 255.0) * 255.0
    printed = (1 - cfg.blend_strength) * ink + cfg.blend_strength * multiplied

    if cfg.texture_strength > 0:
        hi = region - cv2.GaussianBlur(region, (0, 0), 2.0)
        printed = printed + hi * cfg.texture_strength

    out = frame_bgr.copy()
    out[sub] = np.clip(region * (1 - a) + np.clip(printed, 0, 255) * a,
                       0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- driver
def theta_facing_camera(pose: Pose, geom: CupGeometry, v: float = 0.5,
                         n: int = 720) -> float:
    """Surface azimuth (degrees) whose outward normal points most directly
    at the camera for this pose.

    The cup frame's theta = 0 is an arbitrary reference fixed by the axis
    (see pose_estimation3d._basis_from_axis) and bears no relation to where
    the camera happens to be, so placing the logo at theta = 0 by default
    puts it on whichever side the maths happened to pick -- in the first
    run, hidden round the back. This gives a meaningful default: the patch
    of ceramic actually facing the viewer.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = geom.surface_point(th, np.full_like(th, v))
    nrm = geom.surface_normal(th, np.full_like(th, v))
    pc = pts @ pose.R.T + pose.tvec
    nc = nrm @ pose.R.T
    d = pc / (np.linalg.norm(pc, axis=1, keepdims=True) + 1e-12)
    facing = -np.einsum("ij,ij->i", nc, d)      # 1 when normal faces camera
    return float(np.rad2deg(th[int(np.argmax(facing))]))


@dataclass
class ARResult:
    poses: list[Pose | None]
    azimuths: np.ndarray
    reproj_errors: np.ndarray
    n_tracked: int


def estimate_sequence(frames: list[np.ndarray], cfg: ARConfig,
                       verbose: bool = True) -> ARResult:
    """Ellipse tracking -> pose -> azimuth -> smoothing, for a whole clip."""
    H, W = frames[0].shape[:2]
    intr = Intrinsics(cfg.focal, cfg.focal, W / 2.0, H / 2.0, W, H)
    geom = cfg.geometry

    if verbose:
        print("  tracking disc ellipse...")
    tracked = track_sequence(frames, verbose=verbose)

    if verbose:
        print("  solving 6-DoF pose per frame...")
    raw: list[Pose | None] = [None] * len(frames)
    errs = np.full(len(frames), np.nan)
    prev = None
    for i, tr in enumerate(tracked):
        if tr is None:
            continue
        cands = [c for c in cup_pose_candidates(tr.obs, intr, geom, 0.0)
                 if c.physically_valid]
        if not cands:
            continue
        if prev is not None and len(cands) > 1:
            cands.sort(key=lambda c: np.linalg.norm(
                cv2.Rodrigues(c.pose.R @ prev.R.T)[0]))
        raw[i] = cands[0].pose
        errs[i] = reprojection_error(raw[i], tr.obs, intr, geom)
        prev = raw[i]

    n_tracked = sum(p is not None for p in raw)
    if verbose:
        ok = np.isfinite(errs)
        print(f"  poses on {n_tracked}/{len(frames)} frames, "
              f"median disc reprojection error "
              f"{np.median(errs[ok]) if ok.any() else float('nan'):.3f} px")

    if verbose:
        print("  recovering azimuth from surface texture...")
    anchor = next((i for i, p in enumerate(raw) if p is not None), None)
    az = np.zeros(len(frames))
    if anchor is not None:
        ref = reference_profile(frames[anchor], raw[anchor], geom, intr)
        for i, p in enumerate(raw):
            if p is None:
                continue
            a, _score, _ = estimate_azimuth(frames[i], p, geom, intr, ref)
            az[i] = a
        idx = [i for i, p in enumerate(raw) if p is not None]
        az[idx] = _unwrap_series(az[idx])

    # rebuild the poses with the recovered azimuth, then smooth
    final: list[Pose | None] = [None] * len(frames)
    prev = None
    for i, tr in enumerate(tracked):
        if tr is None or raw[i] is None:
            continue
        cands = [c for c in cup_pose_candidates(tr.obs, intr, geom, az[i])
                 if c.physically_valid]
        if not cands:
            continue
        if prev is not None and len(cands) > 1:
            cands.sort(key=lambda c: np.linalg.norm(
                cv2.Rodrigues(c.pose.R @ prev.R.T)[0]))
        final[i] = cands[0].pose
        prev = final[i]

    return ARResult(smooth_poses(final, cfg.pose_smoothing), az, errs, n_tracked)


def render_frame(frame_bgr: np.ndarray, pose: Pose, mesh: Mesh, atlas: np.ndarray,
                  face_subset: np.ndarray, intr: Intrinsics, cfg: ARConfig
                  ) -> np.ndarray:
    fb = render_mesh(mesh, pose, intr, face_subset=face_subset)
    return composite_print(frame_bgr, fb, atlas, intr, cfg)


def process_video(in_path: str, out_path: str, logo_path: str,
                   cfg: ARConfig | None = None, max_frames: int | None = None,
                   progress_cb=None, verbose: bool = True) -> dict:
    """End-to-end: read a video, track/solve, render, write."""
    cfg = cfg or ARConfig()

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded")

    H, W = frames[0].shape[:2]
    intr = Intrinsics(cfg.focal, cfg.focal, W / 2.0, H / 2.0, W, H)

    res = estimate_sequence(frames, cfg, verbose=verbose)

    mesh = build_cup_mesh(cfg.geometry, cfg.mesh_theta, cfg.mesh_v, cap_top=True)
    logo = load_logo_rgba(logo_path)

    placement = cfg.placement
    if placement.theta_center_deg is None:
        anchor = next((p for p in res.poses if p is not None), None)
        auto = theta_facing_camera(anchor, cfg.geometry, placement.v_center) \
            if anchor is not None else 0.0
        placement = LogoPlacement(auto, placement.v_center,
                                  placement.height_frac, placement.opacity)
        if verbose:
            print(f"  logo azimuth auto-set to {auto:.1f} deg (camera-facing "
                  f"at the anchor frame)")
    atlas = build_atlas(logo, cfg.geometry, placement)
    subset = face_has_texture(atlas, mesh)
    if verbose:
        print(f"  mesh {mesh.n_faces} faces, {int(subset.sum())} carry the logo")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (W, H))
    drawn = 0
    for i, f in enumerate(frames):
        p = res.poses[i]
        out = render_frame(f, p, mesh, atlas, subset, intr, cfg) if p is not None else f
        if p is not None:
            drawn += 1
        writer.write(out)
        if progress_cb:
            progress_cb(i + 1, len(frames))
    writer.release()

    ok = np.isfinite(res.reproj_errors)
    return {
        "frames": len(frames),
        "frames_with_logo": drawn,
        "poses_solved": res.n_tracked,
        "median_reproj_px": float(np.median(res.reproj_errors[ok])) if ok.any() else None,
    }
