"""
pose_estimation3d.py
---------------------
Turns a detected disc ellipse into a full 6-DoF pose of the cup in the
camera frame.

  ellipse  --(closed form, pose_from_ellipse.py)-->  circle centre C and
  plane normal n in camera coordinates
           --(this module)-->  cup pose (R, t)

The circle gives 5 DoF. The 6th -- rotation of the cup about its own
axis (azimuth) -- is genuinely unobservable from a surface of revolution's
silhouette, because rotating a cylinder about its axis changes nothing in
the image. It is recovered separately, from the cup's own printed texture,
in `azimuth.py`; this module simply accepts an azimuth argument so the
caller can supply it.

Handedness/orientation conventions match cup_model_3d.py: the cup frame's
origin sits at the centre of the circle resting on the table, +Y runs up
the axis, and the detected disc is the circle at Y = height.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera3d import Intrinsics, Pose, project_points
from .cup_model_3d import CupGeometry, silhouette_points, top_circle_points
from .ellipse_detection import EllipseObs
from .pose_from_ellipse import CirclePose, circle_pose_candidates, ellipse_to_conic


def _basis_from_axis(y_axis: np.ndarray, azimuth_rad: float) -> np.ndarray:
    """Right-handed rotation whose second column is `y_axis`.

    `azimuth_rad` selects which perpendicular direction becomes the cup's
    +Z (theta = 0) reference, i.e. it is the free rotation about the axis.
    The reference perpendicular is chosen deterministically from the axis
    itself so the same axis always yields the same zero of azimuth.
    """
    y = y_axis / np.linalg.norm(y_axis)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, y)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    r0 = np.cross(y, seed)
    r0 /= np.linalg.norm(r0)
    b = np.cross(y, r0)

    z_axis = np.cos(azimuth_rad) * r0 + np.sin(azimuth_rad) * b
    x_axis = np.cross(y, z_axis)
    return np.stack([x_axis, y, z_axis], axis=1)     # columns = cup axes


def pose_from_circle(cp: CirclePose, geom: CupGeometry, azimuth_rad: float,
                      axis_sign: float) -> Pose:
    """Build the cup pose from a recovered disc circle.

    `axis_sign` picks whether the cup body extends along +n or -n from the
    disc; both are geometrically consistent with the same ellipse.
    """
    y_axis = axis_sign * cp.normal / np.linalg.norm(cp.normal)
    R = _basis_from_axis(y_axis, azimuth_rad)
    t = cp.center - geom.height * y_axis          # origin = table-side circle
    return Pose.from_Rt(R, t)


@dataclass
class PoseCandidate:
    pose: Pose
    circle: CirclePose
    axis_sign: float
    body_below: bool          # base projects below the disc in the image
    base_farther: bool        # base is farther from the camera than the disc
    silhouette_iou: float = -1.0

    @property
    def physically_valid(self) -> bool:
        """Both conditions must hold for the real scene.

        The camera sees the mug's top face in every frame, so it is above
        the disc's plane; therefore the base (resting on the table) must be
        both lower in the image *and* farther away. Each condition alone is
        satisfied by a wrong branch as well -- of the four candidates for
        frame 0, one is below-but-nearer and another is farther-but-above --
        so testing only one of them silently selects a mirrored pose.
        """
        return self.body_below and self.base_farther


def cup_pose_candidates(obs: EllipseObs, intr: Intrinsics, geom: CupGeometry,
                         azimuth_rad: float = 0.0) -> list[PoseCandidate]:
    """All poses consistent with the observed ellipse (typically 2-4)."""
    Q = ellipse_to_conic(obs.center, obs.axes, obs.angle)
    circles = circle_pose_candidates(Q, intr.K, geom.r_disc)

    out: list[PoseCandidate] = []
    for cp in circles:
        for sign in (+1.0, -1.0):
            pose = pose_from_circle(cp, geom, azimuth_rad, sign)
            origin_uv, z0 = project_points(np.zeros((1, 3)), pose, intr)
            disc_uv, zd = project_points(
                np.array([[0.0, geom.height, 0.0]]), pose, intr)
            if z0[0] <= 0 or zd[0] <= 0:
                continue
            below = bool(origin_uv[0, 1] > disc_uv[0, 1])
            farther = bool(z0[0] > zd[0])
            out.append(PoseCandidate(pose, cp, sign, below, farther))
    return out


def render_silhouette(pose: Pose, geom: CupGeometry, intr: Intrinsics,
                       n_theta: int = 48) -> np.ndarray:
    """Filled convex silhouette of the cup model, as a uint8 mask.

    A surface of revolution's outline is the convex hull of its two end
    circles under this viewpoint range, which is far cheaper than
    rasterising the mesh and is all the shape fitter needs.
    """
    pts3d = silhouette_points(geom, n=n_theta)
    uv, z = project_points(pts3d, pose, intr)
    mask = np.zeros((intr.height, intr.width), np.uint8)
    if np.any(z <= 0):
        return mask
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


def silhouette_iou(pose: Pose, geom: CupGeometry, intr: Intrinsics,
                    target_mask: np.ndarray) -> float:
    m = render_silhouette(pose, geom, intr)
    inter = np.count_nonzero(m & target_mask)
    union = np.count_nonzero(m | target_mask)
    return inter / union if union else 0.0


def select_pose(obs: EllipseObs, intr: Intrinsics, geom: CupGeometry,
                 azimuth_rad: float = 0.0,
                 body_mask: np.ndarray | None = None,
                 prev_pose: Pose | None = None) -> PoseCandidate | None:
    """Resolve the pose ambiguity and return the best candidate.

    Priority: (1) the body must project below the disc, (2) if a cup
    segmentation is available, agreement with it, (3) continuity with the
    previous frame's pose.
    """
    cands = cup_pose_candidates(obs, intr, geom, azimuth_rad)
    if not cands:
        return None

    valid = [c for c in cands if c.physically_valid]
    pool = valid or [c for c in cands if c.body_below] or cands

    if body_mask is not None:
        for c in pool:
            c.silhouette_iou = silhouette_iou(c.pose, geom, intr, body_mask)
        pool.sort(key=lambda c: -c.silhouette_iou)
        if pool[0].silhouette_iou > 0.0:
            return pool[0]

    if prev_pose is not None:
        def dist(c: PoseCandidate) -> float:
            dt = np.linalg.norm(c.pose.tvec - prev_pose.tvec)
            dR = np.linalg.norm(cv2.Rodrigues(c.pose.R @ prev_pose.R.T)[0])
            return dt / max(np.linalg.norm(prev_pose.tvec), 1e-6) + dR
        pool.sort(key=dist)
    return pool[0]


def reprojection_error(pose: Pose, obs: EllipseObs, intr: Intrinsics,
                        geom: CupGeometry, n: int = 128) -> float:
    """Mean pixel distance between the projected disc circle and the
    observed ellipse -- the residual of the pose fit itself."""
    uv, z = project_points(top_circle_points(geom, n), pose, intr)
    if np.any(z <= 0):
        return float("inf")
    (xc, yc), (MA, ma), ang = obs.as_cv()
    a, b = MA / 2.0, ma / 2.0
    p = np.deg2rad(ang)
    c, s = np.cos(p), np.sin(p)
    dx = uv[:, 0] - xc
    dy = uv[:, 1] - yc
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    scale = 0.5 * (a + b)
    return float(np.mean(np.abs(r - 1.0)) * scale)
