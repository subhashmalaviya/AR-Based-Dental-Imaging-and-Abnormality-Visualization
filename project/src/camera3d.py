"""
camera3d.py
------------
Pinhole camera model: intrinsics, extrinsics, and the 3D -> 2D projection
used by every other 3D module.

Conventions (OpenCV):
  * Camera looks down its own +Z axis; +X right, +Y down in the image.
  * A world point X_w maps to camera coordinates as  X_c = R @ X_w + t.
  * Projection:  u = fx * Xc/Zc + cx,   v = fy * Yc/Zc + cy.

`rvec` is a Rodrigues rotation vector (as produced/consumed by
cv2.Rodrigues and cv2.solvePnP), so poses interchange freely with OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 67.0) -> "Intrinsics":
        """Sensible default for an uncalibrated phone camera.

        A typical smartphone main camera has a horizontal field of view
        around 65-70 degrees. This is only a *prior*: `calibrate3d.py`
        refines fx/fy from the video itself. Documented as an assumption
        rather than a measurement.
        """
        f = 0.5 * width / np.tan(0.5 * np.deg2rad(hfov_deg))
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics for an image resized by factor s (used to run pose
        estimation on downscaled frames without changing the geometry)."""
        return Intrinsics(self.fx * s, self.fy * s, self.cx * s, self.cy * s,
                          int(round(self.width * s)), int(round(self.height * s)))

    def with_focal(self, f: float) -> "Intrinsics":
        return Intrinsics(f, f, self.cx, self.cy, self.width, self.height)


@dataclass
class Pose:
    """Rigid transform world -> camera:  X_c = R @ X_w + t."""
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec.reshape(3, 1))[0]

    @classmethod
    def from_Rt(cls, R: np.ndarray, t: np.ndarray) -> "Pose":
        rvec = cv2.Rodrigues(R)[0].reshape(3)
        return cls(rvec=rvec, tvec=np.asarray(t, dtype=np.float64).reshape(3))

    def camera_center_world(self) -> np.ndarray:
        """Position of the camera expressed in world coordinates."""
        return -self.R.T @ self.tvec

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.rvec, self.tvec])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Pose":
        v = np.asarray(v, dtype=np.float64).reshape(6)
        return cls(rvec=v[:3].copy(), tvec=v[3:].copy())


def transform_points(points_w: np.ndarray, pose: Pose) -> np.ndarray:
    """World -> camera coordinates. points_w: (N,3) -> (N,3)."""
    return points_w @ pose.R.T + pose.tvec


def project_points(points_w: np.ndarray, pose: Pose, intr: Intrinsics
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Full 3D -> 2D pipeline.

    Returns (uv, z_cam): uv is (N,2) pixel coordinates, z_cam is (N,)
    camera-space depth (needed for z-buffering and for rejecting points
    behind the camera).
    """
    pc = transform_points(points_w, pose)
    z = pc[:, 2]
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = intr.fx * pc[:, 0] / z_safe + intr.cx
    v = intr.fy * pc[:, 1] / z_safe + intr.cy
    return np.stack([u, v], axis=1), z


def look_at(eye: np.ndarray, target: np.ndarray,
             up_world: np.ndarray = np.array([0.0, 1.0, 0.0])) -> Pose:
    """Build a world->camera Pose for a camera at `eye` looking at `target`.

    Used by the standalone demo to orbit a virtual camera around the cup.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)

    forward = target - eye
    forward /= np.linalg.norm(forward)          # camera +Z in world

    right = np.cross(forward, up_world)
    n = np.linalg.norm(right)
    if n < 1e-8:                                 # degenerate: looking along `up`
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
    right /= n                                   # camera +X in world

    down = np.cross(forward, right)              # camera +Y (image y points down)

    R = np.stack([right, down, forward], axis=0)  # rows = camera axes in world
    t = -R @ eye
    return Pose.from_Rt(R, t)
