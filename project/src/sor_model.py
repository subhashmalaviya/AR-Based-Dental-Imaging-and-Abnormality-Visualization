"""
sor_model.py
-------------
General surface of revolution from an arbitrary radius profile r(v).

`cup_model_3d.CupGeometry` is a linear frustum, which is all a mug needs.
Testing on tv2.mp4 (a Milton steel bottle) showed why that is too narrow:
the object is still a surface of revolution, but its profile has a rounded
base, a ridged band, a straight body and a tapering shoulder. This module
takes the profile as data instead of hard-coding a straight taper, and
emits the same `Mesh` the rest of the pipeline already consumes, so the
renderer, texture atlas, compositor and azimuth stages need no changes.

The same generalisation is what a dental model needs: replace the profile
with a scanned mesh and everything downstream is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cup_model_3d import Mesh


@dataclass
class ProfileGeometry:
    """Surface of revolution defined by a sampled radius profile.

    `profile_v` and `profile_r` are matched 1-D arrays: the radius (metres)
    at each normalised height v in [0, 1]. Radii are linearly interpolated
    between samples.
    """
    profile_v: np.ndarray
    profile_r: np.ndarray
    height: float

    @classmethod
    def from_callable(cls, f: Callable[[np.ndarray], np.ndarray], height: float,
                       n: int = 129) -> "ProfileGeometry":
        v = np.linspace(0.0, 1.0, n)
        return cls(v, np.asarray(f(v), dtype=np.float64), height)

    def radius_at(self, v):
        return np.interp(np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0),
                         self.profile_v, self.profile_r)

    def dr_dv(self, v, eps: float = 1e-4):
        v = np.asarray(v, dtype=np.float64)
        return (self.radius_at(np.clip(v + eps, 0, 1))
                - self.radius_at(np.clip(v - eps, 0, 1))) / (2 * eps)

    def surface_point(self, theta, v) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        r = self.radius_at(v)
        return np.stack([r * np.sin(theta), v * self.height, r * np.cos(theta)],
                        axis=-1)

    def surface_normal(self, theta, v) -> np.ndarray:
        """Outward normal of the revolved profile.

        dP/dtheta x dP/dv gives (H sin, -dr/dv, H cos) up to scale, the same
        form as the frustum case but with the local profile slope in place
        of a constant taper.
        """
        theta = np.asarray(theta, dtype=np.float64)
        k = self.dr_dv(v)
        n = np.stack([self.height * np.sin(theta),
                      -k * np.ones_like(theta),
                      self.height * np.cos(theta)], axis=-1)
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def build_sor_mesh(geom: ProfileGeometry, n_theta: int = 128, n_v: int = 64,
                    cap_top: bool = False, cap_bottom: bool = False) -> Mesh:
    """Tessellate the surface of revolution, with a duplicated theta seam
    so UV interpolation never wraps across the atlas."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    vs = np.linspace(0.0, 1.0, n_v + 1)
    TH, VV = np.meshgrid(thetas, vs, indexing="xy")

    verts = geom.surface_point(TH, VV).reshape(-1, 3)
    norms = geom.surface_normal(TH, VV).reshape(-1, 3)
    uvs = np.stack([TH / (2 * np.pi), VV], axis=-1).reshape(-1, 2).astype(np.float32)

    cols = n_theta + 1
    faces = []
    for j in range(n_v):
        for i in range(n_theta):
            a = j * cols + i
            b = j * cols + (i + 1)
            c = (j + 1) * cols + (i + 1)
            d = (j + 1) * cols + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int32)
    is_wall = np.ones(len(faces), bool)

    def add_cap(ring_start: int, y: float, up: bool):
        nonlocal verts, norms, uvs, faces, is_wall
        ci = len(verts)
        verts = np.vstack([verts, [[0.0, y, 0.0]]])
        norms = np.vstack([norms, [[0.0, 1.0 if up else -1.0, 0.0]]])
        uvs = np.vstack([uvs, np.array([[0.0, 1.0 if up else 0.0]], np.float32)])
        cf = []
        for i in range(n_theta):
            a, b = ring_start + i, ring_start + i + 1
            cf.append((ci, b, a) if up else (ci, a, b))
        cf = np.asarray(cf, np.int32)
        faces = np.vstack([faces, cf])
        is_wall = np.concatenate([is_wall, np.zeros(len(cf), bool)])

    if cap_top:
        add_cap(n_v * cols, geom.height, True)
    if cap_bottom:
        add_cap(0, 0.0, False)

    return Mesh(vertices=verts, faces=faces, uvs=uvs, normals=norms,
                face_is_wall=is_wall)


def milton_bottle_profile(radius: float = 0.035, height: float = 0.26
                           ) -> ProfileGeometry:
    """Approximate profile of the steel bottle in tv2.mp4, read off its
    silhouette: rounded base, ridged band, straight body, tapering shoulder."""
    v = np.array([0.00, 0.03, 0.07, 0.12, 0.30, 0.55, 0.70, 0.82, 0.92, 1.00])
    r = np.array([0.55, 0.86, 0.97, 1.00, 1.00, 0.99, 0.93, 0.78, 0.60, 0.52])
    return ProfileGeometry(v, r * radius, height)
