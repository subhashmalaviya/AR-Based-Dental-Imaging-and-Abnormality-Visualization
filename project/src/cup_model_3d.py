"""
cup_model_3d.py
----------------
The 3D cup: a parametric surface of revolution (a slightly tapered
frustum) turned into a triangle mesh with per-vertex UV coordinates and
outward normals.

Cup coordinate frame
--------------------
  * Origin at the centre of the circle that rests on the table.
  * +Y is the cup's axis, pointing up out of the table.
  * The other circular end (the disc facing the camera in this footage,
    since the mug is stood upside-down) is at Y = height.

Surface parametrisation, exactly the form requested:

    R(v) = r_bottom + v * (r_top - r_bottom)          v in [0, 1]
    X    = R(v) * sin(theta)
    Y    = v * height
    Z    = R(v) * cos(theta)                          theta in [0, 2pi)

theta = 0 points along +Z, and is the cup's *own* azimuth reference: the
logo is pinned to a fixed theta on this surface, so it is attached to the
cup in 3D and never positioned in 2D.

Analytic outward normal (derived from dP/dtheta x dP/dv):

    n(theta, v) ∝ ( height * sin(theta), -(r_top - r_bottom), height * cos(theta) )

which reduces to the radial direction for a straight cylinder and tilts
correctly for a tapered one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CupGeometry:
    """Metric shape of the cup.

    `r_top` is the scale gauge: monocular video cannot recover absolute
    size (a cup twice as large at twice the distance is pixel-identical),
    so r_top is fixed and everything else is a dimensionless ratio.

    `r_disc_ratio` matters and is easy to miss. The mug is filmed standing
    upside-down, so the bright circle we detect is its *recessed base*,
    which is inset from the outer wall -- not the body radius. Measured on
    frame 0: the disc ellipse gives a 49.9 deg tilt (cos = 210/326 from its
    semi-axes), and combined with the cup's observed projected length that
    forces height/r_disc ~ 3.0. Treating the disc as the full body radius
    instead makes the model far too short for its width, which is exactly
    why an earlier silhouette fit pinned every parameter to its bound.
    """
    r_top: float = 0.041          # body radius at the disc end (m)
    r_bottom: float = 0.041       # body radius at the table end (m)
    height: float = 0.105         # along-axis extent (m)
    r_disc_ratio: float = 0.85    # detected base disc radius / r_top

    @property
    def r_disc(self) -> float:
        """Radius of the circle actually observed as an ellipse."""
        return self.r_top * self.r_disc_ratio

    def radius_at(self, v):
        return self.r_bottom + np.asarray(v) * (self.r_top - self.r_bottom)

    def surface_point(self, theta, v) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        r = self.radius_at(v)
        return np.stack([r * np.sin(theta), v * self.height, r * np.cos(theta)], axis=-1)

    def surface_normal(self, theta, v=None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        k = self.r_top - self.r_bottom
        n = np.stack([
            self.height * np.sin(theta),
            np.full_like(theta, -k),
            self.height * np.cos(theta),
        ], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


@dataclass
class Mesh:
    vertices: np.ndarray       # (N,3) float64, cup coordinates
    faces: np.ndarray          # (F,3) int32, CCW seen from outside
    uvs: np.ndarray            # (N,2) float32, u = theta/2pi, v = height frac
    normals: np.ndarray        # (N,3) float64, outward unit normals
    face_is_wall: np.ndarray   # (F,) bool -- wall (textured) vs. end cap

    @property
    def n_faces(self) -> int:
        return len(self.faces)


def build_cup_mesh(geom: CupGeometry, n_theta: int = 128, n_v: int = 32,
                    cap_top: bool = True) -> Mesh:
    """Tessellate the cup into triangles.

    The theta seam is duplicated (columns 0 and n_theta coincide in space
    but carry u=0 and u=1) so UV interpolation never wraps across the
    texture, which would otherwise smear the atlas horizontally.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)   # inclusive -> seam dup
    vs = np.linspace(0.0, 1.0, n_v + 1)

    TH, VV = np.meshgrid(thetas, vs, indexing="xy")        # (n_v+1, n_theta+1)
    verts = geom.surface_point(TH, VV).reshape(-1, 3)
    norms = geom.surface_normal(TH, VV).reshape(-1, 3)
    uvs = np.stack([TH / (2.0 * np.pi), VV], axis=-1).reshape(-1, 2).astype(np.float32)

    cols = n_theta + 1
    faces = []
    for j in range(n_v):           # along the axis
        for i in range(n_theta):   # around the axis
            a = j * cols + i
            b = j * cols + (i + 1)
            c = (j + 1) * cols + (i + 1)
            d = (j + 1) * cols + i
            # CCW when viewed from outside the surface
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int32)
    face_is_wall = np.ones(len(faces), dtype=bool)

    if cap_top:
        # Flat disc closing the top (the mug's base, which faces the camera).
        centre_idx = len(verts)
        verts = np.vstack([verts, np.array([[0.0, geom.height, 0.0]])])
        norms = np.vstack([norms, np.array([[0.0, 1.0, 0.0]])])
        uvs = np.vstack([uvs, np.array([[0.0, 1.0]], dtype=np.float32)])

        ring0 = n_v * cols
        cap_faces = []
        for i in range(n_theta):
            a = ring0 + i
            b = ring0 + i + 1
            cap_faces.append((centre_idx, b, a))
        cap_faces = np.asarray(cap_faces, dtype=np.int32)
        faces = np.vstack([faces, cap_faces])
        face_is_wall = np.concatenate([face_is_wall, np.zeros(len(cap_faces), bool)])

        # The rim vertices are shared between wall and cap; give the cap its
        # own averaged normal only at the centre so wall shading stays exact.
        norms[centre_idx] = np.array([0.0, 1.0, 0.0])

    return Mesh(vertices=verts, faces=faces, uvs=uvs,
                normals=norms, face_is_wall=face_is_wall)


def silhouette_points(geom: CupGeometry, n: int = 64) -> np.ndarray:
    """3D points on the two bounding circles (table circle and top disc).

    Used by the calibration/pose fitters, which compare the projection of
    these circles against edges detected in the image.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    bottom = geom.surface_point(th, np.zeros_like(th))
    top = geom.surface_point(th, np.ones_like(th))
    return np.concatenate([bottom, top], axis=0)


def top_circle_points(geom: CupGeometry, n: int = 128) -> np.ndarray:
    """3D points of the *detected* base disc: a circle of radius `r_disc`
    in the plane Y = height (inset from the body wall, see CupGeometry)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = geom.r_disc
    return np.stack([r * np.sin(th), np.full_like(th, geom.height),
                     r * np.cos(th)], axis=-1)


def outline_points(geom: CupGeometry, n_theta: int = 96, n_v: int = 12
                    ) -> dict[str, np.ndarray]:
    """3D points on the features that produce strong image edges: the two
    body end-circles and the detected base disc. Used by the edge-alignment
    model fitter, which needs no segmentation."""
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return {
        "bottom": geom.surface_point(th, np.zeros_like(th)),
        "top": geom.surface_point(th, np.ones_like(th)),
        "disc": top_circle_points(geom, n_theta),
    }
