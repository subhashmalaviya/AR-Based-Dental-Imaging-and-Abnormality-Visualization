"""
model_fit3d.py
---------------
Fits the shared, time-invariant unknowns of the scene:

    f              camera focal length (px)
    height/r_top   cup aspect
    r_bottom/r_top cup taper
    r_disc/r_top   how far the detected base disc is inset from the wall

Objective: **edge alignment**, not region overlap. Earlier versions scored
a projected silhouette against a GrabCut mask, and that objective was too
unreliable to optimise -- the segmentation variously swallowed the handle,
leaked onto the table, or missed half the cup, and the fit consequently
pinned every parameter to a grid bound at a mean IoU of only 0.36.

Here the model's outline (the two body end-circles and the base disc) is
projected and scored by how much image gradient sits underneath it. Object
outlines *are* intensity edges, so this needs no segmentation at all, and
it is the standard objective for model-based 3D tracking. Points whose
projected normal disagrees with the local gradient direction are down-
weighted, which stops the score being farmed by unrelated clutter such as
the table edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera3d import Intrinsics, Pose, project_points
from .cup_model_3d import CupGeometry, outline_points
from .ellipse_detection import EllipseObs
from .pose_estimation3d import cup_pose_candidates


@dataclass
class SceneModel:
    focal: float
    height_ratio: float        # height / r_top
    r_bottom_ratio: float      # r_bottom / r_top
    r_disc_ratio: float        # r_disc / r_top
    score: float

    def geometry(self, r_top: float = 0.041) -> CupGeometry:
        return CupGeometry(r_top=r_top,
                           r_bottom=r_top * self.r_bottom_ratio,
                           height=r_top * self.height_ratio,
                           r_disc_ratio=self.r_disc_ratio)

    def intrinsics(self, width: int, height: int) -> Intrinsics:
        return Intrinsics(self.focal, self.focal, width / 2.0, height / 2.0,
                          width, height)


def gradient_field(frame_bgr: np.ndarray, blur: int = 5):
    """Image gradient magnitude and direction, used as the edge evidence."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (blur, blur), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (np.percentile(mag, 99.0) + 1e-6)
    return np.clip(mag, 0, 1), gx, gy


def _sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return img[y, x]


def edge_score(pose: Pose, geom: CupGeometry, intr: Intrinsics,
                mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> float:
    """Mean gradient support along the model's projected outline.

    Only the *silhouette* half of each end-circle is scored (the half whose
    surface normal turns away from the camera is hidden behind the cup and
    has no business finding an edge), and each sample is weighted by how
    well the image gradient direction matches the outline's normal.
    """
    parts = outline_points(geom)
    total, count = 0.0, 0

    for name, pts3d in parts.items():
        uv, z = project_points(pts3d, pose, intr)
        if np.any(z <= 0):
            return 0.0
        inside = ((uv[:, 0] >= 1) & (uv[:, 0] < intr.width - 1) &
                  (uv[:, 1] >= 1) & (uv[:, 1] < intr.height - 1))
        if inside.sum() < 8:
            continue
        uvi = uv[inside]

        # outline tangent -> normal, in image space
        d = np.gradient(uvi, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 1e-6)

        m = _sample(mag, uvi)
        gxs = _sample(gx, uvi)
        gys = _sample(gy, uvi)
        gl = np.hypot(gxs, gys) + 1e-6
        align = np.abs((gxs * nrm[:, 0] + gys * nrm[:, 1]) / gl)

        total += float(np.sum(m * align))
        count += len(uvi)

    return total / max(count, 1)


def fit_scene(frames: list[np.ndarray], ellipses, keyframes: list[int],
               f_range=(900.0, 3600.0), verbose: bool = True) -> SceneModel | None:
    """Coarse-to-fine search for the shared camera/cup parameters."""
    if not keyframes:
        return None
    H, W = frames[keyframes[0]].shape[:2]
    fields = {i: gradient_field(frames[i]) for i in keyframes}
    obs = {i: ellipses[i].obs for i in keyframes}

    def score(f, hr, rbr, rdr) -> float:
        geom = CupGeometry(0.041, 0.041 * rbr, 0.041 * hr, rdr)
        intr = Intrinsics(f, f, W / 2.0, H / 2.0, W, H)
        tot = 0.0
        for i in keyframes:
            mag, gx, gy = fields[i]
            best = 0.0
            for c in cup_pose_candidates(obs[i], intr, geom, 0.0):
                if not c.body_below:
                    continue
                best = max(best, edge_score(c.pose, geom, intr, mag, gx, gy))
            tot += best
        return tot / len(keyframes)

    best = (-1.0, None)
    for f in np.linspace(*f_range, 10):
        for hr in np.linspace(1.8, 4.2, 9):
            for rbr in np.linspace(0.85, 1.25, 5):
                for rdr in np.linspace(0.70, 1.0, 4):
                    s = score(f, hr, rbr, rdr)
                    if s > best[0]:
                        best = (s, (f, hr, rbr, rdr))
    if best[1] is None:
        return None
    if verbose:
        f, hr, rbr, rdr = best[1]
        print(f"  coarse: f={f:.0f} h/r={hr:.2f} rb/rt={rbr:.2f} rd/rt={rdr:.2f}"
              f" score={best[0]:.4f}")

    # local refinement around the coarse optimum
    f, hr, rbr, rdr = best[1]
    for _ in range(3):
        improved = False
        for df, dh, db, dd in [(0.06, 0, 0, 0), (-0.06, 0, 0, 0),
                                (0, 0.12, 0, 0), (0, -0.12, 0, 0),
                                (0, 0, 0.03, 0), (0, 0, -0.03, 0),
                                (0, 0, 0, 0.03), (0, 0, 0, -0.03)]:
            cand = (f * (1 + df), hr + dh, rbr + db, rdr + dd)
            if not (0.6 <= cand[3] <= 1.05 and 1.2 <= cand[1] <= 5.0):
                continue
            s = score(*cand)
            if s > best[0]:
                best = (s, cand)
                f, hr, rbr, rdr = cand
                improved = True
        if not improved:
            break

    if verbose:
        print(f"  refined: f={f:.0f} h/r={hr:.2f} rb/rt={rbr:.2f} rd/rt={rdr:.2f}"
              f" score={best[0]:.4f}")
    return SceneModel(float(f), float(hr), float(rbr), float(rdr), float(best[0]))
