"""
pose_from_ellipse.py
---------------------
Closed-form 3D pose of a circle from its image ellipse ("circle pose from
conic"). This is what makes the pipeline genuinely 3D: the cup's end disc
is a real circle in space, its image is an ellipse, and from that ellipse
plus the camera intrinsics we recover the circle's 3D centre and normal --
i.e. 5 of the cup's 6 pose DoF -- with no homography and no planarity
assumption about the logo.

Derivation (implemented below, verified numerically in `self_test`)
--------------------------------------------------------------------
An image ellipse back-projects to a cone of rays through the camera
centre. In normalised camera coordinates that cone is

    X^T Q X = 0,        Q = K^T Q_img K

Diagonalise Q = V diag(l1,l2,l3) V^T, ordered so that l1 >= l2 > 0 > l3
(negate Q if needed, so the signature is (2,1) as it must be for a real
cone). Work in the eigenbasis.

Which planes cut this cone in a *circle*? Subtracting l2*(x^2+y^2+z^2)
from the cone equation gives

    (l1-l2) x^2 - (l2-l3) z^2 = 0

so on the two planes  sqrt(l1-l2) x  =  +/- sqrt(l2-l3) z  every cone
point also satisfies x^2+y^2+z^2 = const: the sections are circles. Hence
the circular-section normal is, in the eigenbasis,

    n = (g, 0, s*h),   g = sqrt((l1-l2)/(l1-l3)),  h = sqrt((l2-l3)/(l1-l3))

with g^2 + h^2 = 1 and a sign s = +/-1.

For a parallel plane n.X = d, writing X = d*n + s0*u + t0*w with
u = (-s*h, 0, g) and w = (0,1,0), the cone equation becomes a circle in
(s0, t0) with

    t0 = 0,   s0 = d * s * g*h*(l1-l3) / l2
    radius^2 = -d^2 * l1*l3 / l2^2

so a circle of known radius R fixes the plane distance:

    |d| = R * l2 / sqrt(-l1*l3)

and the centre, in the eigenbasis, is

    C = d * ( g - k*h,  0,  s*(h + k*g) ),   k = sqrt((l1-l2)(l2-l3)) / l2

Mapping back through V gives the camera-frame normal and centre. The signs
(s, sign of d) leave the classical two-fold ambiguity after discarding
solutions behind the camera; the caller disambiguates using the cup body
(see `pose_tracking3d.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CirclePose:
    center: np.ndarray   # (3,) circle centre in camera coordinates
    normal: np.ndarray   # (3,) unit normal of the circle's supporting plane
    radius: float


def ellipse_to_conic(center, axes, angle_deg: float) -> np.ndarray:
    """cv2.fitEllipse output -> 3x3 symmetric conic matrix in image coords.

    `axes` are the *full* axis lengths, as OpenCV returns them.
    A point x=(u,v,1) lies on the ellipse iff x^T Q x = 0.
    """
    xc, yc = float(center[0]), float(center[1])
    a = float(axes[0]) / 2.0
    b = float(axes[1]) / 2.0
    phi = np.deg2rad(float(angle_deg))
    c, s = np.cos(phi), np.sin(phi)

    # M maps image homogeneous coords into the ellipse's canonical frame.
    M = np.array([
        [c,  s, -(xc * c + yc * s)],
        [-s, c,  (xc * s - yc * c)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    D = np.diag([1.0 / (a * a), 1.0 / (b * b), -1.0])
    Q = M.T @ D @ M
    return 0.5 * (Q + Q.T)


def _normalise_cone(Q: np.ndarray):
    """Eigendecompose the cone and order eigenvalues as l1 >= l2 > 0 > l3."""
    w, V = np.linalg.eigh(Q)
    # Ensure signature (2,1): two positive, one negative.
    if np.sum(w > 0) == 1:
        w, Q = -w, -Q
        # eigenvectors unchanged by negation
    if np.sum(w > 0) != 2 or np.sum(w < 0) != 1:
        return None

    neg = int(np.where(w < 0)[0][0])
    pos = [i for i in range(3) if i != neg]
    # among the positives, l1 is the larger
    pos.sort(key=lambda i: -w[i])
    order = [pos[0], pos[1], neg]
    lam = w[order]
    Vo = V[:, order]
    # keep a right-handed basis so cross products behave predictably
    if np.linalg.det(Vo) < 0:
        Vo[:, 2] *= -1.0
    return lam, Vo


def circle_pose_candidates(Q_img: np.ndarray, K: np.ndarray,
                            radius: float) -> list[CirclePose]:
    """All physically-admissible 3D circle poses consistent with the ellipse.

    Returns 0-2 candidates (solutions behind the camera are discarded).
    """
    Q = K.T @ Q_img @ K
    Q = 0.5 * (Q + Q.T)
    dec = _normalise_cone(Q)
    if dec is None:
        return []
    (l1, l2, l3), V = dec

    denom = l1 - l3
    if denom <= 1e-12 or (-l1 * l3) <= 0:
        return []

    g = np.sqrt(max(l1 - l2, 0.0) / denom)
    h = np.sqrt(max(l2 - l3, 0.0) / denom)
    d_mag = radius * l2 / np.sqrt(-l1 * l3)
    k = np.sqrt(max(l1 - l2, 0.0) * max(l2 - l3, 0.0)) / l2

    out: list[CirclePose] = []
    for s in (+1.0, -1.0):
        n_eig = np.array([g, 0.0, s * h])
        for sd in (+1.0, -1.0):
            d = sd * d_mag
            c_eig = d * np.array([g - k * h, 0.0, s * (h + k * g)])
            C = V @ c_eig
            n = V @ n_eig
            if C[2] <= 0:                     # circle must be in front of camera
                continue
            n = n / np.linalg.norm(n)
            if np.dot(n, C) > 0:              # orient the normal towards the camera
                n = -n
            out.append(CirclePose(center=C, normal=n, radius=float(radius)))
    return out


# ----------------------------------------------------------------- testing
def _project(points, K, R, t):
    pc = points @ R.T + t
    uv = (pc / pc[:, 2:3]) @ K.T
    return uv[:, :2]


def self_test(verbose: bool = True, n_trials: int = 200, seed: int = 0) -> dict:
    """Numerically verify the closed form against synthetic ground truth.

    Builds a circle at a known random pose, projects it, fits an ellipse to
    the projection, recovers the pose, and measures the error. Run via
        python -m src.pose_from_ellipse
    """
    rng = np.random.default_rng(seed)
    K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]])
    R_true_radius = 0.04

    n_err, c_err, found = [], [], 0
    for _ in range(n_trials):
        # random orientation, kept reasonably front-facing so the ellipse
        # is well conditioned (a near-edge-on circle degenerates to a line)
        tilt = rng.uniform(np.deg2rad(10), np.deg2rad(65))
        roll = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.sin(tilt) * np.cos(roll), np.sin(tilt) * np.sin(roll),
                         -np.cos(tilt)])
        axis /= np.linalg.norm(axis)
        centre = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                           rng.uniform(0.25, 0.6)])

        # orthonormal basis of the circle's plane
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, axis)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

        th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        pts = centre + R_true_radius * (np.cos(th)[:, None] * e1
                                         + np.sin(th)[:, None] * e2)
        uv = _project(pts, K, np.eye(3), np.zeros(3))
        ell = cv2.fitEllipse(uv.astype(np.float32))
        Q_img = ellipse_to_conic(ell[0], ell[1], ell[2])

        cands = circle_pose_candidates(Q_img, K, R_true_radius)
        if not cands:
            continue
        found += 1
        # the two-fold ambiguity is inherent; score the better candidate
        best_n = min(np.degrees(np.arccos(np.clip(abs(np.dot(c.normal, axis)), -1, 1)))
                     for c in cands)
        best_c = min(np.linalg.norm(c.center - centre) for c in cands)
        n_err.append(best_n)
        c_err.append(best_c / np.linalg.norm(centre))

    res = {
        "trials": n_trials,
        "solved": found,
        "normal_err_deg_median": float(np.median(n_err)) if n_err else float("nan"),
        "normal_err_deg_p95": float(np.percentile(n_err, 95)) if n_err else float("nan"),
        "center_relerr_median": float(np.median(c_err)) if c_err else float("nan"),
        "center_relerr_p95": float(np.percentile(c_err, 95)) if c_err else float("nan"),
    }
    if verbose:
        print("circle-pose-from-ellipse self test")
        for k, v in res.items():
            print(f"  {k:28s} {v}")
    return res


if __name__ == "__main__":
    self_test()
