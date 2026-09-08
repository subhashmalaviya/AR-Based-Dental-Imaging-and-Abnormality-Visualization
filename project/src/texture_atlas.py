"""
texture_atlas.py
-----------------
Builds the RGBA texture that is UV-mapped onto the cup mesh.

The atlas is the cup's *unrolled surface*: the horizontal axis is azimuth
theta in [0, 2pi) and the vertical axis is the along-axis coordinate
v in [0, 1]. The logo is painted into this atlas once, at a fixed
(theta, v) location.

This is the structural reason the logo is genuinely attached to the 3D
object: after this step the logo has no 2D image-space existence at all.
It is texture on a surface, and the only thing that changes per video
frame is the camera pose used to render that surface.

The same file is what you would later paint dental abnormality regions
into -- the pipeline downstream does not care what the atlas contains.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .cup_model_3d import CupGeometry


@dataclass
class LogoPlacement:
    """Where the logo lives on the cup surface, in cup coordinates."""
    # None = auto-place on the ceramic facing the camera at the anchor
    # frame; a number pins the logo to that absolute azimuth in cup coords.
    theta_center_deg: float | None = None
    v_center: float = 0.45          # along-axis position, 0 = table, 1 = top disc
    height_frac: float = 0.34       # logo height as a fraction of cup height
    opacity: float = 1.0


def _alpha_from_white_background(bgr: np.ndarray, white_thresh: int = 245,
                                  feather: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg = (gray < white_thresh).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if feather > 0:
        fg = cv2.GaussianBlur(fg, (feather * 2 + 1, feather * 2 + 1), 0)
    return fg


def load_logo_rgba(path: str, feather: int = 2) -> np.ndarray:
    """Load the logo and give it a real alpha channel.

    The supplied iitd_logo.png is RGB on a flat white canvas with no alpha,
    so the mask is derived by thresholding the background. The logo's own
    pixels (emblem, Hindi text, English ring text) are never modified --
    only classified as foreground/background and feathered at the edge.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if raw.shape[2] == 4:
        bgr, alpha = raw[..., :3], raw[..., 3]
        if alpha.min() > 250:      # nominally-opaque alpha carries no information
            alpha = _alpha_from_white_background(bgr, feather=feather)
    else:
        bgr = raw
        alpha = _alpha_from_white_background(bgr, feather=feather)

    rgba = np.dstack([bgr, alpha]).astype(np.uint8)
    ys, xs = np.where(alpha > 8)
    if len(ys) == 0:
        return rgba
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_atlas(logo_rgba: np.ndarray, geom: CupGeometry,
                 placement: LogoPlacement,
                 atlas_w: int = 2048, atlas_h: int = 512) -> np.ndarray:
    """Paint the logo into the unrolled (theta, v) surface atlas.

    The angular width is derived from the logo's aspect ratio and the cup's
    metric dimensions so the print is not stretched: on the real surface the
    logo spans an arc length of R*dtheta and a height of dv*height, and we
    require arc_length / physical_height == logo_w / logo_h.

    Returns an (atlas_h, atlas_w, 4) uint8 BGRA image. Row 0 is v = 1 (the
    top disc end) so the atlas reads upright when viewed as an image.
    """
    atlas = np.zeros((atlas_h, atlas_w, 4), np.uint8)

    lh, lw = logo_rgba.shape[:2]
    dv = float(np.clip(placement.height_frac, 1e-3, 1.0))
    physical_h = dv * geom.height
    r_at_logo = float(geom.radius_at(placement.v_center))
    arc_len = physical_h * (lw / float(lh))
    dtheta = arc_len / max(r_at_logo, 1e-6)          # radians of azimuth covered

    # Convert the (theta, v) footprint into atlas pixels.
    px_w = int(round(dtheta / (2.0 * np.pi) * atlas_w))
    px_h = int(round(dv * atlas_h))
    px_w = max(2, min(px_w, atlas_w))
    px_h = max(2, min(px_h, atlas_h))

    logo_resized = cv2.resize(logo_rgba, (px_w, px_h), interpolation=cv2.INTER_AREA)
    if placement.opacity < 1.0:
        logo_resized = logo_resized.copy()
        logo_resized[..., 3] = (logo_resized[..., 3].astype(np.float32)
                                 * placement.opacity).astype(np.uint8)

    u_center = (np.deg2rad(placement.theta_center_deg) % (2.0 * np.pi)) / (2.0 * np.pi)
    col_center = u_center * atlas_w
    row_center = (1.0 - placement.v_center) * atlas_h   # v=1 -> row 0

    col0 = int(round(col_center - px_w / 2.0))
    row0 = int(round(row_center - px_h / 2.0))

    # Vertical placement is clipped; horizontal placement wraps, because
    # theta is periodic and a logo may legitimately straddle the seam.
    for dy in range(px_h):
        ry = row0 + dy
        if ry < 0 or ry >= atlas_h:
            continue
        cols = (np.arange(px_w) + col0) % atlas_w
        atlas[ry, cols] = logo_resized[dy]

    return atlas


def sample_atlas(atlas: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup. uv: (...,2) with u,v in [0,1].

    u wraps (the surface is periodic in theta); v is clamped.
    Returns float32 BGRA in 0..255.
    """
    h, w = atlas.shape[:2]
    u = uv[..., 0]
    v = uv[..., 1]

    x = (u % 1.0) * w - 0.5
    y = (1.0 - np.clip(v, 0.0, 1.0)) * h - 0.5

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    x0m = x0 % w
    x1m = (x0 + 1) % w
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)

    a = atlas.astype(np.float32)
    c00 = a[y0c, x0m]
    c10 = a[y0c, x1m]
    c01 = a[y1c, x0m]
    c11 = a[y1c, x1m]

    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def face_has_texture(atlas: np.ndarray, mesh, threshold: int = 4) -> np.ndarray:
    """Per-face flag: does this triangle carry any non-transparent texture?

    Used to skip empty geometry when compositing into video -- only the
    handful of triangles under the logo need rasterising, which is a large
    speed-up over drawing the whole cup every frame.
    """
    h, w = atlas.shape[:2]
    alpha = atlas[..., 3]
    uv = mesh.uvs[mesh.faces]                       # (F,3,2)

    # Conservative: test the triangle's UV bounding box in atlas pixels.
    u0 = np.floor((uv[..., 0].min(axis=1) % 1.0) * w).astype(int)
    u1 = np.ceil((uv[..., 0].max(axis=1) % 1.0) * w).astype(int)
    v_lo = np.clip(np.floor((1.0 - uv[..., 1].max(axis=1)) * h).astype(int), 0, h - 1)
    v_hi = np.clip(np.ceil((1.0 - uv[..., 1].min(axis=1)) * h).astype(int), 0, h - 1)

    out = np.zeros(len(mesh.faces), dtype=bool)
    for i in range(len(mesh.faces)):
        a, b = u0[i], max(u1[i], u0[i] + 1)
        lo, hi = v_lo[i], max(v_hi[i], v_lo[i] + 1)
        if b <= w:
            patch = alpha[lo:hi, a:b]
        else:                                        # wraps the seam
            patch = np.concatenate([alpha[lo:hi, a:w], alpha[lo:hi, 0:b % w]], axis=1)
        out[i] = patch.size > 0 and patch.max() > threshold
    return out
