"""
cylindrical_mapping.py
------------------------
The mathematical core of the project: warps a flat RGBA texture (the logo,
or a debug grid) onto the cylindrical cup surface described by a
CylinderModel, using an *inverse* mapping so every destination pixel is
filled with no holes.

Geometry
--------
Model the visible part of the cup as a vertical circular cylinder. At
image row y, the fitted silhouette gives a center cx(y) and a radius
(half-width) a(y). A physical point on the cylinder wall at angular
position theta, measured from the surface normal that points straight at
the camera (theta=0 is dead-centre-front, +/-90 degrees is the silhouette
edge), projects to:

    x(y, theta) = cx(y) + a(y) * sin(theta)

This is exact for an orthographic/weak-perspective camera looking at a
cylinder side-on, and is the same relation used for cylindrical label/panorama
projections. Inverting it:

    theta(x, y) = asin( (x - cx(y)) / a(y) )

which is used here as an *inverse warp*: for every destination pixel we
recover which angle of the cylinder (and therefore which column of the
logo) it corresponds to.

The logo is only placed across a limited angular window [-theta_max,
+theta_max] around a chosen center angle theta_center (both user
adjustable: this is the "curvature / width" control), matching the fact
that a real printed logo covers only part of the cup's circumference, not
the full 360 degrees.

cos(theta) is the cosine-falloff of a Lambertian surface turning away from
the camera and is used both as a realism shading term and implicitly
encodes the horizontal foreshortening (columns near +/-theta_max are
visually compressed because dx/dtheta = a*cos(theta) shrinks there) --
exactly the "natural distortion caused by the cup's curved surface" the
project calls for.

Vertical mapping is linear between the band's top/bottom rows -- correct
because a cylinder's generatrices (vertical surface lines) project to
straight lines, not curves.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .surface_estimation import CylinderModel


@dataclass
class WarpResult:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    bgr: np.ndarray       # float32 HxWx3, warped texture color
    alpha: np.ndarray     # float32 HxW, 0..1
    shading: np.ndarray   # float32 HxW, 0..1 (1 = full-on, darker at grazing angle)
    theta: np.ndarray     # float32 HxW, absolute surface angle (radians) per pixel


def _bbox_for_band(model: CylinderModel, y_lo: int, y_hi: int,
                    theta_center: float, theta_max: float, img_w: int) -> tuple[int, int]:
    ys = np.linspace(y_lo, y_hi, num=25)
    a = model.radius(ys)
    cx = model.center(ys)
    edge_a = cx + a * np.sin(theta_center + theta_max)
    edge_b = cx + a * np.sin(theta_center - theta_max)
    x_lo = int(np.floor(min(edge_a.min(), edge_b.min()))) - 2
    x_hi = int(np.ceil(max(edge_a.max(), edge_b.max()))) + 2
    return max(0, x_lo), min(img_w, x_hi)


def warp_texture_to_cylinder(
    model: CylinderModel,
    texture_rgba: np.ndarray,
    y_center: float,
    band_height: float,
    theta_center_deg: float,
    theta_max_deg: float,
    img_shape: tuple[int, int],
    shading_min: float = 0.35,
) -> WarpResult | None:
    img_h, img_w = img_shape[:2]
    y_lo = int(round(y_center - band_height / 2))
    y_hi = int(round(y_center + band_height / 2))
    y_lo = max(0, y_lo)
    y_hi = min(img_h - 1, y_hi)
    if y_hi - y_lo < 2:
        return None

    theta_center = np.deg2rad(theta_center_deg)
    theta_max = np.deg2rad(max(1.0, theta_max_deg))

    x_lo, x_hi = _bbox_for_band(model, y_lo, y_hi, theta_center, theta_max, img_w)
    if x_hi - x_lo < 2:
        return None

    ys = np.arange(y_lo, y_hi + 1)
    xs = np.arange(x_lo, x_hi + 1)
    Xg, Yg = np.meshgrid(xs, ys)

    a_y = model.radius(Yg)
    cx_y = model.center(Yg)
    a_y_safe = np.where(np.abs(a_y) < 1e-3, 1e-3, a_y)

    sin_theta = np.clip((Xg - cx_y) / a_y_safe, -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    rel = theta - theta_center
    # wrap rel into [-pi, pi] to avoid discontinuities if center is near +/-90deg
    rel = (rel + np.pi) % (2 * np.pi) - np.pi

    valid = (np.abs(rel) <= theta_max) & (a_y > 1.0)

    u = 0.5 + 0.5 * (rel / theta_max)
    v = (Yg - y_lo) / float(max(1, y_hi - y_lo))

    logo_h, logo_w = texture_rgba.shape[:2]
    src_x = (u * (logo_w - 1)).astype(np.float32)
    src_y = (v * (logo_h - 1)).astype(np.float32)

    warped = cv2.remap(
        texture_rgba, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped = warped.astype(np.float32)

    alpha = warped[..., 3] / 255.0
    alpha[~valid] = 0.0
    alpha[(v < 0) | (v > 1)] = 0.0

    shading = shading_min + (1.0 - shading_min) * np.clip(np.cos(theta), 0.0, 1.0)

    return WarpResult(
        bbox=(x_lo, y_lo, x_hi + 1, y_hi + 1),
        bgr=warped[..., :3],
        alpha=alpha,
        shading=shading.astype(np.float32),
        theta=theta.astype(np.float32),
    )
