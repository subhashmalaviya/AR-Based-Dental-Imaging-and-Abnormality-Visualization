"""
visualization.py
-------------------
Debug/demo views for the presentation: silhouette overlay, the fitted
cylinder boundary, a coordinate grid warped onto the surface (visualises
theta/v directly), and a multi-panel figure combining everything.
"""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cylindrical_mapping import warp_texture_to_cylinder
from .surface_estimation import CylinderModel


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray,
                       color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    out = image_bgr.copy()
    colored = np.zeros_like(out)
    colored[:] = color
    m = (mask > 0)[..., None]
    out = np.where(m, (out * (1 - alpha) + colored * alpha).astype(np.uint8), out)
    return out


def draw_cylinder_boundary(image_bgr: np.ndarray, model: CylinderModel) -> np.ndarray:
    out = image_bgr.copy()
    ys, _ = np.where(model.full_mask > 0)
    if len(ys) == 0:
        return out
    y0, y1 = ys.min(), ys.max()
    for y in range(y0, y1 + 1, 2):
        lx = int(round(model.left(y)))
        rx = int(round(model.right(y)))
        if 0 <= lx < out.shape[1]:
            cv2.circle(out, (lx, y), 1, (0, 0, 255), -1)
        if 0 <= rx < out.shape[1]:
            cv2.circle(out, (rx, y), 1, (255, 0, 0), -1)
    cv2.line(out, (0, model.y_top), (out.shape[1] - 1, model.y_top), (0, 255, 255), 1)
    cv2.line(out, (0, model.y_bottom), (out.shape[1] - 1, model.y_bottom), (0, 255, 255), 1)
    return out


def make_grid_texture(size=(600, 600), n_u=12, n_v=6) -> np.ndarray:
    """Synthetic checker+gridline RGBA texture used purely to visualise the
    theta/v cylindrical coordinate mapping (Part 10-C)."""
    h, w = size
    tex = np.full((h, w, 4), 255, np.uint8)
    for i in range(n_u + 1):
        x = int(i * (w - 1) / n_u)
        cv2.line(tex, (x, 0), (x, h - 1), (30, 30, 200, 255), 2)
    for j in range(n_v + 1):
        y = int(j * (h - 1) / n_v)
        cv2.line(tex, (0, y), (w - 1, y), (200, 30, 30, 255), 2)
    cv2.rectangle(tex, (0, 0), (w - 1, h - 1), (0, 150, 0, 255), 4)
    tex[..., 3] = 255
    return tex


def render_coordinate_grid(image_bgr: np.ndarray, model: CylinderModel,
                            y_center, band_height, theta_center_deg, theta_max_deg) -> np.ndarray:
    tex = make_grid_texture()
    warp = warp_texture_to_cylinder(
        model, tex, y_center, band_height, theta_center_deg, theta_max_deg,
        image_bgr.shape, shading_min=1.0,
    )
    out = image_bgr.copy()
    if warp is None:
        return out
    x0, y0, x1, y1 = warp.bbox
    region = out[y0:y1, x0:x1].astype(np.float32)
    a = warp.alpha[..., None]
    region = region * (1 - a) + warp.bgr * a
    out[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def make_panel(images: list[np.ndarray], titles: list[str], out_path: str, ncols: int = 3):
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(bgr_to_rgb(img))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
