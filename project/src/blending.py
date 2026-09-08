"""
blending.py
-------------
Composites a warped texture (see cylindrical_mapping.WarpResult) onto the
cup image so it reads as ink/print on ceramic rather than a flat sticker.

Blend model
-----------
For each pixel under the logo's alpha:

  base_lum      = luminance(cup pixel) / 255                (0..1)
  shaded_logo   = logo_rgb * shading * (0.5 + 0.5*base_lum)   -- the logo
                  picks up both the cylindrical shading falloff (from the
                  geometry) and the cup's own local brightness, so a
                  highlight/shadow already on the ceramic still shows
                  through the print.
  multiplied    = (shaded_logo/255 * cup/255) * 255           -- multiply
                  blend: darkens the print where the ceramic itself is
                  darker, like real transfer-printed ink.
  printed       = lerp(shaded_logo, multiplied, blend_strength)
  out           = lerp(cup, printed, alpha * opacity)

`blend_strength` is the "multiply vs. flat-shaded" slider (Part 9's
"blending strength"); `opacity` is the overall logo opacity slider.

A small high-frequency term is optionally added back from the cup's own
texture (its local detail relative to a blurred version of itself) so the
ceramic's micro-texture / glaze specular isn't completely flattened under
the print -- the "slight texture integration" the spec asks for.
"""
from __future__ import annotations

import cv2
import numpy as np

from .cylindrical_mapping import WarpResult


def composite_logo(
    base_bgr: np.ndarray,
    warp: WarpResult,
    opacity: float = 0.95,
    blend_strength: float = 0.55,
    texture_strength: float = 0.08,
    occlusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Returns a new image (copy of base_bgr) with the logo composited only
    inside warp.bbox; everything outside that box is untouched."""
    out = base_bgr.copy()
    x0, y0, x1, y1 = warp.bbox
    region = out[y0:y1, x0:x1].astype(np.float32)

    alpha = warp.alpha.copy()
    if occlusion_mask is not None:
        occ = occlusion_mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        alpha = alpha * occ

    if alpha.max() <= 0:
        return out

    base_lum = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum_mod = 0.5 + 0.5 * base_lum

    shading = warp.shading[..., None]
    logo_rgb = warp.bgr  # already float32 0..255

    shaded_logo = logo_rgb * shading * lum_mod[..., None]

    multiplied = (shaded_logo / 255.0) * (region / 255.0) * 255.0
    printed = (1 - blend_strength) * shaded_logo + blend_strength * multiplied

    if texture_strength > 0:
        blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=2.0)
        high_freq = region - blurred
        printed = printed + high_freq * texture_strength

    printed = np.clip(printed, 0, 255)

    a = np.clip(alpha * opacity, 0.0, 1.0)[..., None]
    blended = region * (1 - a) + printed * a
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
