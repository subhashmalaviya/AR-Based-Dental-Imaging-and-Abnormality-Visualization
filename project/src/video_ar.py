"""
video_ar.py
-------------
Ties tracking.py + cylindrical_mapping.py + blending.py together into a
frame-by-frame video AR pipeline: the logo stays attached to the tracked
cup surface as the camera/cup move, instead of being pasted at a fixed
frame location.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import cylindrical_mapping, blending
from .tracking import CupTracker
from .surface_estimation import CylinderModel


@dataclass
class PlacementParams:
    # Vertical placement is anchored to the cup's *bottom* edge and scaled
    # by the *locally measured radius* there, both in units of "radius at
    # y_bottom" -- not as a fraction of (y_top..y_bottom). y_bottom (where
    # the cup meets the table) is a much sharper, more stable edge than
    # y_top (a heuristically-detected base/wall transition), and scaling
    # by the local radius keeps the logo size/position correct as the cup
    # grows or shrinks on screen (camera zoom/dolly) without needing y_top
    # at all. This avoids compounding drift from small per-redetect
    # variation in the (noisier) y_top estimate -- see README "Temporal
    # stability" section.
    v_offset_factor: float = 1.40  # distance above y_bottom, in units of radius(y_bottom)
    height_frac: float = 1.30      # logo band height, in units of radius(y_bottom)
    theta_center_deg: float = 0.0  # angular position around the cylinder (curvature control)
    theta_max_deg: float = 42.0   # angular half-width the logo covers
    opacity: float = 0.95
    blend_strength: float = 0.55
    texture_strength: float = 0.08


def render_logo_on_model(
    frame_bgr: np.ndarray,
    model: CylinderModel,
    logo_rgba: np.ndarray,
    params: PlacementParams,
) -> np.ndarray:
    radius_bottom = model.radius(model.y_bottom)
    y_center = model.y_bottom - params.v_offset_factor * radius_bottom
    band_height = params.height_frac * radius_bottom
    # never let the band climb above the detected wall-start row
    min_center = model.y_top + band_height / 2
    y_center = max(y_center, min_center)

    warp = cylindrical_mapping.warp_texture_to_cylinder(
        model, logo_rgba, y_center, band_height,
        params.theta_center_deg, params.theta_max_deg, frame_bgr.shape,
    )
    if warp is None:
        return frame_bgr
    return blending.composite_logo(
        frame_bgr, warp,
        opacity=params.opacity,
        blend_strength=params.blend_strength,
        texture_strength=params.texture_strength,
        occlusion_mask=model.body_mask,
    )


def process_video(
    in_path: str,
    out_path: str,
    logo_rgba: np.ndarray,
    params: PlacementParams,
    tracker: CupTracker | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
) -> dict:
    """Runs the full markerless-AR loop over the input video and writes the
    augmented result to out_path. Returns simple stats for reporting."""
    from .image_loader import open_video, video_properties, make_video_writer

    cap = open_video(in_path)
    props = video_properties(cap)
    writer = make_video_writer(out_path, props["width"], props["height"], props["fps"])

    tracker = tracker or CupTracker()
    n_frames = props["frame_count"] if not max_frames else min(props["frame_count"], max_frames)

    n_tracked = 0
    n_redetected = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        was_redetect = tracker.state.frames_since_redetect == 0
        model = tracker.update(frame)
        if model is not None:
            frame = render_logo_on_model(frame, model, logo_rgba, params)
            n_tracked += 1
            if was_redetect:
                n_redetected += 1
        writer.write(frame)
        idx += 1
        if progress_cb:
            progress_cb(idx, n_frames)

    cap.release()
    writer.release()
    return {"frames_processed": idx, "frames_with_logo": n_tracked, "redetections": n_redetected}
