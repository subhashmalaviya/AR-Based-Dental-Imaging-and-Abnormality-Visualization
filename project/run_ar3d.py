"""
run_ar3d.py
------------
Command-line entry point for the true-3D AR pipeline.

    python run_ar3d.py image   # single photo  -> outputs/ti1_ar3d.png (+ figure)
    python run_ar3d.py video   # whole clip    -> outputs/tv1_ar3d.mp4
    python run_ar3d.py demo    # synthetic turntable (no input footage)

Every result is produced by projecting a textured 3D mesh through an
estimated camera pose. Nothing is composited in 2D.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from src.azimuth import unwrap_visualisation
from src.camera3d import Intrinsics, project_points
from src.cup_model_3d import CupGeometry, build_cup_mesh
from src.disc_tracker import track_sequence
from src.ellipse_detection import draw_ellipse
from src.pose_estimation3d import cup_pose_candidates, reprojection_error
from src.renderer3d import render_mesh
from src.texture_atlas import (LogoPlacement, build_atlas, face_has_texture,
                                load_logo_rgba)
from src.video_ar3d import (ARConfig, composite_print, process_video,
                             theta_facing_camera)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

CUP_IMG = os.path.join(BASE, "data", "cup_image", "ti1.png")
LOGO = os.path.join(BASE, "data", "logos", "iitd_logo.png")
VIDEO = os.path.join(BASE, "data", "videos", "tv1.mp4")


def draw_wireframe(frame, pose, geom, intr):
    v = frame.copy()
    th = np.linspace(0, 2 * np.pi, 120)
    for vv, col in ((0.0, (255, 60, 0)), (1.0, (0, 220, 255))):
        uv, z = project_points(geom.surface_point(th, np.full_like(th, vv)), pose, intr)
        if np.all(z > 0):
            cv2.polylines(v, [uv.astype(np.int32)], True, col, 3, cv2.LINE_AA)
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        uv, z = project_points(geom.surface_point(np.array([a, a]), np.array([0., 1.])),
                                pose, intr)
        if np.all(z > 0):
            cv2.line(v, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)),
                     (255, 255, 255), 1, cv2.LINE_AA)
    return v


def run_image(cfg: ARConfig):
    frame = cv2.imread(CUP_IMG)
    if frame is None:
        raise FileNotFoundError(CUP_IMG)
    H, W = frame.shape[:2]

    # the still is a downscaled crop of the same scene; scale the focal with it
    intr = Intrinsics(cfg.focal * W / 1080.0, cfg.focal * W / 1080.0,
                      W / 2.0, H / 2.0, W, H)
    geom = cfg.geometry

    tracked = track_sequence([frame], verbose=True)
    if tracked[0] is None:
        print("could not find the cup's disc in the still image")
        return
    obs = tracked[0].obs

    cands = [c for c in cup_pose_candidates(obs, intr, geom, 0.0) if c.physically_valid]
    if not cands:
        print("no physically valid pose")
        return
    pose = cands[0].pose
    err = reprojection_error(pose, obs, intr, geom)
    print(f"  pose: distance {np.linalg.norm(pose.tvec)*100:.1f} cm, "
          f"disc reprojection error {err:.3f} px")

    placement = cfg.placement
    if placement.theta_center_deg is None:
        placement = LogoPlacement(theta_facing_camera(pose, geom, placement.v_center),
                                   placement.v_center, placement.height_frac,
                                   placement.opacity)
    mesh = build_cup_mesh(geom, cfg.mesh_theta, cfg.mesh_v, cap_top=True)
    logo = load_logo_rgba(LOGO)
    atlas = build_atlas(logo, geom, placement)
    subset = face_has_texture(atlas, mesh)

    fb = render_mesh(mesh, pose, intr, face_subset=subset)
    result = composite_print(frame, fb, atlas, intr, cfg)
    cv2.imwrite(os.path.join(OUT, "ti1_ar3d.png"), result)

    # ---- presentation figure -------------------------------------------
    ell_vis = draw_ellipse(frame, obs)
    wire = draw_wireframe(frame, pose, geom, intr)

    fb_full = render_mesh(mesh, pose, intr)
    d = fb_full.depth.copy()
    fin = np.isfinite(d)
    dv = np.zeros_like(d)
    if fin.any():
        lo, hi = d[fin].min(), d[fin].max()
        dv[fin] = 1.0 - (d[fin] - lo) / max(hi - lo, 1e-9)
    depth_vis = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_vis[~fin] = 0

    uvv = np.zeros((*fb_full.uv.shape[:2], 3), np.uint8)
    uvv[..., 2] = (fb_full.uv[..., 0] * 255).astype(np.uint8)
    uvv[..., 1] = (fb_full.uv[..., 1] * 255).astype(np.uint8)
    uvv[~fb_full.mask] = 0

    unwrap = unwrap_visualisation(frame, pose, geom, intr)
    unwrap = cv2.resize(unwrap, (W, max(1, H // 4)))
    pad = np.zeros((H, W, 3), np.uint8)
    pad[:unwrap.shape[0]] = unwrap

    panels = [frame, ell_vis, wire, depth_vis, uvv, pad, result]
    titles = ["A original", "B disc ellipse", "C 6-DoF pose (3D model)",
              "D z-buffer", "E UV coords", "F surface unwrap", "G AR result"]
    tiles = []
    for im, t in zip(panels, titles):
        im = im.copy()
        cv2.putText(im, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(im, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(im, (300, int(300 * H / W))))
    cv2.imwrite(os.path.join(OUT, "pipeline_3d.png"), np.hstack(tiles))
    print("  wrote outputs/ti1_ar3d.png and outputs/pipeline_3d.png")


def run_video(cfg: ARConfig, max_frames=None):
    t0 = time.time()

    def cb(i, n):
        if i % 50 == 0 or i == n:
            print(f"   rendering {i}/{n}  ({time.time()-t0:.0f}s)")

    stats = process_video(VIDEO, os.path.join(OUT, "tv1_ar3d.mp4"), LOGO,
                           cfg=cfg, max_frames=max_frames, progress_cb=cb)
    print(f"  {stats}  in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["image", "video", "demo"])
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--focal", type=float, default=1500.0)
    ap.add_argument("--theta", type=float, default=None,
                    help="logo azimuth in cup coords (deg); default = camera-facing")
    ap.add_argument("--v", type=float, default=0.50)
    ap.add_argument("--size", type=float, default=0.42)
    args = ap.parse_args()

    cfg = ARConfig(focal=args.focal)
    cfg.placement = LogoPlacement(args.theta, args.v, args.size)

    if args.mode == "demo":
        import demo_3d
        demo_3d.main()
    elif args.mode == "image":
        run_image(cfg)
    else:
        run_video(cfg, args.frames)


if __name__ == "__main__":
    main()
