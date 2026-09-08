"""
demo_3d.py
-----------
FIRST DEMONSTRATION (standalone, no video involved).

  1. Generate a 3D cup mesh (surface of revolution).
  2. Apply the IIT Delhi logo as a UV texture on that mesh.
  3. Render it from several camera angles orbiting the cup.
  4. Show the logo wrapping around the cylinder: face-on at the centre,
     compressed and curving away toward the silhouette, and disappearing
     around the back -- all produced by the 3D geometry, nothing 2D.

Run:  python demo_3d.py
Out:  outputs/demo3d_turntable.png   (contact sheet of orbit angles)
      outputs/demo3d_buffers.png     (depth / normal / UV debug buffers)
      outputs/demo3d_atlas.png       (the unrolled surface texture)
      outputs/demo3d_turntable.mp4   (continuous orbit)
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.camera3d import Intrinsics, look_at
from src.cup_model_3d import CupGeometry, build_cup_mesh
from src.renderer3d import render_mesh, view_directions
from src.texture_atlas import LogoPlacement, build_atlas, load_logo_rgba, sample_atlas

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

CERAMIC = np.array([238.0, 238.0, 240.0])   # BGR base colour of the mug
BACKDROP = np.array([38.0, 34.0, 30.0])


def shade(fb, intr, atlas, light_to_cam=np.array([-0.35, -0.55, -0.76])):
    """Simple Lambertian + rim-light shading of the ceramic, with the logo
    texture composited on top as ink.

    `light_to_cam` is the direction *towards* the light in camera
    coordinates, so a surface facing the camera (normal ~ (0,0,-1), since
    the camera looks down +Z) is lit rather than dark.
    """
    H, W = fb.depth.shape
    img = np.tile(BACKDROP, (H, W, 1))

    if not fb.mask.any():
        return img.astype(np.uint8)

    n = fb.normal
    v = view_directions(fb, intr)

    L = light_to_cam / np.linalg.norm(light_to_cam)
    lambert = np.clip(n @ L, 0.0, 1.0)
    facing = np.clip(np.einsum("ijk,ijk->ij", n, v), 0.0, 1.0)

    shading = 0.55 + 0.40 * lambert + 0.18 * facing ** 3
    body = CERAMIC[None, None, :] * shading[..., None]

    tex = sample_atlas(atlas, fb.uv)
    a = (tex[..., 3:4] / 255.0)
    ink = tex[..., :3] * shading[..., None]
    surface = body * (1 - a) + ink * a

    out = np.where(fb.mask[..., None], surface, img)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    geom = CupGeometry(r_top=0.040, r_bottom=0.042, height=0.095)
    mesh = build_cup_mesh(geom, n_theta=96, n_v=24, cap_top=True)
    print(f"mesh: {len(mesh.vertices)} vertices, {mesh.n_faces} triangles")

    logo = load_logo_rgba(os.path.join(BASE, "data", "logos", "iitd_logo.png"))
    placement = LogoPlacement(theta_center_deg=0.0, v_center=0.46, height_frac=0.46)
    atlas = build_atlas(logo, geom, placement)
    cv2.imwrite(os.path.join(OUT, "demo3d_atlas.png"), atlas)
    print(f"atlas: {atlas.shape}, logo occupies "
          f"{(atlas[...,3] > 0).mean()*100:.1f}% of the unrolled surface")

    intr = Intrinsics.from_fov(640, 640, hfov_deg=40.0)
    target = np.array([0.0, geom.height * 0.48, 0.0])
    radius = 0.27
    elev = np.deg2rad(18.0)

    # ---- contact sheet over a half-orbit -------------------------------
    angles = np.linspace(-90, 90, 7)
    tiles = []
    for az in angles:
        a = np.deg2rad(az)
        eye = target + np.array([radius * np.sin(a) * np.cos(elev),
                                  radius * np.sin(elev),
                                  radius * np.cos(a) * np.cos(elev)])
        pose = look_at(eye, target)
        fb = render_mesh(mesh, pose, intr)
        img = shade(fb, intr, atlas)
        cv2.putText(img, f"azimuth {az:+.0f}deg", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.hstack(tiles)
    cv2.imwrite(os.path.join(OUT, "demo3d_turntable.png"), sheet)
    print("wrote outputs/demo3d_turntable.png")

    # ---- debug buffers at a 3/4 view ------------------------------------
    a = np.deg2rad(-35.0)
    eye = target + np.array([radius * np.sin(a) * np.cos(elev),
                              radius * np.sin(elev),
                              radius * np.cos(a) * np.cos(elev)])
    fb = render_mesh(mesh, look_at(eye, target), intr)

    d = fb.depth.copy()
    finite = np.isfinite(d)
    dv = np.zeros_like(d)
    if finite.any():
        lo, hi = d[finite].min(), d[finite].max()
        dv[finite] = 1.0 - (d[finite] - lo) / max(hi - lo, 1e-9)
    depth_vis = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_vis[~finite] = 0

    normal_vis = ((fb.normal * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_vis[~fb.mask] = 0

    uv_vis = np.zeros((*fb.uv.shape[:2], 3), np.uint8)
    uv_vis[..., 2] = (fb.uv[..., 0] * 255).astype(np.uint8)   # u -> red
    uv_vis[..., 1] = (fb.uv[..., 1] * 255).astype(np.uint8)   # v -> green
    uv_vis[~fb.mask] = 0

    shaded = shade(fb, intr, atlas)
    for im, label in ((depth_vis, "depth (z-buffer)"), (normal_vis, "surface normals"),
                       (uv_vis, "UV coords"), (shaded, "textured render")):
        cv2.putText(im, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "demo3d_buffers.png"),
                np.hstack([depth_vis, normal_vis, uv_vis, shaded]))
    print("wrote outputs/demo3d_buffers.png")

    # ---- full turntable video -------------------------------------------
    vw = cv2.VideoWriter(os.path.join(OUT, "demo3d_turntable.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 640))
    for az in np.linspace(0, 360, 100, endpoint=False):
        a = np.deg2rad(az)
        eye = target + np.array([radius * np.sin(a) * np.cos(elev),
                                  radius * np.sin(elev),
                                  radius * np.cos(a) * np.cos(elev)])
        vw.write(shade(render_mesh(mesh, look_at(eye, target), intr), intr, atlas))
    vw.release()
    print("wrote outputs/demo3d_turntable.mp4")


if __name__ == "__main__":
    main()
