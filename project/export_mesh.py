"""
export_mesh.py
---------------
Writes the textured 3D cup as a standard OBJ + MTL + PNG triple, so the
same mesh and UV atlas this project renders can be opened directly in
Open3D, MeshLab or Blender:

    python export_mesh.py
    python -c "import open3d as o3d; \
        m=o3d.io.read_triangle_mesh('outputs/cup_textured.obj', True); \
        o3d.visualization.draw_geometries([m])"

An OBJ is written rather than driving Open3D's offscreen renderer from the
pipeline itself: Open3D's GPU rendering needs a working EGL/OpenGL context,
which is not available on a headless machine, whereas the software
rasteriser in src/renderer3d.py always is -- and it hands back the depth,
normal and UV buffers the compositor needs, which a screenshot API does not.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.cup_model_3d import CupGeometry, build_cup_mesh
from src.texture_atlas import LogoPlacement, build_atlas, load_logo_rgba

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")


def write_obj(path: str, mesh, texture_name: str, mtl_name: str):
    with open(path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o cup\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in mesh.uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in mesh.normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("usemtl cupmat\n")
        for a, b, c in mesh.faces:
            f.write(f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}\n")


def write_mtl(path: str, texture_name: str):
    with open(path, "w") as f:
        f.write("newmtl cupmat\n")
        f.write("Ka 0.9 0.9 0.9\nKd 1.0 1.0 1.0\nKs 0.15 0.15 0.15\nNs 32\nd 1.0\n")
        f.write(f"map_Kd {texture_name}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    geom = CupGeometry(r_top=0.035, r_bottom=0.035, height=0.0858, r_disc_ratio=1.0)
    mesh = build_cup_mesh(geom, n_theta=128, n_v=32, cap_top=True)

    logo = load_logo_rgba(os.path.join(BASE, "data", "logos", "iitd_logo.png"))
    atlas = build_atlas(logo, geom, LogoPlacement(0.0, 0.5, 0.42))

    # flatten the RGBA atlas over a ceramic-white base for a viewer-friendly map
    a = atlas[..., 3:4].astype(np.float32) / 255.0
    base = np.full_like(atlas[..., :3], 240, dtype=np.float32)
    tex = (base * (1 - a) + atlas[..., :3].astype(np.float32) * a).astype(np.uint8)

    cv2.imwrite(os.path.join(OUT, "cup_texture.png"), tex)
    write_mtl(os.path.join(OUT, "cup_textured.mtl"), "cup_texture.png")
    write_obj(os.path.join(OUT, "cup_textured.obj"), mesh,
              "cup_texture.png", "cup_textured.mtl")
    print(f"wrote outputs/cup_textured.obj  ({len(mesh.vertices)} verts, "
          f"{mesh.n_faces} faces) + .mtl + cup_texture.png")


if __name__ == "__main__":
    main()
