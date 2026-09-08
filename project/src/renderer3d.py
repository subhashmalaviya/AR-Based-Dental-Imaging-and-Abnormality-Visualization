"""
renderer3d.py
--------------
A small software rasteriser: projects the cup mesh through the pinhole
camera and produces per-pixel depth, UV, and surface normal.

Written explicitly (rather than calling an OpenGL/Open3D offscreen
renderer) for three reasons:
  * it runs headless with no GPU/EGL dependency,
  * every stage of the 3D -> 2D pipeline is inspectable, which is the
    point of the exercise, and
  * it hands back the intermediate buffers (depth, normal, UV) that the
    compositor needs to blend a *print* into real video, which a
    screenshot-style renderer would not expose.

Correctness details that matter here:
  * Visibility uses a z-buffer plus back-face culling. The cup is convex
    in cross-section, so culling alone already resolves front/back wall;
    the z-buffer additionally handles the end cap.
  * UV and normal interpolation is **perspective-correct**: attributes are
    interpolated as attr/z, alongside 1/z, and divided at the end. Plain
    screen-space (affine) interpolation visibly skews a texture on a
    surface seen at a grazing angle, which is exactly our case near the
    silhouette.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera3d import Intrinsics, Pose, project_points, transform_points
from .cup_model_3d import Mesh


@dataclass
class Framebuffer:
    depth: np.ndarray    # (H,W) float32, +inf where nothing was drawn
    uv: np.ndarray       # (H,W,2) float32, surface texture coordinates
    normal: np.ndarray   # (H,W,3) float32, camera-space unit normals
    mask: np.ndarray     # (H,W) bool, geometry coverage
    tri_id: np.ndarray   # (H,W) int32, -1 where empty


def render_mesh(mesh: Mesh, pose: Pose, intr: Intrinsics,
                 face_subset: np.ndarray | None = None,
                 near: float = 1e-4) -> Framebuffer:
    """Rasterise `mesh` seen from `pose` through `intr`.

    face_subset: optional boolean (F,) or index array selecting which
    triangles to draw -- used to render only texture-bearing faces.
    """
    H, W = intr.height, intr.width

    depth = np.full((H, W), np.inf, dtype=np.float32)
    uv_buf = np.zeros((H, W, 2), dtype=np.float32)
    nrm_buf = np.zeros((H, W, 3), dtype=np.float32)
    tri_buf = np.full((H, W), -1, dtype=np.int32)

    verts_cam = transform_points(mesh.vertices, pose)
    uv_all, z_all = project_points(mesh.vertices, pose, intr)
    normals_cam = mesh.normals @ pose.R.T

    faces = mesh.faces
    idx = np.arange(len(faces))
    if face_subset is not None:
        face_subset = np.asarray(face_subset)
        if face_subset.dtype == bool:
            idx = idx[face_subset]
        else:
            idx = face_subset
    if len(idx) == 0:
        return Framebuffer(depth, uv_buf, nrm_buf, np.isfinite(depth), tri_buf)

    tri = faces[idx]                       # (T,3)
    z_tri = z_all[tri]                     # (T,3)
    p_tri = uv_all[tri]                    # (T,3,2) screen coords
    vc_tri = verts_cam[tri]                # (T,3,3)

    # Reject triangles with any vertex at/behind the camera (no clipping
    # implemented -- the cup is always fully in front in this application).
    ok = np.all(z_tri > near, axis=1)

    # Back-face culling via the geometric face normal against the view ray.
    e1 = vc_tri[:, 1] - vc_tri[:, 0]
    e2 = vc_tri[:, 2] - vc_tri[:, 0]
    face_n = np.cross(e1, e2)
    centroid = vc_tri.mean(axis=1)
    facing = np.einsum("ij,ij->i", face_n, centroid) < 0.0
    ok &= facing

    # Screen-space bounding boxes, discard fully offscreen triangles.
    xmin = np.floor(p_tri[..., 0].min(axis=1)).astype(int)
    xmax = np.ceil(p_tri[..., 0].max(axis=1)).astype(int)
    ymin = np.floor(p_tri[..., 1].min(axis=1)).astype(int)
    ymax = np.ceil(p_tri[..., 1].max(axis=1)).astype(int)
    ok &= (xmax >= 0) & (xmin < W) & (ymax >= 0) & (ymin < H)

    sel = np.where(ok)[0]

    uvs_tri = mesh.uvs[tri]                # (T,3,2)
    nrm_tri = normals_cam[tri]             # (T,3,3)

    for k in sel:
        x0 = max(0, xmin[k]); x1 = min(W - 1, xmax[k])
        y0 = max(0, ymin[k]); y1 = min(H - 1, ymax[k])
        if x1 < x0 or y1 < y0:
            continue

        p0, p1, p2 = p_tri[k]
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(area) < 1e-12:
            continue

        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        PX, PY = np.meshgrid(xs, ys)
        PX = PX + 0.5
        PY = PY + 0.5

        # Barycentric coordinates via edge functions.
        w0 = ((p1[0] - PX) * (p2[1] - PY) - (p2[0] - PX) * (p1[1] - PY)) / area
        w1 = ((p2[0] - PX) * (p0[1] - PY) - (p0[0] - PX) * (p2[1] - PY)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        iz0, iz1, iz2 = 1.0 / z_tri[k]
        inv_z = w0 * iz0 + w1 * iz1 + w2 * iz2
        inv_z = np.where(np.abs(inv_z) < 1e-12, 1e-12, inv_z)
        z = 1.0 / inv_z

        sub = np.s_[y0:y1 + 1, x0:x1 + 1]
        better = inside & (z < depth[sub])
        if not better.any():
            continue

        # Perspective-correct attribute interpolation.
        wz0 = w0 * iz0
        wz1 = w1 * iz1
        wz2 = w2 * iz2

        u = (wz0 * uvs_tri[k, 0, 0] + wz1 * uvs_tri[k, 1, 0] + wz2 * uvs_tri[k, 2, 0]) * z
        v = (wz0 * uvs_tri[k, 0, 1] + wz1 * uvs_tri[k, 1, 1] + wz2 * uvs_tri[k, 2, 1]) * z
        nx = (wz0 * nrm_tri[k, 0, 0] + wz1 * nrm_tri[k, 1, 0] + wz2 * nrm_tri[k, 2, 0]) * z
        ny = (wz0 * nrm_tri[k, 0, 1] + wz1 * nrm_tri[k, 1, 1] + wz2 * nrm_tri[k, 2, 1]) * z
        nz = (wz0 * nrm_tri[k, 0, 2] + wz1 * nrm_tri[k, 1, 2] + wz2 * nrm_tri[k, 2, 2]) * z

        d_sub = depth[sub]; d_sub[better] = z[better].astype(np.float32); depth[sub] = d_sub
        uvb = uv_buf[sub]; uvb[better] = np.stack([u, v], -1)[better]; uv_buf[sub] = uvb
        nb = nrm_buf[sub]; nb[better] = np.stack([nx, ny, nz], -1)[better]; nrm_buf[sub] = nb
        tb = tri_buf[sub]; tb[better] = idx[k]; tri_buf[sub] = tb

    n = np.linalg.norm(nrm_buf, axis=2, keepdims=True)
    nrm_buf = np.divide(nrm_buf, n, out=np.zeros_like(nrm_buf), where=n > 1e-9)

    return Framebuffer(depth=depth, uv=uv_buf, normal=nrm_buf,
                       mask=np.isfinite(depth), tri_id=tri_buf)


def view_directions(fb: Framebuffer, intr: Intrinsics) -> np.ndarray:
    """Unit vector from each covered pixel's surface point toward the camera,
    in camera coordinates. Used for the n·v shading term."""
    H, W = fb.depth.shape
    xs = (np.arange(W)[None, :] - intr.cx) / intr.fx
    ys = (np.arange(H)[:, None] - intr.cy) / intr.fy
    z = np.where(np.isfinite(fb.depth), fb.depth, 1.0)
    pts = np.stack([xs * z, ys * z, z], axis=-1)
    d = -pts
    n = np.linalg.norm(d, axis=2, keepdims=True)
    return np.divide(d, n, out=np.zeros_like(d), where=n > 1e-9)
