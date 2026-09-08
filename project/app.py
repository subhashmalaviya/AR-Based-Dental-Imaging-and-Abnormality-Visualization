"""
app.py
-------
Streamlit UI for the true-3D AR pipeline.

Everything shown here is produced by projecting a *textured 3D mesh*
through an estimated 6-DoF camera pose. The logo is texture at a fixed
(theta, v) on the cup's surface -- it is never placed in 2D.

    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from src.azimuth import unwrap_visualisation
from src.camera3d import Intrinsics, look_at, project_points
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
DEF_IMG = os.path.join(BASE, "data", "cup_image", "ti1.png")
DEF_LOGO = os.path.join(BASE, "data", "logos", "iitd_logo.png")
DEF_VIDEO = os.path.join(BASE, "data", "videos", "tv1.mp4")

st.set_page_config(page_title="3D AR Cup Overlay", layout="wide")
st.title("Markerless 3D AR — logo as texture on a 3D cup")
st.caption(
    "3D mesh + UV texture + 6-DoF pose from the base disc's conic + "
    "perspective projection. Foundation for the dental AR project: swap the "
    "cup mesh for an anatomical model and the logo atlas for abnormality maps."
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("3D cup model (metric)")
r_top = st.sidebar.slider("Body radius (mm)", 20.0, 60.0, 35.0, 0.5) / 1000.0
h_ratio = st.sidebar.slider("Height / radius", 1.5, 4.0, 2.45, 0.05)
taper = st.sidebar.slider("Taper (r_bottom / r_top)", 0.80, 1.25, 1.00, 0.01)
disc_ratio = st.sidebar.slider("Detected disc / body radius", 0.70, 1.05, 1.00, 0.01)

st.sidebar.header("Camera")
focal = st.sidebar.slider("Focal length (px, 1080-wide frame)", 700.0, 3000.0, 1500.0, 10.0)
st.sidebar.caption(
    "Focal length is **not identifiable** from this footage: in the weak-"
    "perspective limit the cup's projected length is height·sin(tilt)·a_px / "
    "r_disc, in which f cancels. It mostly trades against depth, so the "
    "overlay is insensitive to it. See README."
)

st.sidebar.header("Logo on the surface")
auto_theta = st.sidebar.checkbox("Auto-place facing the camera", True)
theta_deg = st.sidebar.slider("Azimuth θ on cup (deg)", 0.0, 360.0, 90.0, 1.0,
                               disabled=auto_theta)
v_center = st.sidebar.slider("Height v along axis (0=table, 1=disc)", 0.05, 0.95, 0.50, 0.01)
size = st.sidebar.slider("Logo size (fraction of cup height)", 0.10, 0.90, 0.42, 0.01)

st.sidebar.header("Print blending")
opacity = st.sidebar.slider("Opacity", 0.0, 1.0, 0.95, 0.01)
blend = st.sidebar.slider("Blend strength (flat ↔ multiply)", 0.0, 1.0, 0.55, 0.01)
tex_str = st.sidebar.slider("Ceramic texture integration", 0.0, 0.3, 0.08, 0.01)
smoothing = st.sidebar.slider("Pose smoothing (video)", 0.05, 1.0, 0.45, 0.05)


def make_cfg(width: int) -> ARConfig:
    cfg = ARConfig()
    cfg.geometry = CupGeometry(r_top=r_top, r_bottom=r_top * taper,
                                height=r_top * h_ratio, r_disc_ratio=disc_ratio)
    cfg.focal = focal * width / 1080.0
    cfg.placement = LogoPlacement(None if auto_theta else theta_deg, v_center, size)
    cfg.opacity, cfg.blend_strength, cfg.texture_strength = opacity, blend, tex_str
    cfg.pose_smoothing = smoothing
    return cfg


def load_bgr(up, fallback):
    if up is None:
        return cv2.imread(fallback)
    return cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)


tab_demo, tab_img, tab_vid = st.tabs(["🧊 3D demo", "📷 Image", "🎬 Video"])

# ------------------------------------------------------------- 3D demo tab
with tab_demo:
    st.write("The mesh and its UV texture alone — no photograph involved. "
             "Orbit the virtual camera to see the logo wrap around the surface.")
    az = st.slider("Camera azimuth (deg)", -180.0, 180.0, 0.0, 2.0)
    elev = st.slider("Camera elevation (deg)", -40.0, 70.0, 18.0, 1.0)
    dist = st.slider("Camera distance (m)", 0.12, 0.60, 0.27, 0.01)

    geom = CupGeometry(r_top=r_top, r_bottom=r_top * taper,
                        height=r_top * h_ratio, r_disc_ratio=disc_ratio)
    mesh = build_cup_mesh(geom, 96, 24, cap_top=True)
    logo = load_logo_rgba(DEF_LOGO)
    atlas = build_atlas(logo, geom, LogoPlacement(theta_deg, v_center, size))

    intr = Intrinsics.from_fov(520, 520, hfov_deg=40.0)
    target = np.array([0.0, geom.height * 0.48, 0.0])
    a, e = np.deg2rad(az), np.deg2rad(elev)
    eye = target + np.array([dist * np.sin(a) * np.cos(e), dist * np.sin(e),
                              dist * np.cos(a) * np.cos(e)])
    fb = render_mesh(mesh, look_at(eye, target), intr)

    import demo_3d
    img = demo_3d.shade(fb, intr, atlas)
    c1, c2 = st.columns([1, 1])
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="textured 3D render")
    uvv = np.zeros((*fb.uv.shape[:2], 3), np.uint8)
    uvv[..., 2] = (fb.uv[..., 0] * 255).astype(np.uint8)
    uvv[..., 1] = (fb.uv[..., 1] * 255).astype(np.uint8)
    uvv[~fb.mask] = 0
    c2.image(uvv, caption="UV coordinates (u→red, v→green)")
    st.image(cv2.cvtColor(atlas[..., :3], cv2.COLOR_BGR2RGB),
             caption="the unrolled (θ, v) surface atlas the logo is painted into")

# --------------------------------------------------------------- image tab
with tab_img:
    up_img = st.file_uploader("Cup photo", type=["png", "jpg", "jpeg"], key="im")
    frame = load_bgr(up_img, DEF_IMG)
    H, W = frame.shape[:2]
    cfg = make_cfg(W)
    intr = Intrinsics(cfg.focal, cfg.focal, W / 2.0, H / 2.0, W, H)

    tracked = track_sequence([frame])
    if tracked[0] is None:
        st.error("Could not detect the cup's base disc in this image.")
    else:
        obs = tracked[0].obs
        cands = [c for c in cup_pose_candidates(obs, intr, cfg.geometry, 0.0)
                 if c.physically_valid]
        if not cands:
            st.error("No physically valid pose (base must be below and behind the disc).")
        else:
            pose = cands[0].pose
            err = reprojection_error(pose, obs, intr, cfg.geometry)
            st.info(f"6-DoF pose recovered — distance "
                    f"{np.linalg.norm(pose.tvec)*100:.1f} cm, "
                    f"disc reprojection error {err:.3f} px")

            pl = cfg.placement
            if pl.theta_center_deg is None:
                pl = LogoPlacement(theta_facing_camera(pose, cfg.geometry, pl.v_center),
                                    pl.v_center, pl.height_frac, pl.opacity)
            mesh = build_cup_mesh(cfg.geometry, cfg.mesh_theta, cfg.mesh_v, True)
            atlas = build_atlas(load_logo_rgba(DEF_LOGO), cfg.geometry, pl)
            fb = render_mesh(mesh, pose, intr, face_subset=face_has_texture(atlas, mesh))
            result = composite_print(frame, fb, atlas, intr, cfg)

            a1, a2 = st.columns(2)
            a1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="A: original")
            a2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="E: 3D AR result")

            with st.expander("Computer-vision stages"):
                wire = frame.copy()
                th = np.linspace(0, 2 * np.pi, 120)
                for vv, col in ((0.0, (255, 60, 0)), (1.0, (0, 220, 255))):
                    uv, z = project_points(
                        cfg.geometry.surface_point(th, np.full_like(th, vv)), pose, intr)
                    if np.all(z > 0):
                        cv2.polylines(wire, [uv.astype(np.int32)], True, col, 3, cv2.LINE_AA)
                b1, b2, b3 = st.columns(3)
                b1.image(cv2.cvtColor(draw_ellipse(frame, obs), cv2.COLOR_BGR2RGB),
                         caption="B: base-disc ellipse (the conic)")
                b2.image(cv2.cvtColor(wire, cv2.COLOR_BGR2RGB),
                         caption="C: 3D model at the estimated pose")
                b3.image(cv2.cvtColor(
                    unwrap_visualisation(frame, pose, cfg.geometry, intr),
                    cv2.COLOR_BGR2RGB), caption="D: surface unwrap (θ, v)")

            ok, buf = cv2.imencode(".png", result)
            st.download_button("Download result", buf.tobytes(), "cup_ar3d.png", "image/png")

# --------------------------------------------------------------- video tab
with tab_vid:
    up_v = st.file_uploader("Video", type=["mp4", "mov", "avi"], key="vid")
    if up_v is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(up_v.getvalue())
        tmp.close()
        vpath = tmp.name
    else:
        vpath = DEF_VIDEO

    n_lim = st.number_input("Limit frames (0 = whole clip)", min_value=0, value=0, step=10)
    v1, v2 = st.columns(2)
    v1.subheader("Original")
    v1.video(vpath)

    if st.button("▶ Track in 3D and render", type="primary"):
        cap = cv2.VideoCapture(vpath)
        ok, probe = cap.read()
        cap.release()
        cfg = make_cfg(probe.shape[1] if ok else 1080)

        bar = st.progress(0.0, text="rendering…")
        out_path = os.path.join(tempfile.gettempdir(), "cup_ar3d_out.mp4")
        stats = process_video(vpath, out_path, DEF_LOGO, cfg=cfg,
                               max_frames=(n_lim or None),
                               progress_cb=lambda i, n: bar.progress(
                                   min(1.0, i / max(n, 1)), text=f"frame {i}/{n}"),
                               verbose=False)
        bar.empty()
        st.success(
            f"{stats['frames']} frames · 6-DoF pose solved on "
            f"{stats['poses_solved']} · logo rendered on {stats['frames_with_logo']} · "
            f"median disc reprojection error {stats['median_reproj_px']:.2e} px")
        with v2:
            st.subheader("3D AR result")
            data = open(out_path, "rb").read()
            st.video(data)
            st.download_button("Export video", data, "cup_ar3d.mp4", "video/mp4")
