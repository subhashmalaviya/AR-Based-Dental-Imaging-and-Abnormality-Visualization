# Markerless 3D AR — logo as a texture on a 3D cup

Foundation module for **"AR Based Dental Imaging and Abnormality
Visualization"**. The IIT Delhi logo is not composited in 2D. It is a UV
texture on a 3D mesh, and every pixel you see is that mesh projected
through an estimated 6-DoF camera pose:

```
IITD logo PNG
   -> RGBA texture (alpha from the white background)
   -> painted into the cup's unrolled (theta, v) surface atlas
   -> UV-mapped onto a 3D surface-of-revolution mesh
   -> 6-DoF pose estimated per frame from the base disc's image conic
   -> perspective projection + z-buffer + back-face culling
   -> composited into the real frame as ink on ceramic
```

The logo's position lives in **cup coordinates** (a fixed azimuth `theta`
and height `v`). Nothing about it is chosen per frame; only the camera
pose changes.

---

## 1. Quick start

```bash
cd project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo_3d.py                 # 1) standalone 3D demo, no footage
python run_ar3d.py image          # 2) the supplied photo
python run_ar3d.py video          # 3) the supplied video (~2 min)
python export_mesh.py             # 4) textured OBJ for Open3D/MeshLab/Blender
streamlit run app.py              # 5) interactive UI
```

Verify the core geometry maths independently:

```bash
python -m src.pose_from_ellipse   # synthetic ground-truth self-test
```

---

## 2. First demonstration (`demo_3d.py`)

Builds the mesh, applies the logo as a UV texture, and orbits a virtual
camera — no photograph involved.

| output | what it shows |
| --- | --- |
| `outputs/demo3d_turntable.png` | half-orbit contact sheet: face-on at 0°, compressed toward the silhouette at ±30/±60°, wrapped out of sight by ±90° |
| `outputs/demo3d_buffers.png` | depth (z-buffer), surface normals, UV coordinates, textured render |
| `outputs/demo3d_atlas.png` | the unrolled (θ, v) atlas the logo is painted into |
| `outputs/demo3d_turntable.mp4` | continuous 360° turntable |

The wrap is produced entirely by the geometry — there is no 2D distortion
step anywhere in the code.

---

## 3. The 3D cup model (`src/cup_model_3d.py`)

Surface of revolution, exactly the requested parametrisation:

```
R(v) = r_bottom + v * (r_top - r_bottom)          v in [0, 1]
X = R(v) sin(theta),   Y = v * height,   Z = R(v) cos(theta)
```

with the analytic outward normal (from `dP/dtheta × dP/dv`)

```
n(theta, v)  ∝  ( height·sin(theta),  -(r_top - r_bottom),  height·cos(theta) )
```

Cup frame: origin at the centre of the circle resting on the table, **+Y**
up the axis, and the mug is filmed standing upside-down so the bright disc
we detect is the circle at `Y = height`. The theta seam is duplicated so UV
interpolation never wraps across the atlas.

Fitted values for this mug: **r = 3.5 cm, height = 8.6 cm** (h/r ≈ 2.45),
essentially a straight cylinder. How those were obtained is in §6.

---

## 4. Rendering (`src/renderer3d.py`)

A small software rasteriser: z-buffer, back-face culling, and
**perspective-correct** attribute interpolation (attributes carried as
`attr/z` alongside `1/z` and divided at the end — plain screen-space
interpolation visibly skews a texture at the grazing angles that dominate
near the silhouette).

It is written out rather than calling Open3D's offscreen renderer because
(a) it runs headless with no EGL/OpenGL dependency, (b) every stage of the
3D→2D pipeline stays inspectable, and (c) it returns the depth/normal/UV
buffers the compositor needs, which a screenshot API does not. `export_mesh.py`
still writes a standard textured OBJ so the same mesh opens in Open3D:

```bash
python export_mesh.py
python -c "import open3d as o3d; m=o3d.io.read_triangle_mesh('outputs/cup_textured.obj', True); o3d.visualization.draw_geometries([m])"
```

Visibility: the cup is convex in cross-section, so back-face culling alone
resolves front/back wall correctly — the logo genuinely disappears around
the far side rather than bleeding through. The z-buffer additionally
handles the end cap.

---

## 5. 6-DoF pose from the base disc's conic (`src/pose_from_ellipse.py`)

The mug's base is a **real circle in 3D**, so its image is an ellipse, and
a circle's pose is recoverable from its image conic in closed form. No
homography, no planarity assumption.

Back-project the ellipse to a cone, `Q = Kᵀ Q_img K`, and diagonalise
`Q = V diag(λ1,λ2,λ3) Vᵀ` with `λ1 ≥ λ2 > 0 > λ3`. Subtracting
`λ2(x²+y²+z²)` from the cone equation gives `(λ1−λ2)x² = (λ2−λ3)z²`, i.e.
two planes on which every cone point also lies on a sphere about the
origin — the circular sections. Hence, in the eigenbasis,

```
n = (g, 0, ±h),   g = √((λ1−λ2)/(λ1−λ3)),  h = √((λ2−λ3)/(λ1−λ3)),  g²+h² = 1
|d| = R·λ2 / √(−λ1λ3)
C   = d·( g − k·h,  0,  ±(h + k·g) ),      k = √((λ1−λ2)(λ2−λ3)) / λ2
```

**Verified against synthetic ground truth** (`python -m src.pose_from_ellipse`),
200 random poses:

```
normal error   median 4.0e-06 deg,  p95 1.3e-05 deg
centre error   median 3.1e-08 rel,  p95 6.7e-08 rel
```

i.e. exact to machine precision. On the real footage the disc reprojects
onto its detected ellipse at a median residual of ~1e-13 px.

### Resolving the two-fold ambiguity

Every ellipse admits two circle poses (plus a sign for which way the body
extends). The camera sees the mug's top face in every frame, so it is
above the disc's plane, and therefore the base must be **both** lower in
the image **and** farther from the camera. Each condition *alone* is
satisfied by a wrong branch as well — of frame 0's four candidates one is
below-but-nearer and another is farther-but-above — so testing only one
silently selects a mirrored pose, which is exactly the bug that made the
first overlays flip between frames.

---

## 6. What is and is not identifiable from this footage

**Scale is not.** A cup twice as large at twice the distance is
pixel-identical. `r_top` is therefore fixed as the scale gauge and
everything else is a dimensionless ratio.

**Focal length is not, either.** This is worth stating precisely because
it was not obvious. The disc ellipse alone constrains nothing about `f`:
for any `f` there is a circle pose reproducing the ellipse exactly. And in
the weak-perspective limit the cup's projected length is

```
L_px  ≈  height · sin(tilt) · a_px / r_disc
```

in which **`f` cancels**. Only second-order perspective would pin it down,
and that signal is far too weak here. The measured objective landscape
agreed: silhouette agreement stayed within noise from f = 900 to 3200 px.
So `f` is set from a documented prior (≈1500 px for a 1080-wide portrait
frame — note the ~65° figure quoted for phone cameras refers to the *long*
side; applying it to the short side gives ≈816 px, which is wrong by nearly
2× and puts the cup so close that its far end falls behind the camera) and
exposed as a UI slider. The overlay is insensitive to it because `f` mostly
trades against depth.

**The aspect ratio is identifiable**, and was measured directly. From the
ellipse's semi-axes, `cos(tilt) = b/a`; combined with the observed cup
length this gives `height/r_disc = (L_px/a_px)/sin(tilt)`. Over well-
conditioned frames:

```
h/r = 2.47, 2.46, 2.41, 2.38, 2.40, 2.50   ->  median 2.45
```

a tight, consistent estimate, and the resulting model length matches the
observed length closely (426 vs 430 px, 547 vs 556, 598 vs 619).

An automatic silhouette/edge fit for these parameters is implemented
(`src/calibrate3d.py`, `src/model_fit3d.py`) and both are honest failures
worth recording: the GrabCut-region objective peaked at IoU 0.36 with every
parameter pinned to a grid bound (the segmentation variously swallowed the
handle, leaked onto the table, or missed half the cup), and the edge-
alignment objective was gamed by the mug's own printed flowers, which
offer more gradient than the true base contour. The measurement above is
what the delivered defaults come from.

Practically this matches the dental workflow: **the object model is given**
(there, from a scan), and only the pose is estimated per frame.

---

## 7. Finding the disc: detect once, then track (`src/ellipse_detection.py`, `src/disc_tracker.py`)

Single-frame detection of this rim is genuinely hard, and the failure
modes drove the design:

* Brightness alone confuses disc with table (frame 0: disc V=226, table
  V=156, wall V=105).
* Brightness + low saturation separates disc from table and handle, but
  the white ceramic **wall** is also bright and unsaturated — a threshold
  tuned on frame 0 flooded to 20–40% of the image on frames 80/120/160.
* Fitting the thresholded blob's outline fails once that blob merges disc
  and wall (frame 80: fill 0.71, ellipse score 0.12).
* Global RANSAC over Canny edges recovers frame 0 well (score 0.98) but
  without geometric gates returns enormous near-straight ellipses threaded
  through unrelated edges (centres hundreds of px off-screen, axes >3000 px).

The physical rim is always a strong *intensity edge* and barely moves
between frames, so detection only bootstraps. Per frame the tracker tries,
in increasing cost: a local re-fit in a narrow band around a
constant-velocity prediction; the same with a wider band; a full RANSAC
re-detection; then coasting. Guards that proved necessary:

* **Minimum support.** Without it the tracker "tracks" noise — frames
  89–128 marched linearly across the image on 1–8 inlier edge points while
  reporting success. Real measurements carry 1300–2200 inliers at
  0.75–1.0 angular coverage.
* **Angular coverage**, the single most effective guard against degenerate
  RANSAC fits: a real rim is sampled all the way round, a spurious giant
  ellipse only grazes one short arc.
* **Prior-anchored IRLS.** Plain IRLS diverges: on a motion-blurred frame
  a 523×880 prior collapsed to 86×227 in one iteration.
* **Shape-aware plausibility.** Mean radius alone is not enough — a
  degenerate slit can match it (how frame 160 first slipped through), so
  area and elongation are checked too.
* **No axis extrapolation** in the motion prediction. Extrapolating size
  creates positive feedback: a slightly-large prediction pushes the band
  outward, the fit lands on the outer edge, and it grows again — a
  perfectly linear runaway from frame 234 to a 1297 px radius, larger than
  the 1080 px frame.
* **Bidirectional passes.** A motion-blurred passage blocks the tracker
  one way but is usually traversable from the other; a single forward pass
  left frames 110–143 unrecoverable.

Result on `tv1.mp4`: 206/293 frames measured directly, the rest
interpolated between confident neighbours and flagged; per-frame centre
step median 8.4 px / max 56 px, radius step median 1.9 px / max 27 px
(before these guards: max 672 px and 166 px).

---

## 8. The sixth DoF: azimuth from surface texture (`src/azimuth.py`)

A surface of revolution is invariant to rotation about its own axis, so
the conic cannot supply that DoF — yet it is exactly the one that decides
whether the logo stays on the same *patch of ceramic*.

It is recovered from the mug's own printed decoration. Using the known
5-DoF pose, each frame is back-projected onto the cup's (θ, v) surface —
an "unwrap" of the visible ceramic — and aligned to a reference unwrap.
Because a change of azimuth is **exactly a circular shift along θ**, the
alignment is a 1-D circular cross-correlation (FFT), with no local minima.
The unwraps recover the mug's printed text and crown emblem at consistent
θ across the clip, which is the visual confirmation that it works.

---

## 9. Temporal stability (`src/video_ar3d.py`)

Rotations are smoothed as **quaternions** with hemisphere alignment, not
by filtering Rodrigues vectors (which break at the ±π wrap and are not a
linear space); translation is EMA'd. Azimuth is unwrapped and
median-filtered against jumps. `pose_smoothing` is exposed in the UI.

---

## 10. Blending (`src/video_ar3d.composite_print`)

```
shading  = 0.35 + 0.65 · (n · v)          # n from the 3D rasteriser
ink      = logo_rgb · shading · (0.5 + 0.5 · frame_luma)
printed  = lerp(ink, ink ⊙ frame, blend_strength) + high_freq · texture_strength
out      = lerp(frame, printed, alpha · opacity)
```

The falloff toward the silhouette comes from the real surface normal, not
a 2D approximation. Frame luminance modulates the ink so the ceramic's
existing highlights and shadows read through, and a high-frequency term
keeps its micro-texture from being flattened.

---

## 11. Project structure

```
project/
├── data/{cup_image,logos,videos}/
├── src/
│   ├── camera3d.py            # intrinsics, extrinsics, projection, look_at
│   ├── cup_model_3d.py        # parametric surface of revolution -> mesh + UVs
│   ├── texture_atlas.py       # logo alpha + the unrolled (θ,v) atlas
│   ├── renderer3d.py          # z-buffer rasteriser, perspective-correct UVs
│   ├── pose_from_ellipse.py   # circle pose from conic (+ self-test)
│   ├── ellipse_detection.py   # disc detection / band-refit
│   ├── disc_tracker.py        # bidirectional sequence tracking
│   ├── pose_estimation3d.py   # conic -> 6-DoF cup pose, ambiguity resolution
│   ├── azimuth.py             # surface unwrap + 1-D circular alignment
│   ├── video_ar3d.py          # full pipeline, smoothing, compositing
│   ├── calibrate3d.py         # silhouette-IoU fit (documented failure, §6)
│   └── model_fit3d.py         # edge-alignment fit (documented failure, §6)
├── demo_3d.py    run_ar3d.py    export_mesh.py    app.py
└── requirements.txt   README.md
```

`src/{cup_detection,surface_estimation,cylindrical_mapping,blending,tracking,video_ar,logo_processing,image_loader,visualization}.py`
are the earlier **2D image-space** pipeline, kept only as a baseline for
comparison. They are not part of the 3D system.

---

## 12. Second test object: `tv2.mp4` (steel bottle) — an instructive failure

`tv2.mp4` (480×864, 146 frames) is a **Milton stainless-steel bottle**, not
the mug. Running the pipeline unchanged:

```
forward 0/146 confident, backward 0/146, merged 0/146
poses on 0/146 frames
frames_with_logo: 0
```

It fails, and it fails *cleanly* — the support gates from §7 rejected every
candidate rather than inventing a pose, so the output is a pass-through
rather than a hallucinated overlay. That is the intended behaviour, but the
failure is total. Why:

1. **The pose observable does not exist.** The whole 6-DoF stage is built on
   the conic of a visible circular face. The bottle's mouth is cropped out of
   frame in every shot and its base sits on the table, so *no circular face is
   ever visible*. This is a missing measurement, not a tuning problem.
2. **The photometric cue is inverted.** Disc detection keys on
   "bright + unsaturated" ceramic. Specular steel is mid-grey and reflects the
   room, so the ROI skips the bottle entirely and latches onto the white jar
   lid and panda toy behind it (`outputs/tv2_failure_diagnosis.png`). Those
   false positives scored 0.26–0.35 against 0.98–1.00 on `tv1.mp4` — the score
   separation is exactly what let the gates reject them.
3. **Silhouette segmentation is not a fallback here.** Colour-based
   segmentation of specular steel against a cluttered desk covers 43–68% of
   the frame, bleeding into table and background.

**What still works.** The bottle is a surface of revolution, so
`src/sor_model.py` generalises the mesh layer from a linear frustum to an
arbitrary sampled radius profile r(v) — rounded base, ridged band, straight
body, tapering shoulder — and emits the same `Mesh`. The renderer, texture
atlas, compositor and azimuth stages consume it **unchanged**:

```bash
python -c "from src.sor_model import milton_bottle_profile, build_sor_mesh; \
           print(build_sor_mesh(milton_bottle_profile()).n_faces, 'faces')"
```

`outputs/demo3d_bottle.png` shows the logo UV-mapped onto that bottle profile
and wrapping correctly across a half-orbit. So the split is clean: **the
mesh/UV/render/composite half is object-agnostic; only the pose half is
mug-specific.**

**To make `tv2.mp4` work** you would replace the pose source, not the
renderer. Reasonable options, in order of effort:

* the **red MILTON sticker** — a high-contrast surface landmark that is
  visible from ~frame 25 onward and could drive both azimuth and a
  planar-patch pose;
* **occluding-contour fitting** for a surface of revolution: the bottle's two
  near-vertical silhouette edges plus its known profile constrain axis,
  scale and position — but extracted from *gradients*, not colour;
* the general answer, and the one the dental case needs anyway: `solvePnP`
  against known 3D landmarks, or ICP against a depth stream.

This test is the clearest evidence for the design claim in §13: the pose
observable must be chosen per object class, and it is the only part that
does not transfer.

---

## 13. Extending to the dental project

The cup-specific code is confined to two files. Everything else is written
against generic interfaces:

* Replace `cup_model_3d.py` with the anatomy: load a scanned tooth/arch mesh
  with per-vertex UVs. `renderer3d.py` already consumes an arbitrary
  `Mesh(vertices, faces, uvs, normals)` — no changes needed.
* Replace `texture_atlas.py`'s logo painting with abnormality maps
  (caries, lesions, plaque probability) rendered into the same UV space.
  Downstream code does not care what the atlas contains.
* Replace `pose_from_ellipse.py` + `ellipse_detection.py` with a pose
  source suited to anatomy — `cv2.solvePnP` against landmarks, ICP against
  a depth stream from an intraoral scanner, or a learned pose regressor.
  `video_ar3d.py` only needs a `Pose` per frame.
* The tracker's structure (bootstrap → local refine → re-detect → coast,
  with support gates, plausibility checks and bidirectional passes) carries
  over unchanged; only the measurement changes.

For clinically meaningful registration you would add a **calibrated**
camera (a checkerboard removes both the scale gauge and the focal-length
ambiguity discussed in §6) and, ideally, a depth or stereo intraoral
scanner — which is what makes absolute scale observable, unlike here.

---

## 14. Known limitations

* Focal length and absolute scale are not recoverable from this footage
  (§6); both are parameters, not measurements.
* The cup's dimensions were measured semi-automatically (§6), not fitted
  fully automatically — the two automatic fitters that were attempted are
  included and their failure modes documented rather than hidden.
* 87 of 293 frames have interpolated rather than directly measured disc
  ellipses; those carry more pose error, concentrated in the
  motion-blurred passage around frames 110–143.
* The handle is not in the 3D model, so it cannot occlude the logo. Cup
  self-occlusion (the far wall) *is* handled correctly by back-face
  culling and the z-buffer.
* Rendering is CPU-only: ~0.3 s/frame at 1080×1920, fine for offline
  export, not real-time.
