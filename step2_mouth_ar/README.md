# Steps 2–3 — Face & mouth tracking + real-time tooth detection

Second stage of **"AR Based Dental Imaging and Abnormality Visualization"**.

Step 1 tracked a cylindrical cup and printed a logo onto its 3D surface. Step 2
does the same thing for a **person's mouth**: track the face live from the
device camera, build a stable coordinate system attached to the mouth, and keep
an AR overlay glued to it while the head moves.

It follows the same principle that made Step 1 work:

> **Content is defined once in the object's own coordinates.
> Only the pose changes per frame.**

Nothing the overlay draws is positioned in screen space.

**Step 3** adds real-time detection and tracking of the user's actual visible
teeth, inside the mouth ROI, from the live camera — see §13 onwards.
**Step 3b** gives every tracked tooth its own 3D anchor (§21–27).
**Step 3 v2** replaces the hand-tuned tooth detector with a trained model,
upgrades tracking, and adds local video recording with per-frame metadata and
an evaluation tool — with every accuracy number measured against ground truth
(§28 onwards).

**Scope:** tracking, detection and AR alignment only. No abnormality detection,
no diagnosis, no CT/CBCT registration, no dental image input — those are Step 4,
and §18 describes the seams left for them.

---

## 1. What this is built on, and why

| Choice | Why |
| --- | --- |
| **MediaPipe Face Landmarker** (Tasks Vision, WASM + GPU) | 478 3D landmarks + a 4×4 head matrix at phone frame rates. Runs on Android **and** iOS from one codebase. |
| **Web app** (served over HTTPS, optionally wrapped with Capacitor) | One implementation covers Android phone, iPhone and iPad. A native Android app would need a second iOS implementation for no gain at this stage. Capacitor wraps the *same* code into a real `.apk`/`.ipa` when you want one (§8). |
| **Model + WASM vendored locally** (`public/`) | No CDN at runtime, so it works offline and inside a WebView — a CDN fetch from a Capacitor WKWebView is exactly the kind of thing that fails on a demo day. |
| **Canvas 2D overlay** | Enough for a quad; keeps the render path inspectable. Step 3's 3D dental model wants WebGL/three.js, and `MouthARAnchor.getMatrix()` already hands it a model matrix. |

---

## 2. Quick start

```bash
cd step2_mouth_ar
npm install
npm run vendor:wasm     # already done, re-run if node_modules is rebuilt
npm test                # all unit tests (Steps 2, 3, 3b, 3 v2), no browser needed
npm run dev             # http://localhost:5173  (laptop webcam)
```

To use a **phone**, see §7 — the camera will not start over plain `http://` on
a LAN address, and that is the single most common way this appears "broken".

---

## 3. Architecture

```
CameraManager ──▶ FaceTracker ──▶ MouthTracker ──▶ MouthARAnchor
   (video)        (landmarks)     (mouth geom)     (smoothed pose)
                                                          │
                                    LandmarkRenderer ◀────┤
                                    AROverlayController ◀──┘
```

```
step2_mouth_ar/
├── index.html                     demo screen
├── serve_https.py                 HTTPS server for phone testing (stdlib only)
├── capacitor.config.json          native Android/iOS wrap
├── vite.config.js
├── public/
│   ├── models/face_landmarker.task    3.7 MB, vendored
│   ├── wasm/                          MediaPipe runtime, vendored
│   └── iitd_logo.png                  overlay artwork (same asset as Step 1)
├── src/
│   ├── main.js                    wiring + render loop
│   ├── styles.css
│   ├── core/
│   │   ├── CameraManager.js       getUserMedia, front/back, FPS
│   │   ├── FaceTracker.js         ← face tracking happens here
│   │   ├── MouthTracker.js        lips, corners, opening, bbox, 3D frame
│   │   ├── MouthARAnchor.js       ← the AR coordinate system + gating
│   │   ├── AROverlayController.js warped-quad AR content
│   │   ├── filters/OneEuroFilter.js
│   │   ├── filters/PoseSmoother.js
│   │   └── math/vec3.js
│   ├── ui/{LandmarkRenderer,HUD}.js
│   └── landmarks/FaceLandmarkIndices.js
└── tests/math.test.mjs            headless verification
```

Each requested class exists as its own module, so Step 3 can add a
`ToothDetector`, `DentalRegistration` or `LesionOverlay` beside them without
touching tracking.

---

## 4. Where the face tracking happens

**`src/core/FaceTracker.js`**, via MediaPipe Face Landmarker in `VIDEO` running
mode on the GPU delegate (automatic CPU fallback).

Two things worth knowing:

* `VIDEO` mode is **not** an image detector in a loop. MediaPipe keeps state
  between timestamps and runs its own detect-then-track pipeline internally: a
  detector finds the face once, a lighter landmark model follows it, and
  re-detection only fires when tracking confidence drops. That is what makes it
  fast enough for a phone — and why timestamps must strictly increase, which
  `FaceTracker.detect()` enforces.
* Per frame it returns **478 landmarks** (`x`,`y` normalised to the frame,
  `z` a relative depth) plus a **4×4 facial transformation matrix**.
  `FaceTracker.headPoseFromMatrix()` decomposes that into yaw/pitch/roll and a
  translation for the HUD and for Step 3.

Everything downstream consumes only the plain landmark array, so swapping in
ARKit/ARCore face tracking means rewriting this one file.

---

## 5. How the mouth anchor is calculated

**`MouthTracker`** (`src/core/MouthTracker.js`) converts landmarks to pixels and
measures the mouth. Landmark `x`/`z` are scaled by frame **width** and `y` by
frame **height** — scaling everything by width would skew the frame on a
non-square aspect ratio.

**The frame** (Gram–Schmidt, so it is exactly orthonormal):

```
+X  = normalize(rightCorner − leftCorner)      landmarks 291 and 61
+Y  = normalize(lowerLip − upperLip, orthogonalised against X)
+Z  = X × Y                                    outward facial normal
origin = centroid of the 20-point outer lip ring
scale  = |rightCorner − leftCorner|            the mouth width
```

The origin is the **ring centroid, not the midpoint of the two corners**:
averaging 20 landmarks instead of 2 measurably reduces the jitter a single noisy
point would inject into the anchor.

**Mouth opening** is reported two ways, both divided by mouth width so they are
invariant to how close the face is — verified by test:

* `ratio` — inner-lip gap ÷ mouth width (direct "how far open")
* `areaRatio` — inner-ring area ÷ width² (survives partial lip occlusion)

### The coordinate contract (what Step 3 registers against)

**`MouthARAnchor`** exposes this, and it is the deliberate parallel to Step 1's
cup frame:

| | |
| --- | --- |
| origin | centroid of the outer lip ring |
| +X | towards the subject's right mouth corner |
| +Y | downwards, towards the chin |
| +Z | outwards from the face, towards the camera |
| **unit** | **1.0 = mouth width (corner to corner)** |

So `(±0.5, 0, 0)` are the mouth corners and the teeth sit at slightly negative
Z. Because the unit is the mouth width, **content authored in this frame keeps
the correct size automatically** as the subject moves nearer or further away.

API:

```js
anchor.localToScreen({x, y, z})   // mouth-local → pixels
anchor.screenToLocal({x, y})      // inverse, for picking
anchor.getMatrix()                // column-major 4×4 for WebGL/three.js
anchor.getPose()                  // {origin, basis, quaternion, scale, euler, mouthOpen}
```

Projection is **weak perspective** (place in pixel space, drop z). That is a
deliberate choice, not an oversight: MediaPipe's landmark `z` is a *relative*
depth, not metric, so a full projective model would be false precision, and over
a region as small as a mouth the orthographic approximation is both accurate and
much steadier. `getMatrix()` still exposes a full 4×4 so Step 3 can swap in a
metric camera without changing a single caller.

---

## 6. How smoothing works

Two independent mechanisms, because they solve different problems.

### One-euro filtering (`filters/OneEuroFilter.js`)

A fixed EMA forces one trade-off between jitter and lag: smooth enough to hold
still, and the overlay visibly drags behind a head turn. The **1-euro filter**
adapts its cutoff to the signal's own speed — heavy smoothing when slow, light
when fast:

```
cutoff = minCutoff + beta · |estimated speed|
```

Three presets in the UI (`responsive` / `balanced` / `smooth`). `dt` is clamped
to `[1/240, 0.2]` s so a backgrounded tab or a dropped frame cannot make the
filter degenerate into a pass-through.

`PoseSmoother` filters position, scale, mouth-opening and **orientation as a
quaternion with hemisphere alignment** — never Euler angles, which are not a
linear space and corrupt every wrap through ±180°. (`q` and `−q` are the same
rotation, so the sign is flipped into the previous hemisphere before filtering;
a test asserts a sign flip leaves the orientation unchanged.)

### Outlier gating (`MouthARAnchor._isPlausible`)

The 1-euro filter deliberately opens up on high velocity, which means it will
pass a single wild measurement straight through — it is a motion tracker, not an
outlier rejector. So rejection is a separate step, mirroring the plausibility
gate that proved necessary in Step 1:

* origin may move < **0.8 mouth-widths** per frame
* scale may change by < **1.7×** per frame
* violations hold the previous pose…
* …but **only for 6 frames**. If the "outlier" persists, the subject really did
  move (or the detector re-acquired a different face), so the anchor re-anchors.
  Recovery must never be gated on the state that went stale — the same lesson
  that cost 203 of 293 frames in the Step-1 tracker.

Brief detection dropouts (a blink, motion blur) coast on the last pose for 5
frames before the anchor reports lost, so the overlay does not strobe.

---

## 7. Testing on a real phone or tablet

**The camera needs a secure context.** `http://localhost` qualifies;
`http://192.168.x.x` does **not**, and the failure is silent. The app detects
this and shows a red banner rather than hanging.

### Option A — deploy to Vercel (easiest, works on Android *and* iOS/iPadOS)

This repo is a monorepo — the web app lives in `step2_mouth_ar/`, not the repo
root — so a plain "Import Git Repository" on Vercel builds nothing at the root
and 404s. The root-level **`vercel.json`** fixes that automatically:

```json
{
  "installCommand": "cd step2_mouth_ar && npm install",
  "buildCommand": "cd step2_mouth_ar && npm run build",
  "outputDirectory": "step2_mouth_ar/dist"
}
```

`npm install` there also fires the `postinstall` script (§2), so the MediaPipe
WASM runtime and face model are fetched automatically during the Vercel build
— no manual steps, no dashboard "Root Directory" configuration needed. Once
deployed, Vercel serves it over **HTTPS with a real certificate**, which sidesteps
the self-signed-certificate problem below entirely — this is the easiest way to
test on an iPhone/iPad, since Safari usually refuses self-signed certs outright.
Just open the `https://<project>.vercel.app` URL directly on the device.

### Option B — HTTPS on your own machine, over LAN

```bash
npm run build
python3 serve_https.py            # prints https://<your-lan-ip>:8443/
```

Open that URL on the phone (same Wi-Fi) → *Advanced → Proceed* past the
self-signed warning → **Start** → allow camera.

Or skip certificates entirely with Chrome's origin allowlist:
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add
`http://<lan-ip>:5173` → relaunch, then `npm run dev`.

### Option C — iPhone/iPad without deploying (a tunnel, not the cert)

iOS Safari usually **refuses** self-signed certificates, which is why Option A
(Vercel) is the easiest iOS path. To test local, uncommitted changes without
deploying, use a tunnel instead, which gives a real certificate:

```bash
npm run build && npx vite preview --host     # terminal 1
cloudflared tunnel --url http://localhost:4173   # terminal 2 → https://xyz.trycloudflare.com
```

`npm run dev:https` also works (Vite + `basic-ssl`) where the cert is accepted.

### What to check

| Test | Expected |
| --- | --- |
| Face the camera | *Face detected* + *Mouth tracked* + *Anchor locked*, logo sits on the mouth |
| Move left / right / up / down | overlay tracks, no lag or swimming |
| Move closer / further | overlay **scales with the mouth** (the mouth-width unit) |
| Tilt head sideways | overlay rolls with the head |
| Turn head left / right | overlay foreshortens (warped quad, not a flat sprite) |
| Open / close mouth | opening meter responds; overlay stays attached |
| Cover face briefly | *Anchor holding* for ~5 frames, then *Anchor lost*; re-acquires cleanly |
| Toggle landmarks off | outline and axes vanish, overlay stays |
| Watch FPS | 20–30+ on a recent phone (GPU backend) |

---

## 8. Native Android / iOS build (Capacitor)

The web app is the reference implementation; Capacitor wraps the *same* build
into a real installable app.

```bash
npm i @capacitor/core @capacitor/cli @capacitor/android @capacitor/ios
npx cap init --web-dir dist        # config already committed
npm run build
npx cap add android                # and/or: npx cap add ios
npm run cap:android                # build + sync + open Android Studio
```

**Android** — add to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="false" />
```

Then Android Studio → Run. Requires Android 7+ / WebView 60+.

**iOS** — add to `ios/App/App/Info.plist`, or the app is killed on first camera
access with no useful message:

```xml
<key>NSCameraUsageDescription</key>
<string>Used to track your mouth for the dental AR prototype.</string>
```

Then `npm run cap:ios` → Xcode → select your team → Run on a device.
Requires **iOS 14.3+** (`getUserMedia` inside `WKWebView` landed there) and a
physical device — the simulator has no camera.

---

## 9. Verification

`npm test` — **18 tests, all passing**, no browser or camera needed. A canonical
mouth is synthesised at a known pose, pushed through `MouthTracker` →
`MouthARAnchor`, and the recovered pose compared to ground truth:

* roll/yaw/pitch recovered to **< 2.5°**
* mouth corners reproject to **< 4 px**
* mouth-opening ratio proven invariant to camera distance
* a 3× closer face gives exactly 3× projected size (ratio error < 0.05)
* 1-euro cuts stationary jitter to < 35 % of raw, with less lag than a plain EMA
* quaternion sign flip leaves orientation unchanged
* gating rejects a teleport, re-anchors when it persists, and **never fires on
  normal fast motion**

Writing these caught two real bugs: `basisToEuler` was reading the transposed
matrix and returned **every angle negated** (yaw +20° reported as −20°), and a
test that asserted the 1-euro filter should reject outliers was itself wrong
about what the filter is for — which is what prompted moving rejection into the
anchor as an explicit gate.

The pipeline was also driven in a real headless Chrome with a fake camera
device: MediaPipe initialised on the **GPU** delegate, the loop ran with no JS
errors, and a synthetic face injected through the live in-page modules recovered
roll **exactly 12.00°**, placed the anchor at exactly the synthesised origin, and
painted **82,829 overlay pixels** in a bounding box centred on the mouth.

Face detection accuracy itself is MediaPipe's, and needs a real face — that is
what the §7 device checklist is for.

---

## 10. What Step 3 plugs into

The seams are already cut:

* **A dental image / texture** → replace the image in `AROverlayController`, or
  paint into a mouth-space texture. This is the direct analogue of Step 1's
  UV atlas: author once in object coordinates, project per frame.
* **A tooth segmentation mask** → express it in mouth-local coordinates and use
  `anchor.localToScreen()`. `screenToLocal()` gives you tap-to-select on a tooth.
* **A 3D dental model** → `anchor.getMatrix()` returns a column-major 4×4 ready
  for a three.js `Object3D.matrix`. Swap the Canvas 2D overlay for a WebGL layer
  sharing the same anchor; nothing else changes.
* **Metric registration** — the one real upgrade needed. Today's scale gauge is
  the mouth width in pixels, which is *relative*. For clinically meaningful
  sizing, calibrate the camera and use MediaPipe's 4×4 matrix (already surfaced
  by `FaceTracker.headPoseFromMatrix`) against a metric face model, or take
  depth from an iPhone TrueDepth / LiDAR sensor.
* **Teeth are not lips.** This step anchors to the lip ring, which is the stable,
  always-visible structure. Registering *teeth* will need the mouth open and its
  own landmarks — the anchor gives Step 3 a stabilised, normalised region to run
  that detection inside, rather than searching the whole frame.

---

## 11. Known limitations

* Weak-perspective projection (§5). Fine for a mouth-sized region; a full
  projective model needs a calibrated camera.
* Scale is relative (mouth widths), not metric — no true depth from a single
  RGB camera.
* One face at a time (`numFaces: 1`); raise it in `FaceTracker` if needed.
* Landmark accuracy degrades with heavy motion blur, very low light, or head
  rotation beyond roughly ±45°, where the far mouth corner is self-occluded.
* Face masks / hands over the mouth break lip landmarks — the anchor coasts,
  then reports lost, which is the intended behaviour rather than a silent lie.

---
---

# STEP 3 — Real-time tooth detection & tracking

Adds detection and tracking of the user's **actual visible teeth** from the live
camera, inside the mouth ROI that Step 2 already provides.

```
LIVE CAMERA → FACE → MOUTH → MOUTH ROI → TOOTH SEGMENTATION
           → INDIVIDUAL TEETH → STABLE TRACKING → AR VISUALISATION
```

## 13. What the detector is — read this first

**No neural network is used, and the UI says so on screen.**

The publicly available tooth-segmentation models (railNet, DENTEX and similar)
are trained on **CBCT volumes and panoramic radiographs**, not on RGB photos of
a mouth from a phone camera, and none ship in a browser-runnable form. Running
an X-ray model on a selfie is a domain mismatch that produces noise. So this
step does **not** pretend to run one.

Instead `ToothSegmenter.js` is a **classical computer-vision segmenter** that
measures real pixels from the live camera. It is a genuine CV method — no
hardcoded coordinates, no predefined rectangles, no canned results, no
prerecorded video — but it is explicitly not an AI model, and the panel reads
*"Classical CV (whiteness + arch split) — not a neural network"* so nothing here
can be mistaken for an AI result.

`ToothDetector.js` is the seam: implement `detect()` and a learned model drops
in without touching ROI, tracking, smoothing or rendering.

## 14. How tooth detection works

Developed and measured against real footage (`mouthtestvideo.mp4`) with
`tools/prototype_tooth_seg.py` before being ported to JS.

1. **Rectify the ROI.** `MouthROI` resamples the mouth into a canonical,
   roll-free image using the Step-2 anchor basis — one affine `drawImage`, so
   the compositor does the work. This matters for *detection*, not tidiness:
   once head roll is removed the dental arches are horizontal and interdental
   gaps are vertical. It is the same trick as Step 1's cup unwrap.
   The window is sized from the inner lip ring's own extent plus 16 % padding,
   so it grows as the mouth opens. (A fixed window clipped the teeth outright.)

2. **whiteness = V · (1 − S/255).** Inside an open mouth every competing surface
   is strongly coloured — lips and gums pink, tongue red, cavity near-black — so
   enamel is the only bright *neutral* surface. Plain brightness fails: a lit
   lower lip and the tongue are both bright.

3. **Adaptive threshold** at the 62nd percentile of the aperture's *own*
   whiteness histogram, so it tracks lighting. Otsu was tried first and sat far
   too low — it kept 57 % of the aperture and swallowed the tongue, because the
   distribution is not cleanly bimodal. Two floors are applied: an absolute one,
   and a **relative** one at 0.52 × the aperture's p99, because enamel is the
   brightest neutral surface present — without it a big red tongue gets promoted
   to a "lower arch" when no lower teeth are visible.

4. **Arch extraction.** Per column, keep only the run nearest the aperture's top
   edge and the run nearest its bottom edge. Teeth line the aperture; the tongue
   floats in the middle. This rejects the tongue **structurally** rather than by
   threshold tuning, and is what finally fixed it.

5. **Interdental split.** Crowns touch, so connected components merge a whole
   arch into one blob. The gaps are dark vertical lines — minima in the
   per-column mean whiteness — and cutting there recovers individual teeth.

Measured on 258 open-mouth frames of the real clip: **mean 7.95 teeth/frame**
(median 8), ≥4 teeth on **92.2 %** of frames, **0 %** frames with zero.
Split sensitivity was swept: 0.88 → 6.3 teeth, **0.94 → 8.0 (chosen)**,
0.97 → 9.4 with visible over-segmentation. It is a UI slider.

## 15. How tooth tracking works

`ToothTracker.js` — **IoU + centroid matching, greedy, in mouth-local space.**

The coordinate frame matters more than the algorithm. Detections arrive in
mouth-local units, so the anchor has *already* removed head translation, scale
and roll: a tooth that stays put on the jaw barely moves in this frame even as
the head swings across the camera. That makes association nearly trivial, so the
cheapest adequate method wins.

* **IoU** leads — teeth are equal-sized neighbours in a row, and box overlap
  disambiguates adjacent crowns better than distance alone, which is exactly
  where centroid-only tracking swaps IDs.
* **Centroid distance** breaks ties and rescues frames where a crown was partly
  cut by the split, so IoU collapses but the centre barely moved.
* **Greedy**, not Hungarian: under ~20 tracks the two almost always agree.
* **Upper never matches lower** — a hard constraint, verified by test.

**Rejected:** *optical flow* would re-solve motion the anchor already removed.
*Kalman* models velocity, but in mouth-local space a tooth's velocity is
essentially zero — no dynamic worth modelling, only lag and tuning.

**Occlusion (§6):** closing the mouth hides every tooth at once, so tracks live
for `maxMissing = 18` frames (~0.6 s) before retiring. Reopening reuses the same
IDs rather than renumbering the dentition.

Defaults were tuned by replaying 258 frames of **real** detections through the
tracker (`tests/fixtures_real_detections.json`):

| config | ID persistence | total IDs created |
| --- | --- | --- |
| iou .20 / dist .09 / miss 8 | 0.869 | 99 |
| iou .10 / dist .13 / miss 8 | 0.874 | 89 |
| **iou .06 / dist .16 / miss 18** | **0.873** | **66** ← chosen |

Persistence barely moves, but a third fewer spurious tracks are created.

## 16. How tooth smoothing works

`TrackingSmoother.js` reuses the Step-2 one-euro filter per track, over centre
(u,v), box (w,h) and confidence. Because filtering happens in **mouth-local**
coordinates, head motion is already removed and the filters only ever see
genuine detection noise — so smoothing can be gentle enough not to lag.

Contours are smoothed by centre/scale rather than vertex-by-vertex: split
boundaries shift frame to frame, so a per-vertex filter would fight a changing
vertex count. Three presets (responsive / balanced / smooth) are exposed, and
filter banks are released when a track retires so a long session cannot leak.

## 17. Performance and frame skipping

Face and mouth tracking run **every frame**; detection runs every
`detectEveryN` frames (default **1**), tracker and smoother every frame.
Skipping is safe here specifically because detections live in mouth-local
coordinates — between detections the teeth do not move *in that frame*, so the
anchor alone carries the overlay and there is nothing to extrapolate.

Measured end-to-end in a browser fed the real mouth video as a fake camera:

```
Tooth inference : 1.24 ms
Step 3 total    : 1.30 ms   (ROI + segment + track + smooth)
```

The segmenter works on a 192×144 ROI, so it is far cheaper than the face model
— which is why the default is "every frame" and skipping is offered for slower
devices rather than forced on everyone. Expect **20–30 fps** on a recent phone,
dominated by MediaPipe, not by Step 3.

## 18. Step 4 compatibility

Tooth IDs and geometry are already in the mouth-local frame Step 4 needs:

```js
{ tooth_id: 6, abnormality: 'suspected_caries', confidence: 0.87, mask: ... }
```

* `track.id` is the stable key to attach findings to.
* `track.smoothed.{center,box,contour}` are in mouth-local units — resolution and
  distance independent, so a finding computed once stays attached.
* `anchor.localToScreen()` renders it; `anchor.getMatrix()` gives a 4×4 for a
  WebGL/three.js layer.
* `ToothPipeline.selectAt()` already maps a tap to a tooth.

**Numbering (§9):** IDs are **internal tracking IDs only**. No FDI/anatomical
numbering is claimed or inferred — the panel states this explicitly. Real FDI
mapping needs midline detection and arch-position reasoning that this step does
not implement.

## 19. Test procedure

`npm test` → **35 tests**, no browser or camera needed, including a regression
that replays real recorded detections and asserts ID persistence > 0.80.

On a device, run through these and record the HUD each time:

| # | Condition | Expect |
| --- | --- | --- |
| 1 | Neutral face, mouth closed | Teeth: 0 *(mouth closed)* — correct, not a failure |
| 2 | Smile (teeth showing) | upper arch detected, IDs steady |
| 3 | Mouth slightly open | both arches, 4–10 teeth |
| 4 | Mouth widely open | most teeth; tongue **not** outlined |
| 5 | Head turned left | far teeth drop out, near IDs persist |
| 6 | Head turned right | mirror of 5 |
| 7 | Head tilted | contours roll with the teeth (rectification) |
| 8 | Move closer | contours scale, count stable |
| 9 | Move farther | count may drop as teeth get small |
| 10 | Dim / bright / side light | adaptive threshold should hold; note where it fails |

For each: detection success, ID stability (do numbers churn?), jitter, FPS.
Turn on *Rectified ROI (debug)* to see exactly what the segmenter sees.

## 20. Step 3 known limitations

* **Requires an open mouth.** Below an opening ratio of 0.10 the system reports
  `Teeth: 0 (mouth closed)` rather than segmenting lip highlights.
* **Not a learned model** (§13). It keys on enamel being the brightest neutral
  surface, so it degrades with very dim light, heavy colour casts, strong
  specular highlights on wet lips, or dark/discoloured teeth.
* **Split boundaries are the weakest part.** Crown edges wobble frame to frame,
  which is the main source of ID churn (~0.87 persistence, not 1.0). Temporally
  stabilising the cut positions is the obvious next improvement.
* **Occasional lower-lip inclusion** when the lit lower lip is very bright and
  the lower teeth are dim — visible in the prototype figures.
* Tooth count is **not** anatomical: one detection ≈ one visible crown segment,
  which may merge or split relative to true dentition.

---

# STEP 3b — Tooth-level 3D AR anchoring

Step 3 found and tracked individual teeth in **2D**, inside the mouth-local
frame. Step 3b gives every tracked tooth its **own 3D spatial anchor**: a full
4×4 transform with position, rotation and scale, in real 3D coordinate spaces,
that a 3D tooth mesh / nerve canal / lesion volume can be attached to later.

There is one 2D rectangle nowhere in this stage. Each proxy you see is a set of
vertices in tooth-local metres pushed through that tooth's own transform and
projected with a pinhole camera.

## 21. Read this before trusting any number

Depth is the whole problem. **A single RGB camera cannot measure how far away a
tooth is.** Rather than hide that, every field of every pose carries a
`provenance` tag, and the UI shows them:

| Tag | Meaning | What is tagged with it |
| --- | --- | --- |
| `tracked` | MediaPipe's own per-frame 3D fit | head **orientation**, from the facial transformation matrix |
| `measured` | derived from this frame's pixels | each tooth's **viewing ray** and **apparent size**, from its segmented contour |
| `estimated` | model-based inference, not observation | **distance along that ray** — mouth distance plus a parabolic dental-arch offset |
| `assumed` | a fixed constant | **metric scale**, ultimately pinned to an average adult face |

So: *where a tooth is on the screen* is measured, *which way it faces* is
tracked, *how far away it is* is estimated from anatomy, and *how many
millimetres that is* rests on an assumption.

**This is not medical-grade registration and must not be presented as one.**
It is a correct 3D framework — real transforms, real coordinate spaces, real
head-pose tracking, exact reprojection — with an anatomical prior standing in
for depth sensing that a later stage would supply (CBCT registration, a
TrueDepth/LiDAR camera, or stereo). Swap `archDepthAt()` and the mouth-distance
source for real measurements and every transform downstream keeps working.

## 22. Coordinate spaces

```
tooth-local  ──toothToCamera──►  camera  ──projectPoint──►  pixels
     ▲                              ▲
     └────── face frame ────────────┘   (rigid to the skull)
```

| Space | Units | Axes | Notes |
| --- | --- | --- | --- |
| **camera** | metres | +X right, +Y **down**, +Z into the scene | OpenCV convention; what `projectPoint` and the canvas assume |
| **face** | metres | +X right, +Y **up**, +Z out toward the camera | origin at the mouth centre, rigid to the skull. **Teeth are static here** — which is what makes it the right frame to anchor in |
| **tooth-local** | metres | +X across the arch, +Y crown→root, +Z out of the labial surface | origin at the crown centre. **This is where a 3D tooth model attaches** |

MediaPipe reports its head matrix in an OpenGL-style frame (+Y up, camera down
−Z). `basisFromHeadMatrix()` converts it with `C = diag(1, −1, −1)`; that is
the one place a sign error would silently mirror everything, so it is isolated
and unit-tested on its own.

Note the face frame is **not** the same handedness as `MouthARAnchor`'s 2D
pixel-space mouth-local frame (+Y down, units of mouth width). That frame is a
weak-perspective screen convenience. `ToothPoseEstimator.estimate()` is the
only place the two meet, and it converts explicitly.

## 23. How a tooth pose is built

1. **Distance to the mouth.** Prefer MediaPipe's metric head translation — it
   is far steadier under yaw than depth-from-apparent-width, since a turned
   mouth is foreshortened. Its units differ between builds, so the value is
   accepted only if centimetres *or* metres lands in a physically sensible
   range; otherwise it falls back to `f · 50 mm / mouthWidthPx`. The HUD says
   which one is live.
2. **Metric size.** Apparent mouth width at that distance, divided by the
   foreshortening of the face's +X axis (so teeth do not shrink when the
   subject looks away), and corrected for the fact that the mouth corners sit
   further back on the arch than the midline.
3. **Position.** The tooth is placed **on its measured viewing ray** — only the
   distance along that ray is inferred. This is why an anchor reprojects
   *exactly* onto the tooth the segmenter found (measured error < 0.01 px),
   while depth ordering still comes from the arch.
4. **Orientation.** +Z follows the arch normal at that tooth's position across
   the mouth; +Y runs crown→root (up for the upper arch, down for the lower);
   +X is the cross product, so the basis is right-handed by construction. The
   whole thing is then rotated by the tracked head.
5. **Scale.** Width and height from the segmented contour; labial-lingual
   thickness from the crown prior.

## 24. Modules

```
ToothTracker  ──►  ToothPoseEstimator  ──►  Tooth3DAnchor  ──►  AR3DRenderer
 2D tracks         2D + head pose → 3D      one per tooth,      projects proxies
 with stable IDs   with provenance tags     holds the 4×4       (box / axes / sphere)
```

* `core/math/mat4.js` — column-major 4×4s, rigid inverse, pinhole projection.
* `core/ToothPoseEstimator.js` — the honesty boundary. Everything estimated or
  assumed lives here, tagged.
* `core/Tooth3DAnchor.js` — the anchor. `getMatrix()` is directly usable as a
  three.js `Object3D.matrix`; `attach(model)` is the Step-4 seam.
  `Tooth3DAnchorSet` keeps one anchor per tooth ID so anchors persist with
  their tooth instead of being rebuilt (and renumbered) every frame.
* `ui/AR3DRenderer.js` — draws the debug proxies, sorted back-to-front by real
  camera-space depth.

Attaching real geometry later:

```js
const anchor = teeth.anchors3D.get(toothId);
anchor.attach(myToothMesh);          // authored in tooth-local metres
object3D.matrix.fromArray(anchor.getMatrix());
```

## 25. Verification — proving it is 3D, not a repositioned rectangle

`tests/anchor3d.test.mjs` (17 tests) builds a **forward-simulated scene**:
ground-truth 3D teeth are projected to synthetic 2D observations, those
observations are fed to the estimator, and the recovered 3D is compared with
the truth. Three properties separate a real 3D anchor from a fake one:

| Property | Result |
| --- | --- |
| **Round-trip** — an anchor must reproject onto the tooth it came from | **< 0.01 px** across ±20° yaw |
| **Rigidity** — a tooth must stay put in face space while the head turns | **1.9 mm** over a ±25° yaw sweep — versus **21.3 mm** for the same code run without the head matrix, an 11× difference |
| **Accuracy** — recovered 3D vs the simulated ground truth | **2.3 mm** |

The rigidity control is the important one: an implementation that merely moves
a 2D box around cannot be rigid in the skull's frame, because it has no notion
of the head's 3D rotation at all. The 21.3 mm figure is what that looks like.

The accuracy number measures the **geometry pipeline only** — the simulator
uses the same arch prior the estimator assumes. It says the transforms are
right. It says nothing about how well that prior matches a real mouth, which is
the dominant real-world error and is not measurable from RGB at all.

Also covered: the GL→CV basis conversion, the head-matrix unit heuristic,
distinct transforms per tooth, arch curvature actually bending, metric output
scaling linearly with the assumed mouth width, distance-invariant tooth size,
per-ID anchor lifetime, the attachment seam, and that every pose tags its own
provenance (including downgrading `orientation` to `estimated` when no head
matrix is available).

## 26. Using it

Panel → **3D tooth anchors** → *Show 3D proxy per tooth*.

* **3D box** — the outward face is filled, so you can watch each tooth's facing
  change as you turn your head. That change is the visible proof of 3D.
* **Coordinate axes** — per-tooth X/Y/Z.
* **Sphere** — radius shrinks with distance because it is a projected 3D
  offset, not a fixed pixel size.

The panel shows the live anchor count, the mouth distance, and **which depth
source is running**. Tap a tooth to see its 3D position and its rotation *in
the face frame* — camera-frame Euler angles are correct but read as ±180° at
rest, which means nothing to a human; in the face frame a central tooth sits
near zero and the number is the arch splay.

`Log 1 frame` prints the full anchor table — per tooth: position in cm,
rotation, size in mm — plus the provenance line and the active depth source.

## 27. Step 3b known limitations

* **Depth is a prior, not a measurement.** Every tooth's distance comes from a
  parabolic arch model, not from this person's anatomy. An unusual arch, an
  orthodontic case or a partial dentition will be wrong in depth while still
  looking right on screen, because the reprojection is exact either way.
* **Absolute scale is unobservable.** A big mouth far away and a small mouth
  near by are pixel-identical. Everything metric inherits the average-face
  assumption — expect roughly ±10% between adults.
* **Arch splay drifts a little with head yaw** (< 8° at 25°), because the prior
  is evaluated at the tooth's *observed* position across the mouth and yaw
  shifts that observation.
* **Orientation is the head's, not the tooth's.** Individual tilt, rotation or
  crowding of a real tooth is not measured; each tooth is oriented by the arch
  model plus the tracked head pose.
* **No occlusion and no lighting.** Proxies are drawn over the video, sorted by
  depth but not clipped by lips or by each other.
* **Tooth IDs are tracking IDs**, not FDI/anatomical numbers — unchanged
  from Step 3.

---

# STEP 3 v2 — Accurate tooth detection, stable tracking, recording

Professor's feedback on Step 3: *the tooth detection is not accurate enough —
detect all visible teeth as accurately as possible*, and *add a video recording
button*. This section documents what was audited, what replaced it, how it was
measured, and how to record sessions for analysis. Dental X-ray / CBCT
registration and any diagnosis are **not** part of this step.

## 28. Audit of the v1 detector (before any change)

| Question | Finding |
|---|---|
| Model used | **None.** `ToothSegmenter.js` is hand-written classical CV. |
| Trained for teeth? / dataset | No training, no dataset; thresholds hand-tuned on one clip of one person. |
| Classes | None learned. Upper/lower came from geometry (run nearest the top/bottom lip). |
| Individual teeth or mouth region? | Individual instances — but as **vertical column slices** of a bright band, not tooth boundaries. |
| Method | Heuristic: whiteness `V·(1−S)` → percentile threshold → per-column arch runs → cut at brightness minima. |
| Missed teeth | Side / back teeth darker than the threshold; faint interdental gaps never cut (two teeth → one); a porting bug eroded the lip aperture with a 13×13 kernel instead of the tuned 7×7, clipping upper incisors that touch the lip. |
| False detections | Lit lips, wet-lip highlights, tongue when no lower arch is visible. |
| Overlaps / duplicates | Textured wide crowns cut in two; in clenched smiles upper and lower teeth fused into one tall slice. |
| Partially visible teeth | Slices < 5 % of the arch width dropped. |
| Lighting | Whiteness assumption breaks under colour casts / dim light. |
| Face angle | Yaw compresses side teeth into slices a few px wide → merged or dropped. |
| Tracking | Greedy IoU+centroid; the tracker was fed an empty list on skipped frames, so with "detect every N > 1" every tooth flickered invisible. |

**Measured baseline of the v1 detector** (before any model change; every
number from `tools/eval_detectors.mjs`, the app's own detector code run in
Node on crops cut with the app's own mouth rectification):

| Evaluation set | Ground truth | v1 as shipped | v1 + aperture-erosion fix |
|---|---|---|---|
| `mouthtestvideo.mp4`, 9 frames, **103 teeth** (instances) | provisional point GT, `tests/eval/gt_mouthtestvideo.json` | precision 50.0 %, **recall 35.9 %**, F1 41.8 % — 15 duplicates, 19 merges | precision 63.5 %, recall 45.6 %, F1 53.1 % — 14 duplicates, 23 merges |
| same, clear teeth only (72) | partial teeth ignored | F1 50.3 % | F1 63.8 % |
| EasyPortrait test split, **360 selfies** of many people (300 with teeth) | third-party pixel masks | teeth-pixel **IoU 27.4 %**, recall 31.5 %, precision 67.8 % | IoU 31.1 %, recall 36.4 %, precision 68.4 % |

In words: the v1 detector found roughly **one visible tooth in three**, and
where it did find teeth it often fused two into one box or cut one in two —
which is what the professor saw.

## 29. Model: what was chosen and why

| Candidate | Domain | Per-tooth? | Licence / access | Used? |
|---|---|---|---|---|
| **DentalAI** (P. Valluri, 2023) | intraoral photos, 2,495 images, 22,731 tooth polygons | ✅ | CC BY 4.0, public | ✅ instance supervision |
| **EasyPortrait** (Kvanchiani et al., 2023) | ~20 k selfie portraits, TEETH mask class | semantic | CC BY-SA 4.0 variant | ✅ selfie-domain appearance |
| SegmentAnyTooth (2025) | intraoral photos, FDI numbering | ✅ | weights only via **signed non-commercial agreement** | ✗ — plug-in path documented |
| AlphaDent (2025) | intraoral DSLR | pathology masks only | CC BY-SA 4.0 | ✗ wrong task |
| DENTEX, STS-Tooth, Tufts, OralSeg … | panoramic X-ray / CBCT | — | various | ✗ wrong modality |
| Generic detectors (COCO YOLO, SAM) | generic objects | — | — | ✗ not tooth models |

**ToothNet-lite** is a small U-Net written for this project (≈0.12 M
parameters, ≈80 M multiply-adds at 160×120): stride-2 stem, three scales,
dilated context at the bottleneck, and three output maps — *teeth*,
*tooth-centre heatmap*, *interdental boundary*. DentalAI teaches where one
tooth ends and the next begins; EasyPortrait, cropped with the app's own mouth
rectification, teaches what teeth look like to a phone front camera (its
instance losses are masked out because it has no per-tooth labels).
Ultralytics YOLO was deliberately not used: its AGPL-3.0 licence would extend
to the trained weights.

## 30. How detection works now

```
camera frame → FaceTracker (MediaPipe) → MouthTracker → MouthARAnchor
   → MouthROI: rectified 160×120 mouth crop (head roll removed)
   → ToothNet-lite (ONNX Runtime Web, WASM, on-device)
        teeth P(x)   centre heatmap   boundary map
   → toothDecode: centre peaks seed teeth; each enamel pixel joins the seed it
     reaches most cheaply, crossing a predicted boundary is expensive
     (Dial's bucket-queue watershed); unseeded enamel blobs still become teeth
   → jaw: geometric (instance height vs the lip-aperture midline, split at the
     largest gap) — the training data has no jaw labels
   → visibility: partial if cut by the ROI/lip edge or < 45 % of median size
   → ToothTracker v2 → one-euro smoothing → 3D anchors → AR rendering
```

Per tooth the pipeline exposes: tracking ID, jaw, bounding box, contour and an
instance-mask reference, centre, confidence, visibility, tracking state and
the Step-3b 3D anchor (position / orientation / scale / provenance).

## 31. How tracking works now (v2)

* **Duplicate suppression (NMS)** on raw detections (IoU > 0.5 or 80 % containment).
* **Hungarian** (optimal) assignment instead of greedy.
* **ByteTrack-style two stages**: confident detections vs all tracks, then weak
  detections vs confirmed tracks only — weak evidence keeps a tooth alive but
  can never mint a new ID.
* **Jaw-relative coordinates**: upper teeth measured from the upper inner lip,
  lower from the lower inner lip, cancelling mouth opening (the largest motion
  left after the head is removed).
* Lifecycle: tentative → confirmed after 2 hits; tentative misses are dropped;
  confirmed tracks survive 18 frames of occlusion (closed mouth) and get their
  IDs back; converged duplicate tracks are merged keeping the older ID.
* **Stability is measured**: mean Jaccard overlap of visible IDs between
  consecutive frames (last 30), shown live and stored in the metadata.
* Kalman / optical flow were rejected: in jaw-relative mouth-local space a
  tooth's velocity is ~0, so a motion model adds lag without information.

## 32. Accuracy — measured, before vs after

All numbers from `tools/eval_detectors.mjs` (the app's own detector modules,
run in Node on crops cut with the app's own mouth rectification). Decoder
thresholds were tuned on **DentalAI validation** crops only; none of the sets
below was used for training or tuning.

| Test set | Ground truth | v1 as shipped | v1 + aperture fix | **v2 learned** |
|---|---|---|---|---|
| `mouthtestvideo.mp4`, 9 frames, 103 teeth — **all teeth** | provisional point GT (own annotation, see §36) | P 50.0 %, R 35.9 %, **F1 41.8 %** | P 63.5 %, R 45.6 %, F1 53.1 % | P 86.7 %, R 75.7 %, **F1 80.8 %** |
| ↳ missed / false pos. / duplicates / merges | | 66 / 37 / 15 / 19 | 56 / 27 / 14 / 23 | **25 / 12 / 2 / 15** |
| ↳ count error per frame (MAE) · jaw accuracy | | 3.22 · 78.4 % | 3.22 · 74.5 % | **1.67 · 100 %** |
| same, **clear teeth only** (72; partial teeth ignored) | | R 50.0 %, F1 50.3 %, 36 missed, 13 merges | R 62.5 %, F1 63.8 % | **R 90.3 %, F1 86.1 %, 7 missed, 2 merges** |
| EasyPortrait test — 360 selfies of different people (300 with teeth) | third-party pixel masks | teeth IoU 27.4 % (P 67.8, R 31.5) | IoU 31.1 % (P 68.4, R 36.4) | **teeth IoU 68.7 %** (P 94.6, R 71.5) |
| ↳ closed-mouth images with any detection | | 0 / 60 | 0 / 60 | **0 / 60** |
| DentalAI test — 250 intraoral photos, 2,187 teeth | third-party per-tooth polygons | not applicable¹ | not applicable¹ | **P 79.5 %, R 84.7 %, F1 82.0 %**, mask IoU 0.79 |

¹ the classical method needs the lip aperture to find the arches; clinical
intraoral crops have none, so running it there would not be a fair test.

How to read these:

* On the project's own video the learned detector finds **about three in four
  visible teeth (nine in ten clearly visible ones)** versus one in three
  before, with far fewer split teeth. Most remaining errors are merges of two
  lower incisors in blurred, clenched frames and the thin upper band of a
  wide-open mouth.
* The EasyPortrait number is the strictest one: it scores what the app
  actually outputs — the union of per-tooth outlines, clipped to the lip
  aperture and gated by mouth opening — against pixel masks drawn by the
  dataset's annotators. (The model's raw teeth mask scores IoU 0.85 on the
  EasyPortrait validation split; outline decimation and the aperture clip
  account for the gap.)
* The own-video ground truth is provisional (§39). The two public test sets
  are independent of this project.

### 32b. v3.3 — low-light model (ToothNet-lite 1.1)

Real recordings made in a dim room with a laptop webcam showed two problems.
First, the trained model was not running at all (see §38b — the classical
fallback was active). Second, in simulated dim light the v1.0 model still
*sees* the teeth (mask IoU barely drops) but **fails to separate them**:
neighbouring crowns, especially the upper row, merge into one region.

v1.1 is v1.0 fine-tuned for 16 epochs with low-light augmentation (strong
under-exposure, warm lamp cast, amplified sensor noise). It was **selected on
validation data only** against v1.0, a test-time auto-gain, and wide-region
re-splitting (normal + simulated low light; both extras lost on validation and
stay off). Test results — "low light" is the same test data darkened to 28–45 %
exposure with a warm cast and sensor noise:

| Test set | Classical v1 | Model 1.0 (3.2.0) | **Model 1.1 (3.3.0)** |
|---|---|---|---|
| Own video, all teeth — normal light | F1 53.1 % | F1 83.9 % (R 78.6 %) | F1 83.1 % (R 78.6 %) |
| Own video, all teeth — **low light** | F1 20.1 % | F1 57.3 % (R 41.7 %, 50 merges) | **F1 79.1 % (R 69.9 %, 20 merges)** |
| Own video, clear teeth — normal light | F1 63.8 % | F1 88.7 % | F1 85.5 % |
| Own video, clear teeth — **low light** | F1 25.2 % | F1 69.0 % | **F1 84.9 %** |
| EasyPortrait test teeth IoU — normal / low light | 31.1 % / 5.2 % | 68.7 % / 67.3 % | 68.4 % / 67.9 % |
| DentalAI test instances F1 | — | 82.0 % | **83.2 %** (303 vs 335 missed) |

(Model-1.0 numbers here use the final decoder setting, boundary weight 20,
so they differ slightly from §32.) In dim light 1.1 recovers most of what 1.0
lost; in normal light it is equal on the public test sets and slightly lower on
the small own-video set (about −1 to −3 F1 points). The thin upper row of a
wide-open mouth can still come out as one region when it is only a few pixels
tall. A **raw** recording in the target room, annotated in `eval.html`, is the
way to confirm these numbers on real low-light footage.

## 33. Performance

| Quantity | Measured | Where |
|---|---|---|
| Model size | 124,899 parameters, 505 KB ONNX, ≈80 M multiply-adds per 160×120 crop | model card |
| Model inference, browser (Chrome 153, WASM, multithreaded via COOP/COEP) | **median 6.6 ms**, p90 6.9 ms | desktop, 8-core x86, in-page |
| Model inference, WASM single-thread (no isolation) | median 16.5 ms, p90 23 ms | same machine, ONNX Runtime Web |
| Decode + jaw + outline per frame | ≈2–4 ms | included in 9–15 ms/frame harness totals |
| Classical v1 detector, for comparison | 2–6 ms | same harness |

* Inference is **asynchronous**: one inference in flight, tracking and
  smoothing carry the teeth in between, results land in mouth-local
  coordinates so a ~1-frame-old result is still correctly placed. A slower
  device lowers the detection rate, not the camera frame rate; the live panel
  shows camera FPS, detection FPS, inference and tracking time separately.
* Camera → tooth pipeline stays entirely on the device: no frame is sent to
  Vercel, an API route or any server.
* **Not measured on a physical phone in this work.** Mid-range phone CPUs run
  WASM roughly 3–5× slower than this desktop, i.e. an *estimated* 20–60 ms per
  inference (≈15–40 detections/s) — to be confirmed with the test plan, whose
  recordings store per-frame FPS and inference time.
* Headless-Chrome end-to-end runs used for functional testing reached only
  4–13 camera FPS because MediaPipe ran on a software GPU there; those
  frame rates are an artefact of the test environment, not a result.


## 34. Recording

* **● RECORD VIDEO** (top of the side panel) starts/stops; a blinking red dot
  and timer appear in the top bar; the evaluation panel shows Recording ON and
  the duration.
* **Record annotations** off → the camera's own stream is recorded (raw,
  unmirrored — use this for ground truth, datasets and re-running detection).
  On → the preview as seen (video + tooth masks, IDs, confidence, landmarks, AR
  overlay) plus a burned-in line with FPS, tooth count, stability and time.
* After stopping: an inline **preview**, **Download video**, **Download
  metadata**, **Share / Save to device** (Web Share sheet — the practical way to
  save into Photos/Files on iOS and Android) and **Discard**.
* Format: WebM (VP9 → VP8) on Chrome / Firefox / Android; Safari (iPhone /
  iPad / macOS) records MP4/H.264, selected automatically.
* File names: `dental_tracking_YYYYMMDD_HHMMSS.webm` + `.json`, saved to the
  browser's download folder (desktop / Android) or wherever the share sheet
  puts them (iOS).
* Everything is local: MediaRecorder encodes in the browser, no frame is ever
  uploaded, and the Vercel deployment serves static files only.
* **Capture frame for ground truth** saves the current raw frame as PNG + its
  detections as JSON, for quick single-frame annotation.

## 35. Metadata format (`dental-ar-recording/1`)

```json
{
  "header": {
    "schema": "dental-ar-recording/1", "app": "dental-ar", "appVersion": "3.2.0",
    "startedAt": "2026-09-11T10:15:30.120Z", "mode": "raw",
    "video": { "mimeType": "video/webm;codecs=vp9", "width": 1280, "height": 720 },
    "mirrored": { "preview": true, "annotatedVideo": false, "rawVideo": false },
    "coordinates": "bbox/center/contour: raw (unmirrored) video pixels; uv: mouth-local ...",
    "timebase": "t_ms is milliseconds since the MediaRecorder start event",
    "detector": { "key": "learned", "name": "Learned tooth segmenter (U-Net)", "learned": true,
                  "model": { "name": "ToothNet-lite", "version": "1.0" } },
    "camera": { "width": 1280, "height": 720, "facingMode": "user", "frameRate": 30 }
  },
  "summary": { "frames": 362, "duration_ms": 12071, "mean_fps": 29.6,
               "mean_teeth_when_mouth": 9.4, "unique_tooth_ids": 14, ... },
  "frames": [
    { "i": 0, "t_ms": 16.4, "fps": 29.8, "face": true, "mouth": true, "opening": 0.41,
      "n": 9, "avg_conf": 0.83, "stability": "stable",
      "timing": { "detect_ms": 11.2, "track_ms": 1.4, "total_ms": 3.1 },
      "teeth": [ { "id": 3, "jaw": "upper", "conf": 0.91, "state": "stable", "hits": 57,
                   "uv": [0.071, -0.083], "center": [652.3, 401.8],
                   "bbox": [640.1, 380.2, 24.5, 41.0],
                   "contour": [[641.0, 382.5], ...],
                   "pos3d_m": [0.004, 0.018, 0.312], "depth_src": "mediapipe-metric-head-model" } ] }
  ]
}
```

## 36. Evaluation and ground truth

* **`eval.html`** (served with the app, offline): load a recording + its JSON,
  step through frames, click the crown centre of every visible tooth
  (upper/lower, partial), draw ignore regions, and read precision / recall /
  F1 / missed / false positives / duplicates / merges / count error / jaw
  accuracy, plus FPS, latency and ID continuity from the metadata. Export the
  ground truth (reusable for the next model) and a JSON report.
* **`tools/analyze_recording.mjs`**: the same statistics for many recordings
  from the command line, optionally scored against exported ground truth.
* **`tools/eval_detectors.mjs`**: runs the app's detector modules in Node on
  rectified crops and scores them — the source of every number in §32.
* Metrics are only ever computed from annotations; with none, the UI says so.

## 37. Reproducing the model

```bash
cd step2_mouth_ar
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install onnx onnxruntime opencv-python mediapipe numpy
# 1. data (DentalAI tarball from Dataset Ninja; EasyPortrait subset by range requests)
python tools/train/fetch_easyportrait.py --stats mask_stats.json --out _data/ep
python tools/train/build_dataset.py --dentalai-tar dentalai.tar --ep-dir _data/ep \
       --ep-ann annotations.zip --model public/models/face_landmarker.task --out _data/ds
# 2. train (CPU, ~40 min) -> tooth_seg.onnx + tooth_seg.json
python tools/train/train_tooth_model.py --data _data/ds --out _data/model --epochs 35
cp _data/model/tooth_seg.{onnx,json} public/models/
```

A different model (e.g. SegmentAnyTooth weights obtained under their licence,
or a model fine-tuned on your own annotated recordings) plugs in by
implementing `detectAsync()` in a `ToothDetector` subclass and registering it.

## 38. Test plan

See [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md): twelve recording conditions
(straight, smile, wide / slightly open, head left / right / tilted, near / far,
lighting, partial visibility, background), how to annotate them, and which
metrics to report for each.

## 38b. Troubleshooting: "only the front / lower teeth are detected"

Check the burned-in line of an annotated recording or the *Detector* box in
the panel. If it says **classical**, the trained model is not running and you
are seeing the v1 baseline — which, as §32 shows, misses most upper and side
teeth.

Up to v3.2.0 this happened silently under `npm run dev`: Vite rewrote ONNX
Runtime's relative WebAssembly URL, the request was answered with
`index.html`, the model failed to initialise, and the app fell back to the
classical detector. The fallback banner was then cleared as soon as the
camera started, so nothing on screen said so.

Fixed in 3.3.0:

* `scripts/vendor-ort.mjs` (run by `npm install`) copies the runtime into
  `public/ort/`, and the detector always loads it from there — the same path
  in dev, preview, Vercel and the Capacitor app.
* If the model still cannot load, a **persistent warning box** stays in the
  Tooth-detection panel (it is not cleared by other messages).

If you pulled the code without reinstalling, run `npm run vendor:ort` once.

## 39. Step 3 v2 known limitations

* **Not 100 % and not medical-grade.** The numbers in §32 are the measured
  performance; teeth are still missed and occasionally split or merged.
* **Evaluation on the project's own video uses provisional ground truth**
  (visual annotation of a 480×864 phone clip, one person). Treat it as
  indicative; re-annotate with `eval.html` and report the test-plan recordings.
  The EasyPortrait and DentalAI results use third-party labels.
* **Low-light results use simulated darkening** of real test images (§32b);
  no raw low-light recording with ground truth exists yet.
* **Instance supervision comes from clinical intraoral photos** (DentalAI).
  The selfie-domain data (EasyPortrait) has only an all-teeth mask, so tooth
  separation in selfies is learned by transfer. Errors concentrate on lower
  incisors in clenched smiles, shadowed premolars and heavily blurred frames.
  Fine-tuning on annotated recordings made with this app is the direct fix.
* **Upper/lower jaw is assigned geometrically**, not learned (no jaw labels
  exist in the training data); its accuracy is measured in §32.
* **No FDI tooth numbers**: IDs are tracking IDs.
* **Detection needs an open mouth** (opening ratio ≥ 0.10), as in v1.
* **Phone performance was not measured on a physical phone in this work.**
  Desktop timings are measured (§33); on phones, detection runs asynchronously
  so a slower model lowers the *detection* rate, not the camera frame rate.
  Use the FPS / latency fields in the recorded metadata to measure a device.
* **Recording:** Chrome's WebM files carry no duration header (the evaluation
  tool works around it; ffmpeg can remux); Safari records MP4; on iOS use
  *Share / Save to device*. Annotated recordings of the front camera are
  mirrored like the preview — annotate ground truth on **raw** recordings.
* **Cross-origin isolation** (COOP/COEP headers) is required for multithreaded
  inference; the dev server and `vercel.json` send it. Any future third-party
  resource must be served with CORP/CORS headers or it will be blocked.

## 40. What should be implemented next

1. Record the twelve test-plan sessions on the target phones and annotate
   them in `eval.html` (≥ 10 frames each): this gives per-condition accuracy and
   real device FPS.
2. Fine-tune ToothNet on those annotations (same `train_tooth_model.py`, add a
   third source) — the largest expected accuracy gain, because it closes the
   selfie-vs-clinical gap for tooth *separation*.
3. Learn the jaw label (upper/lower) as a fourth output once annotated data
   has jaw labels.
4. If the SegmentAnyTooth weights are obtained under their licence, wrap them
   as a `ToothDetector` and compare on the same ground truth.
5. Only then proceed to Step 4 (dental image registration).

