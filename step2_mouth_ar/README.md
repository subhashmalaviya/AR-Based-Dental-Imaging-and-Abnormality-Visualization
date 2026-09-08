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
npm test                # 35 tests (Step 2 + Step 3), no browser needed
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

### Android (easiest)

```bash
npm run build
python3 serve_https.py            # prints https://<your-lan-ip>:8443/
```

Open that URL on the phone (same Wi-Fi) → *Advanced → Proceed* past the
self-signed warning → **Start** → allow camera.

Or skip certificates entirely with Chrome's origin allowlist:
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add
`http://<lan-ip>:5173` → relaunch, then `npm run dev`.

### iPhone / iPad

iOS Safari usually **refuses** self-signed certificates. Use a tunnel, which
gives a real certificate:

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
