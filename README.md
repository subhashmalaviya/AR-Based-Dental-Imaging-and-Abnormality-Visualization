# AR Based Dental Imaging and Abnormality Visualization

IIT Delhi B.Tech project. Built incrementally as three working prototypes,
each one a real, tested system rather than a mockup of the next:

| Step | What it does | Stack | Status |
| --- | --- | --- | --- |
| **1 — Surface AR** | Tracks a real cylindrical cup from a photo/video and prints a logo onto its actual 3D surface (UV-mapped mesh, 6-DoF pose from the object's own geometry, not a flat overlay). | Python, OpenCV, NumPy | ✅ working, tested |
| **2 — Face & mouth tracking** | Live camera face tracking, mouth landmarks, and a stable AR coordinate anchor attached to the mouth that survives head motion. | Web (Vite + MediaPipe), runs on Android/iOS/desktop | ✅ working, tested |
| **3 — Tooth detection & tracking** | Detects the user's actual visible teeth inside the mouth ROI from the live camera, segments individual crowns, and tracks them with stable IDs frame to frame. | Same web app, builds on Step 2 | ✅ working, tested |

Steps 2 and 3 live in the same app (`step2_mouth_ar/`) because Step 3 is built
directly on Step 2's mouth anchor, not a separate pipeline. Step 4
(dental image input → AI analysis → map to a tracked tooth → AR visualization
of the result) is **not implemented** — see each module's README for the
documented seam it will plug into.

The throughline across all three steps: **content is defined once in the
object's own coordinates (a cup's UV space, a mouth's local frame), and only
the camera pose changes per frame.** Nothing is a flat screen-space sticker.

---

## Repository layout

```
.
├── project/                  Step 1 — cup surface AR (Python)
│   ├── README.md             full write-up: geometry, pose math, calibration
│   ├── requirements.txt
│   ├── src/                  camera model, mesh, pose estimation, rendering
│   ├── data/                 sample cup images/videos
│   └── outputs/              generated demo renders (regenerate with the scripts below)
│
├── step2_mouth_ar/           Steps 2-3 — face/mouth tracking + tooth detection (web)
│   ├── README.md             full write-up: anchor math, tooth CV pipeline, tuning
│   ├── package.json
│   ├── src/core/             CameraManager, FaceTracker, MouthTracker, MouthARAnchor,
│   │                         MouthROI, ToothDetector/Segmenter/Tracker, smoothing
│   ├── src/ui/                overlay + debug renderers, HUD
│   └── tests/                automated tests (35 passing, incl. a real-data regression)
│
├── ti1.png, iitd_logo.png    Step 1 sample cup photo + logo to overlay
├── tv1.mp4, mouthtestvideo.mp4   Step 1 sample cup video; real mouth footage used
│                                  to develop/validate Step 3's tooth detector
├── vercel.json               builds step2_mouth_ar/ when deploying this monorepo to Vercel
└── .gitignore
```

---

## Quick start

Each step is independent to run (Step 3 is inside the Step 2 app). Pick one:

### Step 1 — cup surface AR (Python)

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo_3d.py                  # standalone 3D demo, no input footage needed
python run_ar3d.py image           # AR overlay on the sample cup photo
python run_ar3d.py video           # AR overlay on the sample cup video
streamlit run app.py               # interactive UI
```

Full details — the pose math, what's identifiable from a single camera vs.
not, calibration, known limitations — are in **[`project/README.md`](project/README.md)**.

### Steps 2-3 — live face/mouth/tooth tracking (web app)

Requires **Node.js 18+**.

```bash
cd step2_mouth_ar
npm install                        # also fetches the MediaPipe model + WASM runtime
npm test                           # 35 tests, no browser or camera needed
npm run dev                        # http://localhost:5173 — camera works on localhost
```

Open the printed URL in a browser, click **Start**, and allow camera access.

To test on an actual **phone/tablet** (recommended — this is a live-camera
system), the camera requires a secure context (HTTPS or `localhost`).
**Easiest: deploy to Vercel** — the root-level `vercel.json` builds this
subdirectory automatically (`cd step2_mouth_ar && npm install && npm run build`,
serving `step2_mouth_ar/dist`), and Vercel's real HTTPS certificate works on
Android *and* iOS/iPadOS out of the box, unlike a self-signed local cert. Just
import this repo on [vercel.com/new](https://vercel.com/new) and open the
deployed URL on the device. For local-network testing instead, see
**[`step2_mouth_ar/README.md` §7](step2_mouth_ar/README.md#7-testing-on-a-real-phone-or-tablet)**
(`python3 serve_https.py`, or a `cloudflared` tunnel for iOS).

That README also covers: the mouth-anchor coordinate math, why a classical
CV method (not a neural net) is used for tooth segmentation and how it works,
how tooth identity is tracked and smoothed, native Android/iOS builds via
Capacitor, and a full device test procedure.

---

## Prerequisites

| Tool | Needed for | Check |
| --- | --- | --- |
| Python 3.9+ | Step 1 | `python3 --version` |
| Node.js 18+ / npm | Steps 2-3 | `node --version` |
| A camera-equipped browser (Chrome/Safari/Edge) | Steps 2-3 | — |
| `ffmpeg` (optional) | inspecting/re-encoding the sample videos | `ffmpeg -version` |

No GPU is required for either step; Step 2-3's face model runs on a GPU
delegate in-browser when available and falls back to CPU automatically.

---

## What's gitignored, and why

`.venv/`, `node_modules/`, `dist/`, and the self-signed HTTPS cert are build
artifacts — regenerate them, don't commit them. `step2_mouth_ar/public/wasm/`
and `.../public/models/` are vendored/downloaded binaries (MediaPipe's WASM
runtime and the face landmark model); `npm install` fetches both automatically
via a `postinstall` script (`npm run setup` to re-run it manually). Everything
else — sample images/videos, source code, and the test fixture derived from
real footage (`step2_mouth_ar/tests/fixtures_real_detections.json`) — is
tracked, so a fresh clone plus the two commands above reproduces everything.

---

## Project scope note

Per the project brief, each step deliberately stops where the next begins:
Step 1 does not track a face; Step 2 does not detect teeth; **Step 3 detects
and tracks teeth only — no cavity/disease diagnosis, no lesion classification,
no X-ray/CBCT processing.** That is Step 4, not yet implemented. See
`step2_mouth_ar/README.md` §18 for exactly how Step 3's tooth IDs and
coordinate system are meant to be reused when Step 4 is built.
