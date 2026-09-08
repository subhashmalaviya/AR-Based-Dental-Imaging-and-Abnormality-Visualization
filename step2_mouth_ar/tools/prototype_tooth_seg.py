#!/usr/bin/env python3
"""
prototype_tooth_seg.py -- develop/validate the tooth segmentation algorithm
against real footage before porting it to JavaScript.

DEVELOPMENT TOOL, not part of the shipped app. The algorithm proven here is
reimplemented in src/core/ToothSegmenter.js; this script exists so it could be
iterated against mouthtestvideo.mp4 and measured, rather than guessed at in a
browser.

Algorithm (see README §"How tooth detection works"):
  1. rectify the mouth into a canonical, roll-free image using the Step-2 anchor
  2. aperture = inner lip ring polygon, eroded off the lips
  3. whiteness = V*(1-S/255): enamel is the only bright *neutral* surface inside
     a mouth, where everything else (lips, gums, tongue) is strongly red
  4. threshold at a high adaptive percentile of the aperture's own whiteness
  5. keep, per column, only the run nearest the aperture's top edge and the run
     nearest its bottom edge -- teeth line the aperture, the tongue floats in
     the middle, so this rejects the tongue structurally rather than by tuning
  6. split each arch at interdental gaps: local minima of per-column brightness

Run: project/.venv/bin/python step2_mouth_ar/tools/prototype_tooth_seg.py
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "mouthtestvideo.mp4"
MODEL = ROOT / "step2_mouth_ar" / "public" / "models" / "face_landmarker.task"
OUT = ROOT / "step2_mouth_ar" / "tools" / "_diag_output"

CORNER_L, CORNER_R, UPPER_OUTER, LOWER_OUTER = 61, 291, 0, 17
INNER_RING = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

RECT_W, RECT_H = 192, 144
PAD = 0.16
WHITE_PCT = 62.0      # keep the whitest ~38% of the aperture
MIN_OPEN = 0.10       # below this mouth-opening ratio, teeth are not visible
WHITE_REL = 0.52     # relative whiteness floor (fraction of aperture p99)
SPLIT_RATIO = 0.94    # interdental valley sensitivity (swept: 0.88->6.3, 0.94->8.2, 0.97->9.4 teeth)


def frame_basis(lm, w, h):
    def px(i):
        p = lm[i]
        return np.array([p.x * w, p.y * h, p.z * w])
    cl, cr, up, lo = px(CORNER_L), px(CORNER_R), px(UPPER_OUTER), px(LOWER_OUTER)
    xa = cr - cl
    width = float(np.linalg.norm(xa))
    if width < 1e-6:
        return None
    xa = xa / width
    ya = lo - up
    ya = ya - xa * np.dot(ya, xa)
    n = np.linalg.norm(ya)
    if n < 1e-6:
        return None
    ya = ya / n
    ring = [px(i) for i in INNER_RING]
    origin = np.mean(ring, axis=0)
    local = np.array([[np.dot(p - origin, xa) / width,
                       np.dot(p - origin, ya) / width] for p in ring])
    # opening ratio, same definition as Step 2's MouthTracker
    opening = float(np.linalg.norm(px(14) - px(13)) / width)
    return dict(origin=origin, xa=xa, ya=ya, width=width, local=local, opening=opening)


def roi_bounds(local):
    """ROI from the inner ring's own extent + padding: grows as the mouth opens
    instead of clipping the teeth (a fixed window did exactly that)."""
    u0, v0 = local.min(axis=0)
    u1, v1 = local.max(axis=0)
    du, dv = u1 - u0, v1 - v0
    return (u0 - du * PAD, u1 + du * PAD, v0 - dv * PAD, v1 + dv * PAD)


def rectify(frame, b, bounds):
    u0, u1, v0, v1 = bounds
    U, V = np.meshgrid(np.linspace(u0, u1, RECT_W), np.linspace(v0, v1, RECT_H))
    X = b["origin"][0] + b["width"] * (U * b["xa"][0] + V * b["ya"][0])
    Y = b["origin"][1] + b["width"] * (U * b["xa"][1] + V * b["ya"][1])
    return cv2.remap(frame, X.astype(np.float32), Y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def aperture_mask(b, bounds):
    u0, u1, v0, v1 = bounds
    pts = [[(u - u0) / (u1 - u0) * RECT_W, (v - v0) / (v1 - v0) * RECT_H]
           for u, v in b["local"]]
    m = np.zeros((RECT_H, RECT_W), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    # Pull in off the lips: the ring landmarks sit ON the lip edge, and the lit
    # lower lip is bright enough to be mistaken for enamel otherwise.
    k = max(3, int(RECT_H * 0.045)) | 1
    return cv2.erode(m, np.ones((k, k), np.uint8))


def whiteness_mask(rect, aperture):
    """Enamel is the only bright *neutral* surface inside the mouth."""
    if cv2.countNonZero(aperture) < 80:
        return None, 0.0
    hsv = cv2.cvtColor(cv2.GaussianBlur(rect, (5, 5), 0), cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    white = V * (1.0 - S / 255.0)

    inside = aperture > 0
    vals = white[inside]
    thr = float(np.percentile(vals, WHITE_PCT))
    # Absolute floor so a mouth containing *no* teeth (fully dark) cannot have
    # its darkest 38% promoted into a "detection" by the percentile alone, plus
    # a relative floor: enamel is the brightest neutral surface in a mouth, so
    # anything far darker than the brightest one is not enamel (this is what
    # stops a big red tongue being promoted to a "lower arch").
    thr = max(thr, 45.0, WHITE_REL * float(np.percentile(vals, 99)))

    m = ((white >= thr) & inside).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m, white


def arch_masks(mask, aperture):
    """Keep only the run nearest the aperture's top edge and the one nearest its
    bottom edge, per column. Teeth line the aperture; the tongue floats in the
    middle, so this removes it structurally instead of by threshold tuning."""
    upper = np.zeros_like(mask)
    lower = np.zeros_like(mask)
    h, w = mask.shape
    for x in range(w):
        col = mask[:, x] > 0
        ap = aperture[:, x] > 0
        if not ap.any() or not col.any():
            continue
        ys = np.where(ap)[0]
        ap_top, ap_bot = ys[0], ys[-1]
        ap_h = max(ap_bot - ap_top, 1)

        # contiguous runs of candidate pixels in this column
        runs, start = [], None
        for y in range(h):
            if col[y] and start is None:
                start = y
            elif not col[y] and start is not None:
                runs.append((start, y - 1)); start = None
        if start is not None:
            runs.append((start, h - 1))
        if not runs:
            continue

        top_run = min(runs, key=lambda r: r[0])
        bot_run = max(runs, key=lambda r: r[1])
        # A run only counts as an arch if it actually hugs the aperture edge.
        if (top_run[0] - ap_top) / ap_h < 0.42:
            upper[top_run[0]:top_run[1] + 1, x] = 255
        if (ap_bot - bot_run[1]) / ap_h < 0.42:
            lower[bot_run[0]:bot_run[1] + 1, x] = 255
    return upper, lower


def split_arch(arch, white):
    """Split one arch into individual teeth at the dark interdental lines.

    Crowns touch, so connected components merge them. The gaps are dark vertical
    lines -- minima in the per-column mean brightness of the arch -- which is a
    far stronger signal than the mask merely getting thinner.
    """
    cols = (arch > 0).sum(axis=0).astype(np.float32)
    if cols.sum() < 25:
        return []
    xs = np.where(cols > 0)[0]
    x_lo, x_hi = int(xs.min()), int(xs.max())
    span = x_hi - x_lo
    if span < 10:
        return []

    # mean whiteness per column, over arch pixels only
    prof = np.zeros(arch.shape[1], np.float32)
    for x in range(x_lo, x_hi + 1):
        sel = arch[:, x] > 0
        if sel.any():
            prof[x] = white[sel, x].mean()
    k = np.ones(3, np.float32) / 3
    prof = np.convolve(prof, k, mode="same")

    # combine "dark" and "thin" evidence; both mark an interdental gap
    thin = cols / max(cols.max(), 1)
    bright = prof / max(prof.max(), 1e-6)
    score = 0.65 * bright + 0.35 * thin       # low score => likely a gap

    win = max(2, int(span * 0.05))
    cuts = [x_lo]
    for x in range(x_lo + win, x_hi - win + 1):
        w = score[x - win:x + win + 1]
        if score[x] <= w.min() + 1e-6 and score[x] < SPLIT_RATIO * w.max() \
                and (x - cuts[-1]) >= win:
            cuts.append(x)
    cuts.append(x_hi + 1)

    out = []
    min_w = max(3, span * 0.05)
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if b - a < min_w:
            continue
        seg = np.zeros_like(arch)
        seg[:, a:b] = arch[:, a:b]
        if cv2.countNonZero(seg) < 18:
            continue
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 18:
            continue
        out.append(c)
    return out


def detect(frame, b):
    if b["opening"] < MIN_OPEN:
        return [], None, None, "mouth closed"
    bounds = roi_bounds(b["local"])
    rect = rectify(frame, b, bounds)
    ap = aperture_mask(b, bounds)
    m, white = whiteness_mask(rect, ap)
    if m is None:
        return [], rect, None, "aperture too small"
    upper, lower = arch_masks(m, ap)
    teeth = split_arch(upper, white) + split_arch(lower, white)
    return teeth, rect, (upper | lower), None


def main():
    if not VIDEO.exists():
        sys.exit(f"missing {VIDEO}")
    OUT.mkdir(parents=True, exist_ok=True)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    lmk = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=1))

    cap = cv2.VideoCapture(str(VIDEO))
    want = {150, 240, 300, 400, 540}
    tiles, counts, open_counts, i, faces = [], [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = lmk.detect_for_video(mpi, int(i * 1000 / 30))
        if res.face_landmarks:
            b = frame_basis(res.face_landmarks[0], w, h)
            if b:
                faces += 1
                teeth, rect, mask, why = detect(frame, b)
                counts.append(len(teeth))
                if b["opening"] >= MIN_OPEN:
                    open_counts.append(len(teeth))
                if i in want and rect is not None:
                    vis = rect.copy()
                    ov = vis.copy()
                    cv2.drawContours(ov, teeth, -1, (0, 255, 255), cv2.FILLED)
                    vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
                    cv2.drawContours(vis, teeth, -1, (0, 140, 255), 1)
                    for n, c in enumerate(teeth):
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cv2.putText(vis, str(n + 1),
                                        (int(M["m10"] / M["m00"]) - 3,
                                         int(M["m01"] / M["m00"]) + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(vis, f"f{i} n={len(teeth)} open={b['opening']:.2f}",
                                (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    mv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None \
                        else np.zeros_like(vis)
                    tiles.append(np.vstack([vis, mv]))
        i += 1
    cap.release()

    if tiles:
        cv2.imwrite(str(OUT / "proto_seg.png"), np.hstack(tiles))
    a = np.array(counts)
    o = np.array(open_counts) if open_counts else np.array([0])
    print(f"frames             : {i}, with face {faces}")
    print(f"mouth-open frames  : {len(o)}")
    print(f"teeth (open frames): mean {o.mean():.2f} median {np.median(o):.0f} "
          f"p10 {np.percentile(o,10):.0f} p90 {np.percentile(o,90):.0f} max {o.max()}")
    print(f"  >=4 teeth        : {(o>=4).mean()*100:.1f}%")
    print(f"  >=6 teeth        : {(o>=6).mean()*100:.1f}%")
    print(f"  zero             : {(o==0).mean()*100:.1f}%")
    print(f"wrote {OUT/'proto_seg.png'}")


if __name__ == "__main__":
    main()
