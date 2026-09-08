#!/usr/bin/env python3
"""
diag_channels.py -- measure what actually separates enamel from tongue/gums.

Rather than assuming "teeth are bright and unsaturated", this dumps the S and V
channels and a few candidate discriminants over the real mouth aperture so the
right one can be chosen from evidence. Development tool only.
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "mouthtestvideo.mp4"
OUT = ROOT / "step2_mouth_ar" / "tools" / "_diag_output"
MODEL = ROOT / "step2_mouth_ar" / "public" / "models" / "face_landmarker.task"

CORNER_L, CORNER_R, UPPER_OUTER, LOWER_OUTER = 61, 291, 0, 17
INNER_RING = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
RECT_W, RECT_H, PAD = 256, 192, 0.18


def frame_basis(lm, w, h):
    def px(i):
        p = lm[i]
        return np.array([p.x * w, p.y * h, p.z * w])
    cl, cr, up, lo = px(CORNER_L), px(CORNER_R), px(UPPER_OUTER), px(LOWER_OUTER)
    xa = cr - cl
    width = np.linalg.norm(xa)
    if width < 1e-6:
        return None
    xa /= width
    ya = lo - up
    ya -= xa * np.dot(ya, xa)
    n = np.linalg.norm(ya)
    if n < 1e-6:
        return None
    ya /= n
    ring = [px(i) for i in INNER_RING]
    origin = np.mean(ring, axis=0)
    local = [((np.dot(p - origin, xa)) / width, (np.dot(p - origin, ya)) / width)
             for p in ring]
    return origin, xa, ya, width, np.array(local)


def roi_bounds(local):
    """Adaptive ROI: the inner-lip ring's own extent plus padding, so the window
    grows as the mouth opens instead of clipping the teeth."""
    u0, v0 = local.min(axis=0)
    u1, v1 = local.max(axis=0)
    du, dv = (u1 - u0), (v1 - v0)
    u0 -= du * PAD; u1 += du * PAD
    v0 -= dv * PAD; v1 += dv * PAD
    return u0, u1, max(v0, -1.0), min(v1, 1.0)


def rectify(frame, basis, bounds):
    origin, xa, ya, width, _ = basis
    u0, u1, v0, v1 = bounds
    U, V = np.meshgrid(np.linspace(u0, u1, RECT_W), np.linspace(v0, v1, RECT_H))
    X = origin[0] + width * (U * xa[0] + V * ya[0])
    Y = origin[1] + width * (U * xa[1] + V * ya[1])
    return cv2.remap(frame, X.astype(np.float32), Y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def aperture_mask(basis, bounds):
    _, _, _, _, local = basis
    u0, u1, v0, v1 = bounds
    pts = [[(u - u0) / (u1 - u0) * RECT_W, (v - v0) / (v1 - v0) * RECT_H]
           for u, v in local]
    m = np.zeros((RECT_H, RECT_W), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return cv2.erode(m, np.ones((3, 3), np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    lmk = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=1))

    cap = cv2.VideoCapture(str(VIDEO))
    want = {150, 240, 300, 540}
    tiles, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            h, w = frame.shape[:2]
            mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(mpi, int(i * 1000 / 30))
            if res.face_landmarks:
                b = frame_basis(res.face_landmarks[0], w, h)
                if b:
                    bounds = roi_bounds(b[4])
                    rect = rectify(frame, b, bounds)
                    ap = aperture_mask(b, bounds)
                    hsv = cv2.cvtColor(cv2.GaussianBlur(rect, (5, 5), 0), cv2.COLOR_BGR2HSV)
                    S, V = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
                    white = V * (1.0 - S / 255.0)      # candidate discriminant

                    inside = ap > 0
                    print(f"\n--- frame {i}  aperture px={inside.sum()} ---")
                    for name, ch in (("S", S), ("V", V), ("whiteness", white)):
                        q = np.percentile(ch[inside], [10, 25, 50, 75, 90])
                        print(f"  {name:9s} p10..p90: " + " ".join(f"{v:6.1f}" for v in q))

                    ot, _ = cv2.threshold(white[inside].astype(np.uint8), 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    seg = ((white >= ot) & inside).astype(np.uint8) * 255
                    print(f"  otsu(whiteness)={ot:.0f}  -> {seg.sum()//255} px "
                          f"({(seg>0).sum()/max(inside.sum(),1)*100:.0f}% of aperture)")

                    def gray(x, lo=0, hi=255):
                        g = np.clip((x - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
                        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    lab = rect.copy()
                    cv2.putText(lab, f"f{i}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1)
                    tiles.append(np.vstack([lab, gray(S), gray(V), gray(white),
                                            cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)]))
        i += 1
    cap.release()
    if tiles:
        cv2.imwrite(str(OUT / "diag_channels.png"), np.hstack(tiles))
        print(f"\nwrote {OUT/'diag_channels.png'}  (rows: rect, S, V, whiteness, otsu)")


if __name__ == "__main__":
    main()
