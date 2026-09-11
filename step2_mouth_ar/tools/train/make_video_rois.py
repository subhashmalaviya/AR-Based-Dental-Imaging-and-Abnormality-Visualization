#!/usr/bin/env python3
"""
make_video_rois.py -- rectified mouth ROIs from real recordings.

Produces the evaluation set: for sampled frames of a real video, the exact
mouth crop the app's detectors see (same rectification as MouthROI.js), at
both detector input sizes, plus an enlarged grid image for manual annotation
of ground-truth teeth.

Output (per frame id "<video>_f<frame>"):
  <out>/<id>_192x144.rgb.z   raw RGB, zlib-compressed (read by tools/eval_detectors.mjs)
  <out>/<id>_160x120.rgb.z
  <out>/<id>_grid.png        4x enlarged 192x144 ROI with a 16 px coordinate grid
  <out>/frames.json          geometry per frame (bounds, inner ring, opening)
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_geometry import (make_landmarker, mouth_frame, rectify,  # noqa: E402
                          roi_bounds)

SIZES = [(192, 144), (160, 120)]


def grid_image(big, scale=4, step=16):
    """`big` is the ROI rectified at scale x 192x144 straight from the source
    frame (not an upscaled crop), so annotation sees every real pixel."""
    # Local contrast enhancement (CLAHE on lightness) makes the faint
    # interdental lines visible for annotation. Annotation aid only: the
    # detectors are evaluated on the unmodified crops.
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(lab[..., 0])
    big = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h, w = big.shape[0] // scale, big.shape[1] // scale
    for x in range(0, w + 1, step):
        col = (0, 220, 255) if x % 64 == 0 else (90, 90, 90)
        cv2.line(big, (x * scale, 0), (x * scale, h * scale), col, 1)
        cv2.putText(big, str(x), (x * scale + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    for y in range(0, h + 1, step):
        col = (0, 220, 255) if y % 64 == 0 else (90, 90, 90)
        cv2.line(big, (0, y * scale), (w * scale, y * scale), col, 1)
        cv2.putText(big, str(y), (2, y * scale - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video[:every] e.g. mouth.mp4:10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="face_landmarker.task")
    ap.add_argument("--min-open", type=float, default=0.08)
    ap.add_argument("--pad-top", type=float, default=None, help="top padding (default: same as the sides)")
    ap.add_argument("--pad-bottom", type=float, default=None, help="bottom padding (default: same as the sides)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in a.videos:
        path, every = (spec.rsplit(":", 1) + ["10"])[:2] if ":" in spec else (spec, "10")
        every = int(every)
        name = Path(path).stem
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        run = make_landmarker(a.model, video=True)
        i = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = i * 1000.0 / fps
            lm = run(frame, t_ms)
            if lm is not None and i % every == 0:
                h, w = frame.shape[:2]
                fr = mouth_frame(lm, w, h)
                if fr is not None and fr["opening"] >= a.min_open:
                    b = roi_bounds(fr, pad_top=a.pad_top, pad_bottom=a.pad_bottom)
                    fid = f"{name}_f{i:04d}"
                    rec = {"id": fid, "video": Path(path).name, "frame": i, "t_ms": round(t_ms, 1),
                           "frame_size": [w, h], "opening": round(fr["opening"], 4),
                           "mouth_width_px": round(fr["width"], 1),
                           "bounds": {"u0": b[0], "u1": b[1], "v0": b[2], "v1": b[3]},
                           "localRing": [[round(u, 5), round(v, 5)] for u, v in fr["local"]]}
                    for (W, H) in SIZES:
                        roi = rectify(frame, fr, b, W, H)
                        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        (out / f"{fid}_{W}x{H}.rgb.z").write_bytes(zlib.compress(rgb.tobytes(), 6))
                        if (W, H) == (192, 144):
                            big = rectify(frame, fr, b, W * 4, H * 4, cv2.INTER_CUBIC)
                            cv2.imwrite(str(out / f"{fid}_grid.png"), grid_image(big))
                    records.append(rec)
                    kept += 1
            i += 1
        print(f"{name}: {i} frames, kept {kept}", flush=True)
    json.dump(records, open(out / "frames.json", "w"), indent=1)


if __name__ == "__main__":
    main()
