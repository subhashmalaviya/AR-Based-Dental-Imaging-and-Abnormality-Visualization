#!/usr/bin/env python3
"""
make_valid_fixtures.py -- DentalAI *validation* crops in eval-harness format.

Decoder thresholds are tuned on these, never on the test sets the README
reports, so the reported numbers are not fitted to their own ground truth.
"""
import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

EW, EH = 192, 144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    d = np.load(a.npz)
    img, inst = d["img"], d["inst"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, gt = [], {"frames": {}}
    step = max(1, len(img) // a.limit)
    for k in range(0, len(img), step):
        fid = f"dav_{k:05d}"
        H, W = img[k].shape[:2]
        (out / f"{fid}_{W}x{H}.rgb.z").write_bytes(zlib.compress(img[k].tobytes(), 6))
        big = cv2.resize(img[k], (EW, EH), interpolation=cv2.INTER_LINEAR)
        (out / f"{fid}_{EW}x{EH}.rgb.z").write_bytes(zlib.compress(big.tobytes(), 6))
        lab = cv2.resize(inst[k], (EW, EH), interpolation=cv2.INTER_NEAREST)
        teeth = []
        for v in range(1, int(lab.max()) + 1):
            cs, _ = cv2.findContours((lab == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) < 6:
                continue
            c = c[:, 0, :].astype(float) + 0.5
            x, y, w, h = cv2.boundingRect(c.astype(np.float32))
            teeth.append({"box": [x, y, w, h], "polygon": c.round(1).tolist()})
        frames.append({"id": fid, "video": "DentalAI-valid", "frame": k,
                       "bounds": {"u0": 0, "u1": 1, "v0": 0, "v1": 0.75},
                       "localRing": [[0, 0], [1, 0], [1, 0.75], [0, 0.75]], "noAperture": True})
        gt["frames"][fid] = {"teeth": teeth}
        if len(frames) >= a.limit:
            break
    json.dump(frames, open(out / "frames.json", "w"))
    json.dump(gt, open(out / "gt.json", "w"))
    print(f"{len(frames)} validation fixtures, {sum(len(v['teeth']) for v in gt['frames'].values())} teeth")


if __name__ == "__main__":
    main()
