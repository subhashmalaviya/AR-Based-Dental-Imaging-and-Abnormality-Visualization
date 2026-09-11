#!/usr/bin/env python3
"""
build_dataset.py -- training / evaluation crops for the tooth model.

Two sources, each contributing what it is actually good for:

  DentalAI (CC BY 4.0)       intraoral photos with one polygon PER TOOTH.
                             -> instance supervision: teeth mask, tooth
                                centres, interdental boundaries.
  EasyPortrait (CC BY-SA 4.0 selfie portraits with a pixel TEETH class.
  variant)                   -> appearance supervision in the app's domain.
                                Cropped with the app's own mouth
                                rectification (roi_geometry.py), so the model
                                trains on exactly the view it will see live.
                                No per-tooth labels: instance losses are
                                masked out for these samples.

Outputs (in --out):
  da_{train,valid,test}.npz   img (N,H,W,3) RGB uint8, inst (N,H,W) uint8
                              (0 = background, k = tooth k)
  ep_{train,val,test}.npz     img, sem (N,H,W) uint8 (1 = teeth)
  fixtures_ep_test/           EasyPortrait test crops in the eval-harness
                              format (frames.json + .rgb.z + teeth mask .u8.z)
  fixtures_da_test/           DentalAI test crops + per-tooth polygon GT
"""
import argparse
import json
import random
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_geometry import (make_landmarker, mouth_frame, rectify,  # noqa: E402
                          roi_bounds)

W, H = 160, 120            # model input
EW, EH = 192, 144          # evaluation fixtures (classical detector size)


# ------------------------------------------------------------------ DentalAI
def dentalai_items(tar_path):
    tf = tarfile.open(tar_path)
    members = {m.name: m for m in tf.getmembers() if m.isfile()}
    for name, m in members.items():
        if "/ann/" not in name or not name.endswith(".json"):
            continue
        split = name.split("/")[0]
        img_name = name.replace("/ann/", "/img/")[:-5]
        if img_name not in members:
            continue
        ann = json.load(tf.extractfile(m))
        teeth = [o["points"]["exterior"] for o in ann["objects"]
                 if o["classTitle"] == "Tooth" and o["geometryType"] == "polygon"
                 and len(o["points"]["exterior"]) >= 3]
        if not teeth:
            continue
        buf = np.frombuffer(tf.extractfile(members[img_name]).read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield split, Path(img_name).name, img, teeth


def instance_map(shape, polys):
    inst = np.zeros(shape[:2], np.uint16)
    # Larger teeth first so a small tooth drawn on top keeps its own pixels.
    order = sorted(range(len(polys)), key=lambda k: -cv2.contourArea(np.array(polys[k], np.float32)))
    for rank, k in enumerate(order, start=1):
        cv2.fillPoly(inst, [np.round(np.array(polys[k])).astype(np.int32)], int(rank))
    return inst


def crop_boxes(inst, rng, n, aspect=W / H):
    """Crops around the dentition, at several zoom levels (4:3, like the ROI)."""
    ys, xs = np.nonzero(inst)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    out = []
    for k in range(n):
        if k == 0:
            w = max(bw, bh * aspect) * rng.uniform(1.05, 1.3)
            ccx, ccy = cx, cy
        else:  # zoom in on part of the arch: bigger teeth, fewer per crop
            w = max(bw, bh * aspect) * rng.uniform(0.45, 0.85)
            ccx = cx + rng.uniform(-0.3, 0.3) * bw
            ccy = cy + rng.uniform(-0.2, 0.2) * bh
        h = w / aspect
        out.append((ccx - w / 2, ccy - h / 2, w, h))
    return out


def crop(img, inst, box, ow, oh):
    x, y, w, h = box
    M = np.array([[ow / w, 0, -x * ow / w], [0, oh / h, -y * oh / h]], np.float32)
    im = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_AREA if w > ow else cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    lab = cv2.warpAffine(inst, M, (ow, oh), flags=cv2.INTER_NEAREST, borderValue=0)
    return im, lab


def relabel(lab, min_px=6):
    """Consecutive labels 1..k, dropping slivers; returns uint8."""
    out = np.zeros(lab.shape, np.uint8)
    k = 0
    for v in np.unique(lab):
        if v == 0:
            continue
        m = lab == v
        if m.sum() < min_px or k >= 254:
            continue
        k += 1
        out[m] = k
    return out


def build_dentalai(tar_path, out, rng, crops_per_image):
    data = {"train": ([], []), "valid": ([], []), "test": ([], [])}
    fix_dir = out / "fixtures_da_test"
    fix_dir.mkdir(parents=True, exist_ok=True)
    fixtures, gt = [], {"frames": {}}
    sizes = []
    for split, name, img, teeth in dentalai_items(tar_path):
        sizes.append(img.shape[:2])
        inst = instance_map(img.shape, teeth)
        n = 1 if split == "test" else crops_per_image
        for bi, box in enumerate(crop_boxes(inst, rng, n)):
            im, lab = crop(img, inst, box, W, H)
            lab = relabel(lab)
            if lab.max() == 0:
                continue
            data[split][0].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            data[split][1].append(lab)
            if split == "test" and bi == 0:
                fid = f"da_{Path(name).stem[:40]}"
                im_e, lab_e = crop(img, inst, box, EW, EH)
                lab_e = relabel(lab_e)
                for (ow, oh), im_s in (((EW, EH), im_e), ((W, H), im)):
                    rgb = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)
                    (fix_dir / f"{fid}_{ow}x{oh}.rgb.z").write_bytes(zlib.compress(rgb.tobytes(), 6))
                ring = [[0, 0], [1, 0], [1, 0.75], [0, 0.75]]
                fixtures.append({"id": fid, "video": "DentalAI-test", "frame": 0,
                                 "bounds": {"u0": 0, "u1": 1, "v0": 0, "v1": 0.75},
                                 "localRing": ring, "noAperture": True})
                polys = []
                for v in range(1, lab_e.max() + 1):
                    cs, _ = cv2.findContours((lab_e == v).astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
                    if not cs:
                        continue
                    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) + 0.5
                    x, y, w, h = cv2.boundingRect(c.astype(np.float32))
                    polys.append({"box": [x, y, w, h], "polygon": c.round(1).tolist()})
                gt["frames"][fid] = {"teeth": polys}
    for split, (ims, labs) in data.items():
        np.savez_compressed(out / f"da_{split}.npz", img=np.array(ims, np.uint8),
                            inst=np.array(labs, np.uint8))
        print(f"DentalAI {split}: {len(ims)} crops", flush=True)
    json.dump(fixtures, open(fix_dir / "frames.json", "w"))
    json.dump(gt, open(fix_dir / "gt.json", "w"))
    s = np.array(sizes)
    print(f"DentalAI image sizes: median {np.median(s, 0)}, min {s.min(0)}, max {s.max(0)}", flush=True)


# -------------------------------------------------------------- EasyPortrait
def jitter(fr, b, rng):
    ang = np.deg2rad(rng.uniform(-8, 8))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    xa, ya = fr["xa"].copy(), fr["ya"].copy()
    xa[:2], ya[:2] = R @ fr["xa"][:2], R @ fr["ya"][:2]
    u0, u1, v0, v1 = b
    sc = rng.uniform(0.88, 1.15)
    cu = (u0 + u1) / 2 + rng.uniform(-0.06, 0.06) * (u1 - u0)
    cv_ = (v0 + v1) / 2 + rng.uniform(-0.06, 0.06) * (v1 - v0)
    hu, hv = (u1 - u0) / 2 * sc, (v1 - v0) / 2 * sc
    return dict(fr, xa=xa, ya=ya), (cu - hu, cu + hu, cv_ - hv, cv_ + hv)


def build_easyportrait(ep_dir, ann_zip, model, out, rng, crops_per_image):
    manifest = json.load(open(ep_dir / "manifest.json"))
    zf = zipfile.ZipFile(ann_zip)
    run = make_landmarker(model, video=False)
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    fix_dir = out / "fixtures_ep_test"
    fix_dir.mkdir(parents=True, exist_ok=True)
    fixtures = []
    stats = {"no_file": 0, "no_face": 0, "ok": 0}
    for rec in manifest:
        split = rec["mask"].split("/")[0]
        ip = ep_dir / rec["image"]
        if not ip.exists():
            stats["no_file"] += 1
            continue
        img = cv2.imread(str(ip))
        mask = cv2.imdecode(np.frombuffer(zf.read(rec["mask"]), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            stats["no_file"] += 1
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Landmarks on a downscaled copy (normalised coords are resolution-free).
        scale = 960 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        lm = run(small)
        if lm is None:
            stats["no_face"] += 1
            continue
        h, w = img.shape[:2]
        fr = mouth_frame(lm, w, h)
        if fr is None:
            stats["no_face"] += 1
            continue
        b = roi_bounds(fr)
        teeth = (mask == 8).astype(np.uint8)
        n = 1 if split == "test" else crops_per_image
        for k in range(n):
            frk, bk = (fr, b) if k == 0 else jitter(fr, b, rng)
            im = rectify(img, frk, bk, W, H)
            sm = rectify(teeth, frk, bk, W, H, cv2.INTER_NEAREST)
            data[split][0].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            data[split][1].append(sm)
        if split == "test":
            fid = f"ep_{Path(rec['image']).stem[:36]}"
            for (ow, oh) in ((EW, EH), (W, H)):
                rgb = cv2.cvtColor(rectify(img, fr, b, ow, oh), cv2.COLOR_BGR2RGB)
                (fix_dir / f"{fid}_{ow}x{oh}.rgb.z").write_bytes(zlib.compress(rgb.tobytes(), 6))
            gm = rectify(teeth, fr, b, EW, EH, cv2.INTER_NEAREST)
            (fix_dir / f"{fid}_mask.u8.z").write_bytes(zlib.compress(gm.tobytes(), 6))
            fixtures.append({"id": fid, "video": "EasyPortrait-test", "frame": 0,
                             "kind": rec["kind"], "opening": round(fr["opening"], 4),
                             "bounds": {"u0": b[0], "u1": b[1], "v0": b[2], "v1": b[3]},
                             "localRing": [[round(u, 5), round(v, 5)] for u, v in fr["local"]]})
        stats["ok"] += 1
    for split, (ims, sems) in data.items():
        np.savez_compressed(out / f"ep_{split}.npz", img=np.array(ims, np.uint8),
                            sem=np.array(sems, np.uint8))
        print(f"EasyPortrait {split}: {len(ims)} crops", flush=True)
    json.dump(fixtures, open(fix_dir / "frames.json", "w"))
    print(f"EasyPortrait: {stats}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dentalai-tar")
    ap.add_argument("--ep-dir")
    ap.add_argument("--ep-ann")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    if a.dentalai_tar:
        build_dentalai(a.dentalai_tar, out, rng, a.crops)
    if a.ep_dir:
        build_easyportrait(Path(a.ep_dir), a.ep_ann, a.model, out, rng, a.crops)


if __name__ == "__main__":
    main()
