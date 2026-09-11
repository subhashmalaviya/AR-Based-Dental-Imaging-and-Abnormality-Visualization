#!/usr/bin/env python3
"""
train_tooth_model.py -- train ToothNet-lite, the on-device tooth segmenter.

Architecture (≈0.1 M params, ≈80 M multiply-adds at 160x120): a small U-Net
whose first convolution has stride 2, so every feature map lives at half
resolution or below and the network stays fast enough for a phone's
WebAssembly runtime. Three output maps, upsampled back to 160x120 in-graph:

  teeth     semantic enamel mask            (all samples)
  center    Gaussian peak per tooth          (DentalAI only: needs instances)
  boundary  interdental border between teeth (DentalAI only)

EasyPortrait samples carry no per-tooth labels, so their centre/boundary
losses are masked to zero -- they teach appearance, DentalAI teaches
separation. Heavy photometric / low-resolution / blur / JPEG augmentation
bridges the gap between clinical photos and a phone front camera.

Exports tooth_seg.onnx (sigmoid applied in-graph) plus tooth_seg.json with
input spec, decode parameters, dataset provenance and validation metrics.
"""
import argparse
import datetime as dt
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W, H = 160, 120
MEAN, STD = 0.5, 0.25


# ------------------------------------------------------------------ model
def cbr(i, o, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride=s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ToothNet(nn.Module):
    def __init__(self, c=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = c
        self.stem = nn.Sequential(cbr(3, c1, s=2), cbr(c1, c1))                    # 80x60
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), cbr(c1, c2), cbr(c2, c2))      # 40x30
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), cbr(c2, c3),                   # 20x15
                                  cbr(c3, c3, d=2), cbr(c3, c3, d=4))
        self.red2 = cbr(c3 + c2, c2, k=1)
        self.dec2 = cbr(c2, c2)
        self.red1 = cbr(c2 + c1, c1, k=1)
        self.dec1 = cbr(c1, c1)
        self.head = nn.Conv2d(c1, 3, 1)
        # CenterNet-style prior: centre and boundary pixels are rare, so start
        # their logits near p=0.1 instead of 0.5 (avoids a huge initial loss).
        with torch.no_grad():
            self.head.bias[1] = -2.19
            self.head.bias[2] = -2.0

    def forward(self, x):
        a = self.stem(x)
        b = self.enc2(a)
        c = self.enc3(b)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        b = self.dec2(self.red2(torch.cat([up(c), b], 1)))
        a = self.dec1(self.red1(torch.cat([up(b), a], 1)))
        return up(self.head(a))                                                     # 160x120 logits


class Exported(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, roi):
        return torch.sigmoid(self.net(roi))


# ---------------------------------------------------------------- targets
def centers_and_boundary(inst):
    """Gaussian centre heatmap + inter-tooth boundary mask from an instance map."""
    heat = np.zeros(inst.shape, np.float32)
    info = []
    for v in range(1, int(inst.max()) + 1):
        ys, xs = np.nonzero(inst == v)
        if len(xs) < 4:
            continue
        cx, cy = xs.mean(), ys.mean()
        sig = float(np.clip(0.16 * math.sqrt(len(xs)), 1.0, 4.0))
        r = int(3 * sig) + 1
        x0, x1 = max(0, int(cx) - r), min(inst.shape[1], int(cx) + r + 2)
        y0, y1 = max(0, int(cy) - r), min(inst.shape[0], int(cy) + r + 2)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sig * sig))
        g /= g.max()
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
        info.append((cx, cy, len(xs)))
    b = np.zeros(inst.shape, bool)
    a, c = inst[:, 1:], inst[:, :-1]
    m = (a > 0) & (c > 0) & (a != c)
    b[:, 1:] |= m
    b[:, :-1] |= m
    a, c = inst[1:, :], inst[:-1, :]
    m = (a > 0) & (c > 0) & (a != c)
    b[1:, :] |= m
    b[:-1, :] |= m
    return heat, b.astype(np.float32), info


# ----------------------------------------------------------- augmentation
LOWLIGHT = False   # set by --lowlight: extra dim-room augmentation
WEBCAM = False     # set by --webcam: laptop-webcam / recorded-video degradation


def augment(img, lab, rng):
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    # geometry
    ang = rng.uniform(-7, 7)
    sc = rng.uniform(0.9, 1.12)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    M[:, 2] += (rng.uniform(-0.05, 0.05) * W, rng.uniform(-0.05, 0.05) * H)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    lab = cv2.warpAffine(np.ascontiguousarray(lab), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    # photometric
    x = img.astype(np.float32) / 255
    x *= np.array([rng.uniform(0.88, 1.12) for _ in range(3)], np.float32)        # white balance
    x = x * rng.uniform(0.6, 1.35)                                                 # exposure
    m = x.mean()
    x = (x - m) * rng.uniform(0.7, 1.3) + m                                        # contrast
    g = x.mean(2, keepdims=True)
    x = g + (x - g) * rng.uniform(0.6, 1.4)                                        # saturation
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.45)                                 # gamma
    if LOWLIGHT and rng.random() < 0.45:
        # Dim room lit by a warm lamp, seen by a laptop/phone front camera:
        # strong under-exposure, orange cast, and the sensor noise that the
        # camera's auto-gain amplifies in the dark.
        x = x * rng.uniform(0.22, 0.6)
        x = x * np.array([rng.uniform(1.0, 1.25), rng.uniform(0.9, 1.05), rng.uniform(0.65, 0.9)], np.float32)
        x = x + np.random.normal(0, rng.uniform(0.01, 0.035), x.shape).astype(np.float32)
    img = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.4:                                                         # low-res phone
        f = rng.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4))
    if rng.random() < 0.15:                                                        # motion blur
        k = rng.choice([3, 5, 7])
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1), (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    if rng.random() < 0.35:
        img = np.clip(img + np.random.normal(0, rng.uniform(2, 9), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    if WEBCAM and rng.random() < 0.5:
        # 720p laptop webcam at arm's length, then video compression: the mouth
        # is small (few pixels per tooth) and dark regions go blocky.
        f = rng.uniform(0.28, 0.55)
        small = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, rng.randint(12, 45)])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)


class Mixed(torch.utils.data.Dataset):
    """Samples: ('da', idx) with instance labels, ('ep', idx) semantic only."""

    def __init__(self, da, ep, items, train):
        self.da, self.ep, self.items, self.train = da, ep, items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        src, i = self.items[k]
        rng = random.Random((k * 7919 + i) ^ random.getrandbits(30)) if self.train else random.Random(k)
        if src == "da":
            img, lab = self.da["img"][i], self.da["inst"][i]
        else:
            img, lab = self.ep["img"][i], self.ep["sem"][i]
        if self.train:
            img, lab = augment(img, lab, rng)
        x = (img.astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        sem = torch.from_numpy((lab > 0).astype(np.float32))
        if src == "da":
            heat, bnd, _ = centers_and_boundary(lab)
            w = 1.0
        else:
            heat = np.zeros(lab.shape, np.float32)
            bnd = np.zeros(lab.shape, np.float32)
            w = 0.0
        return x, sem, torch.from_numpy(heat), torch.from_numpy(bnd), torch.tensor(w)


# ------------------------------------------------------------------ losses
def dice_loss(logit, t):
    p = torch.sigmoid(logit)
    inter = (p * t).sum((1, 2))
    return (1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()


def center_focal(logit, heat, w):
    """CenterNet penalty-reduced focal loss, per sample weight `w`."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (heat > 0.99).float()
    neg_w = (1 - heat) ** 4
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum((1, 2))
    ln = -(torch.log(1 - p) * p ** 2 * neg_w * (1 - pos)).sum((1, 2))
    npos = pos.sum((1, 2)).clamp(min=1)
    return ((lp + ln) / npos * w).sum() / w.sum().clamp(min=1)


def boundary_bce(logit, bnd, w):
    l = F.binary_cross_entropy_with_logits(logit, bnd, pos_weight=torch.tensor(4.0), reduction="none")
    return (l.mean((1, 2)) * w).sum() / w.sum().clamp(min=1)


# -------------------------------------------------------------- validation
@torch.no_grad()
def evaluate(net, da, ep, bs=64):
    net.eval()
    res = {}
    for name, d, key in (("ep", ep, "sem"), ("da", da, "inst")):
        if d is None or len(d["img"]) == 0:
            continue
        inter = union = 0
        tp = fp = fn = 0
        for s in range(0, len(d["img"]), bs):
            imgs = d["img"][s:s + bs]
            x = torch.from_numpy(((imgs.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2).copy())
            p = torch.sigmoid(net(x))
            sem = (p[:, 0] > 0.5).numpy()
            gt = d[key][s:s + bs] > 0
            inter += (sem & gt).sum()
            union += (sem | gt).sum()
            if name == "da":
                ctr = p[:, 1]
                peaks = (ctr == F.max_pool2d(ctr[:, None], 7, 1, 3)[:, 0]) & (ctr > 0.25) & (p[:, 0] > 0.5)
                for j in range(len(imgs)):
                    _, _, info = centers_and_boundary(d[key][s + j])
                    py, px = np.nonzero(peaks[j].numpy())
                    used = set()
                    for cx, cy, area in info:
                        r = 0.5 * math.sqrt(area)
                        best, bd = -1, r
                        for q in range(len(px)):
                            if q in used:
                                continue
                            dd = math.hypot(px[q] - cx, py[q] - cy)
                            if dd <= bd:
                                best, bd = q, dd
                        if best >= 0:
                            used.add(best)
                            tp += 1
                        else:
                            fn += 1
                    fp += len(px) - len(used)
        res[f"{name}_teeth_iou"] = float(inter / max(union, 1))
        if name == "da":
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            res.update(da_center_precision=pr, da_center_recall=rc,
                       da_center_f1=2 * pr * rc / max(pr + rc, 1e-9))
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--init", help="fine-tune from this state_dict (.pt)")
    ap.add_argument("--lowlight", action="store_true", help="extra low-light augmentation")
    ap.add_argument("--webcam", action="store_true", help="webcam / video-compression augmentation")
    ap.add_argument("--pad-top", type=float, default=None, help="ROI top padding the EP crops were built with (recorded in the model card)")
    a = ap.parse_args()
    global LOWLIGHT, WEBCAM
    LOWLIGHT, WEBCAM = a.lowlight, a.webcam

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    load = lambda n: dict(np.load(data / n)) if (data / n).exists() else None  # noqa: E731
    da_tr, da_va = load("da_train.npz"), load("da_valid.npz")
    ep_tr, ep_va = load("ep_train.npz"), load("ep_val.npz")
    n_da = len(da_tr["img"])
    n_ep = len(ep_tr["img"]) if ep_tr else 0
    print(f"train: DentalAI {n_da} crops, EasyPortrait {n_ep} crops", flush=True)

    net = ToothNet()
    if a.init:
        net.load_state_dict(torch.load(a.init))
        print(f"fine-tuning from {a.init}", flush=True)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ToothNet-lite: {n_params} parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total_steps = a.epochs * math.ceil(a.per_epoch / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.1)

    best, best_score, history = None, -1, []
    for epoch in range(a.epochs):
        # Balanced sampling: half instance-labelled, half selfie-domain.
        half = a.per_epoch // 2 if n_ep else a.per_epoch
        items = [("da", random.randrange(n_da)) for _ in range(half)]
        items += [("ep", random.randrange(n_ep)) for _ in range(a.per_epoch - half)] if n_ep else []
        random.shuffle(items)
        dl = torch.utils.data.DataLoader(Mixed(da_tr, ep_tr, items, True), batch_size=a.bs,
                                         num_workers=a.workers, drop_last=True, persistent_workers=False)
        t0, agg = time.time(), np.zeros(4)
        for x, sem, heat, bnd, w in dl:
            y = net(x)
            ls = F.binary_cross_entropy_with_logits(y[:, 0], sem) + dice_loss(y[:, 0], sem)
            lc = center_focal(y[:, 1], heat, w)
            lb = boundary_bce(y[:, 2], bnd, w)
            loss = ls + lc + 0.5 * lb
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            agg += [loss.item(), ls.item(), lc.item(), lb.item()]
        agg /= max(1, len(dl))
        v = evaluate(net, da_va, ep_va)
        score = v.get("ep_teeth_iou", 0) + v.get("da_teeth_iou", 0) + v.get("da_center_f1", 0)
        rec = {"epoch": epoch + 1, "loss": round(agg[0], 4), "sem": round(agg[1], 4),
               "center": round(agg[2], 4), "boundary": round(agg[3], 4),
               **{k: round(val, 4) for k, val in v.items()}, "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if score > best_score:
            best_score, best = score, {k: t.clone() for k, t in net.state_dict().items()}
            torch.save(best, out / "tooth_seg_best.pt")

    net.load_state_dict(best)
    val = evaluate(net, da_va, ep_va)
    net.eval()          # evaluate() leaves the net in train mode; export must use BN running stats
    onnx_path = out / "tooth_seg.onnx"
    dummy = torch.zeros(1, 3, H, W)
    wrapped = Exported(net).eval()
    torch.onnx.export(wrapped, dummy, str(onnx_path), input_names=["roi"], output_names=["maps"],
                      opset_version=17, dynamo=False)
    # torch.onnx.export restores the module's previous training flag
    # recursively on exit; pin eval mode again for the reference comparison.
    net.eval()

    # Verify the exported graph reproduces PyTorch.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xs = torch.randn(1, 3, H, W)
    with torch.no_grad():
        ref = torch.sigmoid(net(xs)).numpy()
    got = sess.run(None, {"roi": xs.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    print(f"ONNX vs PyTorch max |diff| = {max_diff:.2e}", flush=True)

    info = {
        "name": "ToothNet-lite",
        "version": "1.0",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "task": "tooth instance segmentation on the rectified mouth ROI "
                "(teeth mask + tooth-centre heatmap + interdental boundary)",
        "input": {"name": "roi", "width": W, "height": H, "layout": "NCHW", "color": "RGB",
                  "mean": [MEAN] * 3, "std": [STD] * 3},
        "outputs": {"name": "maps", "channels": ["teeth", "center", "boundary"], "activation": "sigmoid"},
        "roi": {"padding": 0.16, "padTop": a.pad_top if a.pad_top is not None else 0.16},
        "decode": {"semThr": 0.5, "ctrThr": 0.25, "peakRadius": 3, "boundaryWeight": 10},
        "parameters": int(n_params),
        "training": {
            "epochs": a.epochs, "samples_per_epoch": a.per_epoch, "batch": a.bs, "seed": a.seed,
            "init": a.init, "lowlight_augmentation": a.lowlight, "webcam_augmentation": a.webcam,
            "torch": torch.__version__,
            "datasets": [
                {"name": "DentalAI", "license": "CC BY 4.0", "author": "Pawan Valluri (2023)",
                 "url": "https://www.kaggle.com/datasets/pawanvalluri/dental-segmentation",
                 "mirror": "https://datasetninja.com/dentalai", "crops": n_da,
                 "used_for": "per-tooth instance supervision (Tooth class only)"},
                {"name": "EasyPortrait", "license": "CC BY-SA 4.0 (variant, see dataset licence)",
                 "author": "Kvanchiani et al. (2023)",
                 "url": "https://huggingface.co/datasets/gofixyourself/EasyPortrait", "crops": n_ep,
                 "used_for": "teeth appearance in selfie images (TEETH class, semantic only)"},
            ],
        },
        "validation": {k: round(v2, 4) for k, v2 in val.items()},
        "onnx_max_abs_diff": max_diff,
        "history": history,
        "limitations": [
            "No upper/lower jaw labels in training data; jaw is assigned geometrically in the app.",
            "DentalAI is clinical intraoral photography; EasyPortrait provides selfie appearance "
            "but no per-tooth labels, so tooth separation in selfie images is learned by transfer.",
            "Not a diagnostic device. Does not detect caries or any pathology.",
        ],
    }
    json.dump(info, open(out / "tooth_seg.json", "w"), indent=1)
    print(json.dumps(info["validation"]), flush=True)


if __name__ == "__main__":
    main()
