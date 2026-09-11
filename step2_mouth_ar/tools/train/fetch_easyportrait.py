#!/usr/bin/env python3
"""
fetch_easyportrait.py -- pull a teeth-focused subset of EasyPortrait without
downloading the 28 GB image archive.

EasyPortrait (Kvanchiani et al., CC BY-SA 4.0 variant) is a face-parsing
dataset of ~20k selfie-style portraits with a pixel-level TEETH class (label 8).
It is the closest public data to what this app actually sees: a phone front
camera looking at a smiling / open mouth. Only a few thousand of its images are
useful here (visible teeth, plus some closed-mouth negatives), so rather than
fetch the whole archive this script:

  1. reads the ZIP central directory over HTTP range requests,
  2. picks members using per-mask statistics (mask_stats.json, produced from
     the 246 MB annotation archive),
  3. fetches just those members' byte ranges in parallel and inflates them.

Source: https://huggingface.co/datasets/gofixyourself/EasyPortrait
Paper:  https://arxiv.org/abs/2304.13509
"""
import argparse
import concurrent.futures as cf
import io
import json
import random
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

IMAGES_URL = ("https://huggingface.co/datasets/gofixyourself/EasyPortrait/"
              "resolve/main/data/images.zip")


def http_range(url, start, end, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 -- network hiccups are retried
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return b""


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with read-ahead."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self._cache_start, self._cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        cs, c = self._cache_start, self._cache
        if not (cs <= self.pos and end <= cs + len(c)):
            fetch_end = min(max(end, self.pos + self.block), self.size) - 1
            self._cache_start, self._cache = self.pos, http_range(self.url, self.pos, fetch_end)
            cs, c = self._cache_start, self._cache
        out = c[self.pos - cs:end - cs]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch_member(url, info):
    """Download and inflate one ZIP member using only its byte range."""
    guess = 30 + len(info.filename.encode()) + 256
    buf = http_range(url, info.header_offset, info.header_offset + guess + info.compress_size)
    if buf[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    fn_len, ex_len = struct.unpack("<HH", buf[26:30])
    start = 30 + fn_len + ex_len
    need = start + info.compress_size
    if len(buf) < need:
        buf += http_range(url, info.header_offset + len(buf), info.header_offset + need - 1)
    raw = buf[start:need]
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompressobj(-15).decompress(raw)
    raise ValueError(f"unsupported compression {info.compress_type}")


def select(stats, per_split, neg_per_split, min_teeth_px, seed):
    """Choose mask names: images with clearly visible teeth + closed-mouth negatives."""
    rng = random.Random(seed)
    chosen = []
    for split in ("train", "val", "test"):
        rows = [r for r in stats if r[0].startswith(split + "/")]
        pos = [r for r in rows if r[3] >= min_teeth_px]
        neg = [r for r in rows if r[3] == 0 and r[4] > 0]   # lips visible, no teeth
        rng.shuffle(pos)
        rng.shuffle(neg)
        chosen += [(r[0], "teeth") for r in pos[:per_split.get(split, 0)]]
        chosen += [(r[0], "noteeth") for r in neg[:neg_per_split.get(split, 0)]]
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", required=True, help="mask_stats.json from the annotation archive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=1400)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--neg-train", type=int, default=350)
    ap.add_argument("--neg-test", type=int, default=60)
    ap.add_argument("--min-teeth-px", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    stats = json.load(open(a.stats))
    want = select(stats, {"train": a.train, "val": a.val, "test": a.test},
                  {"train": a.neg_train, "test": a.neg_test}, a.min_teeth_px, a.seed)
    print(f"selected {len(want)} images", flush=True)

    zf = zipfile.ZipFile(HttpFile(IMAGES_URL))
    by_stem = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        stem = parts[-1].rsplit(".", 1)[0]
        by_stem[(parts[-2] if len(parts) > 1 else "", stem)] = info
    print(f"central directory: {len(by_stem)} members", flush=True)

    out = Path(a.out)
    manifest = []
    jobs = []
    for mask_name, kind in want:
        split, fname = mask_name.split("/")
        stem = fname.rsplit(".", 1)[0]
        info = by_stem.get((split, stem))
        if info is None:
            continue
        dst = out / split / (stem + "." + info.filename.rsplit(".", 1)[-1])
        manifest.append({"mask": mask_name, "image": str(dst.relative_to(out)), "kind": kind})
        if not dst.exists():
            jobs.append((info, dst))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=0)

    t0, done, nbytes = time.time(), 0, 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_member, IMAGES_URL, info): dst for info, dst in jobs}
        for f in cf.as_completed(futs):
            dst = futs[f]
            try:
                data = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {dst.name}: {e}", flush=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(dst)
            done += 1
            nbytes += len(data)
            if done % 50 == 0 or done == len(jobs):
                dt = time.time() - t0
                print(f"{done}/{len(jobs)}  {nbytes / 1e6:.0f} MB  {nbytes / 1e6 / dt:.2f} MB/s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
