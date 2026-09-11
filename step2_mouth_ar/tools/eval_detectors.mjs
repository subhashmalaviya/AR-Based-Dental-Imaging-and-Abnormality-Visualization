#!/usr/bin/env node
/**
 * eval_detectors.mjs — measure tooth detectors against manual ground truth.
 *
 * Runs the app's own detector code (the same modules the browser runs) on
 * rectified mouth crops cut from real recordings by
 * tools/train/make_video_rois.py, and scores them with src/eval/metrics.js.
 * No numbers are produced for frames without ground truth.
 *
 *   node tools/eval_detectors.mjs --rois <dir> --gt tests/eval/gt_mouthtestvideo.json \
 *        [--detectors classical,classical-v1-aperture,learned] [--dump out.json]
 *
 * Ground truth format (ROI 192x144 pixel coordinates):
 *   { "frames": { "<frame id>": { "teeth": [ {"point":[x,y], "jaw":"upper", "partial":false} ] } } }
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { buildApertureMask } from '../src/core/MouthROI.js';
import { ToothSegmenter } from '../src/core/ToothSegmenter.js';
import { evaluateFrame, aggregate, formatSummary } from '../src/eval/metrics.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]?.startsWith('--') ? true : arr[i + 1]]);
  return acc;
}, []));

const ROI_DIR = args.rois;
const GT_FILE = args.gt;
const SEMANTIC = args.semantic === true || args.semantic === 'true';
// The app never runs a detector on a mouth opened less than this (see
// ToothPipeline: "mouth closed"). Apply the same gate so the numbers describe
// what a user actually gets. --opening-gate 0 disables it.
const OPENING_GATE = args['opening-gate'] != null ? Number(args['opening-gate']) : 0.10;
const DET_NAMES = (args.detectors ?? 'classical').split(',');
const GT_W = 192, GT_H = 144;

if (!ROI_DIR || (!GT_FILE && !SEMANTIC)) {
  console.error('usage: eval_detectors.mjs --rois <dir> (--gt <gt.json> | --semantic) [--detectors a,b] [--dump out.json]');
  process.exit(2);
}

/** Rasterise detections (GT pixel space) into a 0/1 mask: polygon, else box. */
function rasterize(dets, W, H) {
  const m = new Uint8Array(W * H);
  for (const d of dets) {
    const poly = d.polygon && d.polygon.length >= 3 ? d.polygon : [
      { x: d.box.x, y: d.box.y }, { x: d.box.x + d.box.w, y: d.box.y },
      { x: d.box.x + d.box.w, y: d.box.y + d.box.h }, { x: d.box.x, y: d.box.y + d.box.h }];
    for (let y = 0; y < H; y++) {
      const yc = y + 0.5, xs = [];
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const a = poly[i], b = poly[j];
        if ((a.y <= yc && b.y > yc) || (b.y <= yc && a.y > yc)) xs.push(a.x + ((yc - a.y) / (b.y - a.y)) * (b.x - a.x));
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(W - 1, Math.floor(xs[k + 1] - 0.5)); x++) m[y * W + x] = 1;
      }
    }
  }
  return m;
}

/** A MouthROI stand-in: just the geometry, no canvas. */
function makeRoi(rec, W, H) {
  const b = rec.bounds;
  return {
    width: W, height: H, bounds: b,
    roiToLocal: (px, py) => ({ u: b.u0 + (px / W) * (b.u1 - b.u0), v: b.v0 + (py / H) * (b.v1 - b.v0) }),
    localToRoi: (u, v) => ({ x: ((u - b.u0) / (b.u1 - b.u0)) * W, y: ((v - b.v0) / (b.v1 - b.v0)) * H }),
  };
}

function loadRgba(file, W, H) {
  const rgb = zlib.inflateSync(fs.readFileSync(file));
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1];
    data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 255;
  }
  return { data, width: W, height: H };
}

/** mouth-local detection -> GT pixel space (192x144 ROI of the same bounds). */
function toGtSpace(det, rec) {
  const b = rec.bounds;
  const X = (u) => ((u - b.u0) / (b.u1 - b.u0)) * GT_W;
  const Y = (v) => ((v - b.v0) / (b.v1 - b.v0)) * GT_H;
  return {
    box: { x: X(det.box.u), y: Y(det.box.v), w: X(det.box.u + det.box.w) - X(det.box.u), h: Y(det.box.v + det.box.h) - Y(det.box.v) },
    center: { x: X(det.center.u), y: Y(det.center.v) },
    polygon: det.contour?.length >= 3 ? det.contour.map((p) => ({ x: X(p.u), y: Y(p.v) })) : null,
    jaw: det.arch,
    conf: det.confidence,
  };
}

// ------------------------------------------------------------ detectors
async function makeDetectors() {
  const out = [];
  for (const name of DET_NAMES) {
    if (name === 'classical') {
      out.push({ name, W: 192, H: 144, det: new ToothSegmenter(), radius: undefined });
    } else if (name === 'classical-v1-aperture') {
      // The shipped Step-3 behaviour before the aperture-erosion fix (13x13 kernel).
      out.push({ name, W: 192, H: 144, det: new ToothSegmenter(), radius: Math.max(2, Math.round(144 * 0.045)) });
    } else if (name === 'learned') {
      const { LearnedToothDetector } = await import('../src/core/LearnedToothDetector.js');
      const model = args.model ?? path.resolve('public/models/tooth_seg.onnx');
      const det = new LearnedToothDetector({ modelUrl: model });
      await det.init();
      out.push({ name, W: det.inputWidth, H: det.inputHeight, det, radius: undefined, async: true });
    } else {
      throw new Error(`unknown detector ${name}`);
    }
  }
  return out;
}

// ------------------------------------------------------------ main
const frames = JSON.parse(fs.readFileSync(path.join(ROI_DIR, 'frames.json'), 'utf8'));
const gt = GT_FILE ? JSON.parse(fs.readFileSync(GT_FILE, 'utf8')) : { frames: {} };
const annotated = SEMANTIC
  ? frames.filter((r) => fs.existsSync(path.join(ROI_DIR, `${r.id}_mask.u8.z`)))
  : frames.filter((r) => gt.frames?.[r.id]);
if (!annotated.length) {
  console.log('No annotated frames found — nothing to evaluate.');
  process.exit(0);
}

const detectors = await makeDetectors();
const dump = {};
const partialOnly = args['clear-only'] === true || args['clear-only'] === 'true';

for (const D of detectors) {
  const perFrame = [];
  let ms = 0;
  for (const rec of annotated) {
    const file = path.join(ROI_DIR, `${rec.id}_${D.W}x${D.H}.rgb.z`);
    const image = loadRgba(file, D.W, D.H);
    const roi = makeRoi(rec, D.W, D.H);
    const ring = rec.localRing.map(([u, v]) => ({ u, v }));
    const aperture = rec.noAperture
      ? new Uint8Array(D.W * D.H).fill(255)       // DentalAI crops: no lips to bound them
      : buildApertureMask(ring, rec.bounds, D.W, D.H, D.radius);
    const t0 = performance.now();
    const gated = rec.opening != null && rec.opening < OPENING_GATE;
    const raw = gated ? []
      : D.async ? await D.det.detectAsync(image, aperture, roi) : D.det.detect(image, aperture, roi);
    ms += performance.now() - t0;
    const dets = (raw ?? []).map((d) => toGtSpace(d, rec));
    (dump[D.name] ??= {})[rec.id] = dets;
    if (SEMANTIC) {
      const gm = zlib.inflateSync(fs.readFileSync(path.join(ROI_DIR, `${rec.id}_mask.u8.z`)));
      const dm = rasterize(dets, GT_W, GT_H);
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < GT_W * GT_H; i++) {
        if (dm[i] && gm[i]) tp++; else if (dm[i]) fp++; else if (gm[i]) fn++;
      }
      perFrame.push({ id: rec.id, kind: rec.kind, tp, fp, fn, n: dets.length, gtPx: tp + fn });
      continue;
    }
    const g = gt.frames[rec.id];
    let gts = g.teeth.map((t) => (t.point
      ? { point: { x: t.point[0], y: t.point[1] }, jaw: t.jaw, partial: !!t.partial }
      : { box: { x: t.box[0], y: t.box[1], w: t.box[2], h: t.box[3] },
          polygon: t.polygon?.map(([x, y]) => ({ x, y })), jaw: t.jaw }));
    const ignore = (g.ignore ?? []).map(([x, y, w, h]) => ({ x, y, w, h }));
    if (partialOnly) {
      // "clear teeth only": partial teeth become ignore regions, not misses
      for (const t of gts.filter((q) => q.partial)) {
        ignore.push({ x: t.point.x - 8, y: t.point.y - 10, w: 16, h: 20 });
      }
      gts = gts.filter((q) => !q.partial);
    }
    const r = evaluateFrame(dets, gts, { ignore });
    perFrame.push({ id: rec.id, ...r });
  }
  if (SEMANTIC) {
    const sum = (k, f = perFrame) => f.reduce((a, x) => a + x[k], 0);
    const tp = sum('tp'), fp = sum('fp'), fn = sum('fn');
    const withTeeth = perFrame.filter((f) => f.gtPx > 0);
    const noTeeth = perFrame.filter((f) => f.gtPx === 0);
    const perImgIoU = withTeeth.map((f) => f.tp / Math.max(1, f.tp + f.fp + f.fn));
    console.log(`${D.name}  (${annotated.length} images, ${(ms / annotated.length).toFixed(2)} ms/image in Node)`);
    console.log(`  teeth-pixel IoU ${(tp / Math.max(1, tp + fp + fn) * 100).toFixed(1)}%   `
      + `precision ${(tp / Math.max(1, tp + fp) * 100).toFixed(1)}%   recall ${(tp / Math.max(1, tp + fn) * 100).toFixed(1)}%`);
    console.log(`  mean per-image IoU (images with teeth, n=${withTeeth.length}) `
      + `${(perImgIoU.reduce((a, v) => a + v, 0) / Math.max(1, perImgIoU.length) * 100).toFixed(1)}%`);
    console.log(`  no-teeth images: ${noTeeth.length}, with any detection: `
      + `${noTeeth.filter((f) => f.n > 0).length} (false-alarm rate `
      + `${(noTeeth.filter((f) => f.n > 0).length / Math.max(1, noTeeth.length) * 100).toFixed(1)}%)\n`);
    continue;
  }
  const s = aggregate(perFrame);
  console.log(formatSummary(`${D.name}  (${annotated.length} frames, ${(ms / annotated.length).toFixed(2)} ms/frame in Node)`, s));
  if (args.verbose) {
    for (const f of perFrame) {
      console.log(`    ${f.id.padEnd(24)} GT ${String(f.countGT).padStart(2)}  det ${String(f.countDet).padStart(2)}`
        + `  TP ${f.tp}  FP ${f.fp}  FN ${f.fn}  dup ${f.duplicates}  merge ${f.merges}`);
    }
  }
  console.log('');
}

if (args.dump) fs.writeFileSync(args.dump, JSON.stringify(dump));
