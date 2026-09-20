#!/usr/bin/env node
/**
 * tune_decoder_point_gt.mjs — decode-threshold grid search against POINT
 * ground truth, with or without TTA.
 *
 * tune_decoder.mjs assumes box/polygon ground truth. This project's own real
 * ground truth (tests/eval/gt_mouthtestvideo.json) is point-per-tooth, so
 * this is the tool that reproduces §32d's README numbers, and the one to
 * re-run after re-annotating a clip in eval.html or regenerating ROIs with
 * tools/train/make_video_rois.py.
 *
 *   node tools/tune_decoder_point_gt.mjs --rois tests/eval/rois \
 *        --gt tests/eval/gt_mouthtestvideo.json [--model <onnx>] [--tta true]
 *
 * (`--tta` as the very last argument parses as undefined, not true — same
 * quirk as eval_detectors.mjs's `--clear-only`; always write `--tta true`.)
 *
 * The network runs once (twice with --tta) per crop; only the decoder is
 * re-run per grid point.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { decodeToothMaps, assignJaws } from '../src/core/toothDecode.js';
import { buildApertureMask } from '../src/core/MouthROI.js';
import { evaluateFrame, aggregate } from '../src/eval/metrics.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]?.startsWith('--') ? true : arr[i + 1]]);
  return acc;
}, []));
if (!args.rois || !args.gt) {
  console.error('usage: tune_decoder_point_gt.mjs --rois <dir> --gt <gt.json> [--model <onnx>] [--tta true]');
  process.exit(2);
}

const GT_W = 192, GT_H = 144;
const useTta = args.tta === true || args.tta === 'true';

const frames = JSON.parse(fs.readFileSync(path.join(args.rois, 'frames.json'), 'utf8'));
const gt = JSON.parse(fs.readFileSync(args.gt, 'utf8'));
const det = new LearnedToothDetector({ modelUrl: args.model ?? path.resolve('public/models/tooth_seg.onnx') });
await det.init();
const W = det.inputWidth, H = det.inputHeight;

const annotated = frames.filter((r) => gt.frames[r.id]);
const cache = [];
for (const rec of annotated) {
  const rgb = zlib.inflateSync(fs.readFileSync(path.join(args.rois, `${rec.id}_${W}x${H}.rgb.z`)));
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[4 * i] = rgb[3 * i]; data[4 * i + 1] = rgb[3 * i + 1]; data[4 * i + 2] = rgb[3 * i + 2]; data[4 * i + 3] = 255;
  }
  const image = { data, width: W, height: H };
  const m = useTta ? await det.predictMapsTTA(image) : await det.predictMaps(image);
  const ring = rec.localRing.map(([u, v]) => ({ u, v }));
  const aperture = buildApertureMask(ring, rec.bounds, W, H);
  const sx = W / GT_W, sy = H / GT_H;
  const gts = gt.frames[rec.id].teeth.map((t) => ({
    point: { x: t.point[0] * sx, y: t.point[1] * sy }, jaw: t.jaw, partial: !!t.partial,
  }));
  const ignore = (gt.frames[rec.id].ignore ?? []).map(([x, y, w, h]) => ({ x: x * sx, y: y * sy, w: w * sx, h: h * sy }));
  cache.push({
    maps: { teeth: Float32Array.from(m.teeth), center: Float32Array.from(m.center), boundary: Float32Array.from(m.boundary) },
    aperture, gts, ignore,
  });
}

function scoreParams(params) {
  const res = cache.map(({ maps, aperture, gts, ignore }) => {
    const { instances } = decodeToothMaps(maps, W, H, aperture, params);
    const jaws = assignJaws(instances, W, H, aperture);
    const dets = instances.map((t, k) => ({
      box: { x: t.bbox.x0, y: t.bbox.y0, w: t.bbox.x1 - t.bbox.x0, h: t.bbox.y1 - t.bbox.y0 },
      center: { x: t.cx, y: t.cy },
      polygon: t.contour.map(([x, y]) => ({ x, y })),
      jaw: jaws[k],
    }));
    return evaluateFrame(dets, gts, { ignore });
  });
  return aggregate(res);
}

const grid = [];
for (const ctrThr of [0.12, 0.15, 0.18, 0.2, 0.25, 0.3]) {
  for (const semThr of [0.3, 0.35, 0.4, 0.45, 0.5]) {
    for (const boundaryWeight of [10, 15, 20, 25, 30]) {
      const s = scoreParams({ ctrThr, semThr, boundaryWeight });
      grid.push({ ctrThr, semThr, boundaryWeight, ...s });
    }
  }
}
grid.sort((a, b) => b.f1 - a.f1);
console.log(`crops: ${cache.length}  TTA: ${useTta}`);
for (const g of grid.slice(0, 10)) {
  console.log(`ctr ${g.ctrThr}  sem ${g.semThr}  bnd ${String(g.boundaryWeight).padStart(2)}  `
    + `F1 ${(g.f1 * 100).toFixed(1)}  P ${(g.precision * 100).toFixed(1)}  R ${(g.recall * 100).toFixed(1)}  `
    + `TP ${g.tp} FP ${g.fp} FN ${g.fn} dup ${g.duplicates} merge ${g.merges}`);
}
console.log('BEST', JSON.stringify({ ctrThr: grid[0].ctrThr, semThr: grid[0].semThr, boundaryWeight: grid[0].boundaryWeight }));
