#!/usr/bin/env node
/**
 * tune_decoder.mjs — choose the learned detector's decode thresholds on a
 * VALIDATION set (never on the test sets that are reported).
 *
 *   node tools/tune_decoder.mjs --rois <valid fixtures> --gt <valid gt.json> --model tooth_seg.onnx
 *
 * The network runs once per crop; only the (cheap) decoder is re-run per
 * parameter setting. Prints the grid sorted by F1 and the best setting.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { decodeToothMaps } from '../src/core/toothDecode.js';
import { evaluateFrame, aggregate } from '../src/eval/metrics.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
  return acc;
}, []));

const frames = JSON.parse(fs.readFileSync(path.join(args.rois, 'frames.json'), 'utf8'));
const gt = JSON.parse(fs.readFileSync(args.gt, 'utf8'));
const det = new LearnedToothDetector({ modelUrl: args.model });
await det.init();
const W = det.inputWidth, H = det.inputHeight;
const GW = 192, GH = 144;

// Run the network once per crop and keep its maps.
const cache = [];
for (const rec of frames) {
  if (!gt.frames[rec.id]) continue;
  const rgb = zlib.inflateSync(fs.readFileSync(path.join(args.rois, `${rec.id}_${W}x${H}.rgb.z`)));
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[4 * i] = rgb[3 * i]; data[4 * i + 1] = rgb[3 * i + 1]; data[4 * i + 2] = rgb[3 * i + 2]; data[4 * i + 3] = 255;
  }
  const m = await det.predictMaps({ data, width: W, height: H });
  cache.push({
    rec,
    maps: { teeth: Float32Array.from(m.teeth), center: Float32Array.from(m.center), boundary: Float32Array.from(m.boundary) },
    gts: gt.frames[rec.id].teeth.map((t) => ({
      box: { x: t.box[0], y: t.box[1], w: t.box[2], h: t.box[3] },
      polygon: t.polygon?.map(([x, y]) => ({ x, y })),
    })),
  });
}

const sx = GW / W, sy = GH / H;
const grid = [];
for (const ctrThr of [0.15, 0.2, 0.25, 0.3, 0.4]) {
  for (const boundaryWeight of [4, 10, 20]) {
    for (const semThr of [0.4, 0.5, 0.6]) {
      const res = cache.map(({ maps, gts }) => {
        const { instances } = decodeToothMaps(maps, W, H, null, { ctrThr, boundaryWeight, semThr });
        const dets = instances.map((t) => ({
          box: { x: t.bbox.x0 * sx, y: t.bbox.y0 * sy, w: (t.bbox.x1 - t.bbox.x0) * sx, h: (t.bbox.y1 - t.bbox.y0) * sy },
          center: { x: t.cx * sx, y: t.cy * sy },
          polygon: t.contour.map(([x, y]) => ({ x: x * sx, y: y * sy })),
        }));
        return evaluateFrame(dets, gts, { iouThreshold: 0.5 });
      });
      const s = aggregate(res);
      grid.push({ ctrThr, boundaryWeight, semThr, f1: s.f1, precision: s.precision, recall: s.recall, meanIoU: s.meanIoU });
    }
  }
}
grid.sort((a, b) => b.f1 - a.f1);
console.log(`validation crops: ${cache.length}`);
for (const g of grid.slice(0, 8)) {
  console.log(`  ctr ${g.ctrThr}  bnd ${String(g.boundaryWeight).padStart(2)}  sem ${g.semThr}  `
    + `F1 ${(g.f1 * 100).toFixed(1)}  P ${(g.precision * 100).toFixed(1)}  R ${(g.recall * 100).toFixed(1)}  mIoU ${g.meanIoU?.toFixed(3)}`);
}
console.log('BEST', JSON.stringify({ ctrThr: grid[0].ctrThr, boundaryWeight: grid[0].boundaryWeight, semThr: grid[0].semThr }));
