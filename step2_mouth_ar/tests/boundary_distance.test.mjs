import test from 'node:test';
import assert from 'node:assert/strict';
import { boundaryDistance, labelBoundary, matchedBoundaryDistances } from '../src/eval/boundaryDistance.js';
import { compareInstances } from '../src/eval/toothInstances.js';
import { enhancePixels } from '../src/ui/photoEnhancement.js';

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test('known 3-4-5 offset gives exact pixels, calibrated mm and normalized error', () => {
  const a = new Int32Array(100), b = a.slice(); a[11] = 1; b[54] = 2;
  const d = boundaryDistance(a, b, 10, 10, 2);
  close(d.assdPx, 5); close(d.assdMm, 2.5); close(d.chamferPx, 5); close(d.chamferMm, 2.5);
  close(d.normalizedAssd, 5 / Math.hypot(10, 10));
  assert.deepEqual(d, boundaryDistance(b, a, 10, 10, 2));
});

test('ASSD and Chamfer document distinct weights for unequal boundary sample counts', () => {
  const d = boundaryDistance([1, 0, 0, 0, 0], [1, 0, 0, 0, 2], 5, 1);
  close(d.assdPx, 4 / 3); close(d.chamferPx, 1);
  close(d.truthToPredictionPx, 0); close(d.predictionToTruthPx, 2);
});

test('missing boundaries remain unavailable and invalid scales do not produce mm', () => {
  for (const [a, b, status] of [[[0], [0], 'both empty'], [[1], [0], 'missing prediction'], [[0], [1], 'missing ground truth']]) {
    const d = boundaryDistance(a, b, 1, 1);
    assert.equal(d.assdPx, null); assert.equal(d.chamferPx, null); assert.equal(d.status, status);
  }
  for (const scale of [null, 0, -1, Infinity, NaN]) {
    const d = boundaryDistance([1, 0], [0, 1], 2, 1, scale);
    assert.equal(d.assdMm, null); assert.equal(d.chamferMm, null);
  }
  assert.throws(() => boundaryDistance([1], [1, 0], 1, 1), /dimensions/);
});

test('tooth separators contribute even when binary foreground is identical', () => {
  const a = new Int32Array(49).fill(1), b = a.slice();
  for (let y = 0; y < 7; y++) for (let x = 4; x < 7; x++) a[y * 7 + x] = 2;
  assert.ok(boundaryDistance(a, b, 7, 7).assdPx > 0);
  assert.equal(boundaryDistance(a, a, 7, 7).assdPx, 0);
});

test('distance transform agrees with brute-force nearest-boundary distances', () => {
  let seed = 345;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let run = 0; run < 35; run++) {
    const w = 5 + run % 7, h = 4 + run % 5;
    const a = Int32Array.from({ length: w * h }, () => random() > 0.55 ? 1 : 0);
    const b = Int32Array.from({ length: w * h }, () => random() > 0.7 ? 2 : 0);
    a[0] = b[b.length - 1] = 1;
    const points = labels => [...labelBoundary(labels, w, h)].flatMap((v, i) => v ? [[i % w, Math.floor(i / w)]] : []);
    const pa = points(a), pb = points(b);
    const sum = (p, q) => p.reduce((s, [x, y]) => s + Math.min(...q.map(([u, v]) => Math.hypot(x - u, y - v))), 0);
    close(boundaryDistance(a, b, w, h).assdPx, (sum(pa, pb) + sum(pb, pa)) / (pa.length + pb.length));
  }
});

test('per-tooth cropped distances agree with full-sized individual masks; missed teeth stay unavailable', () => {
  const a = new Int32Array(100), b = a.slice();
  for (let y = 2; y < 5; y++) for (let x = 2; x < 5; x++) { a[y * 10 + x] = 1; b[y * 10 + x + 1] = 7; }
  a[99] = 2;
  const { rows } = compareInstances(a, b, 10);
  matchedBoundaryDistances(a, b, 10, 10, rows, 4);
  assert.deepEqual(rows[0].boundary, boundaryDistance(a.map(v => v === 1 ? 1 : 0), b, 10, 10, 4));
  assert.equal(rows[1].boundary, null);
});

test('grayscale and contrast preserve source pixels and alpha', () => {
  const source = Uint8ClampedArray.from([255, 0, 0, 255, 50, 50, 50, 128, 200, 200, 200, 255]);
  const saved = source.slice(), gray = enhancePixels(source, true, 1);
  assert.equal(gray[0], gray[1]); assert.equal(gray[1], gray[2]);
  const contrast = enhancePixels(source, true, 2);
  assert.ok(contrast[4] < gray[4]); assert.ok(contrast[8] > gray[8]);
  assert.equal(contrast[7], 128); assert.deepEqual(source, saved);
  assert.deepEqual(enhancePixels(source, false, 1), source);
});
