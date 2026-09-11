/**
 * Step 3 v2 verification: optimal assignment, evaluation metrics, tracker v2
 * and the learned-model instance decoder. Everything here is deterministic and
 * runs without a browser: `npm test`.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { hungarian, assign, FORBIDDEN } from '../src/core/math/hungarian.js';
import { evaluateFrame, aggregate, boxIoU, polygonIoU } from '../src/eval/metrics.js';
import { ToothTracker } from '../src/core/ToothTracker.js';
import { decodeToothMaps, assignJaws, findPeaks } from '../src/core/toothDecode.js';
import { recordingBaseName } from '../src/core/SessionRecorder.js';
import { MetadataLogger, METADATA_SCHEMA } from '../src/core/MetadataLogger.js';

// ------------------------------------------------------------ hungarian
test('hungarian finds the optimal assignment where greedy would not', () => {
  // Greedy picks (0,0)=1 first and is then forced into (1,1)=10: total 11.
  // Optimal is (0,1)+(1,0) = 2 + 2 = 4.
  const cost = [[1, 2], [2, 10]];
  assert.deepEqual(hungarian(cost), [1, 0]);
});

test('hungarian handles rectangular matrices both ways', () => {
  assert.deepEqual(hungarian([[5, 1, 9]]), [1]);
  const r = hungarian([[3], [1], [2]]);
  assert.deepEqual(r, [-1, 0, -1]);
});

test('assign() never returns forbidden pairs', () => {
  const pairs = assign([[FORBIDDEN, FORBIDDEN], [FORBIDDEN, 0.2]]);
  assert.deepEqual(pairs, [[1, 1]]);
});

// ------------------------------------------------------------ metrics
const box = (x, y, w = 10, h = 12) => ({ box: { x, y, w, h }, center: { x: x + w / 2, y: y + h / 2 } });
const pt = (x, y, jaw = 'upper') => ({ point: { x, y }, jaw });

test('perfect point-GT frame scores 100% with no errors', () => {
  const dets = [box(0, 0), box(12, 0), box(24, 0)].map((d) => ({ ...d, jaw: 'upper' }));
  const gts = [pt(5, 6), pt(17, 6), pt(29, 6)];
  const r = evaluateFrame(dets, gts);
  assert.deepEqual([r.tp, r.fp, r.fn, r.duplicates, r.merges], [3, 0, 0, 0, 0]);
  assert.equal(r.jawCorrect, 3);
});

test('a merged box (two teeth in one detection) is counted as a merge + a miss', () => {
  const dets = [{ box: { x: 0, y: 0, w: 22, h: 12 } }];
  const gts = [pt(5, 6), pt(17, 6)];
  const r = evaluateFrame(dets, gts);
  assert.equal(r.tp, 1);
  assert.equal(r.fn, 1);
  assert.equal(r.merges, 1);
});

test('a split tooth (two detections on one tooth) is counted as a duplicate', () => {
  // one 12-px-wide tooth, crown centre at x=6, cut into two 6-px halves
  for (const x of [6, 5.5, 6.8]) {
    const r = evaluateFrame([box(0, 0, 6, 12), box(6, 0, 6, 12)], [pt(x, 6)]);
    assert.equal(r.tp, 1, `tp @x=${x}`);
    assert.equal(r.fp, 1, `fp @x=${x}`);
    assert.equal(r.duplicates, 1, `duplicate @x=${x}`);
  }
});

test('an adjacent, unannotated tooth is a false positive, not a duplicate', () => {
  // three side-by-side teeth detected, only the middle one annotated
  const dets = [box(0, 0, 10, 16), box(10, 0, 10, 16), box(20, 0, 10, 16)];
  const r = evaluateFrame(dets, [pt(15, 8)]);
  assert.equal(r.tp, 1);
  assert.equal(r.fp, 2);
  assert.equal(r.duplicates, 0);
});

test('box-GT matching uses IoU and polygon IoU when both have polygons', () => {
  assert.ok(Math.abs(boxIoU({ x: 0, y: 0, w: 10, h: 10 }, { x: 5, y: 0, w: 10, h: 10 }) - 1 / 3) < 1e-9);
  const sq = (x) => [{ x, y: 0 }, { x: x + 10, y: 0 }, { x: x + 10, y: 10 }, { x, y: 10 }];
  assert.ok(Math.abs(polygonIoU(sq(0), sq(5)) - 1 / 3) < 0.03);
  const r = evaluateFrame([{ box: { x: 1, y: 0, w: 10, h: 10 } }], [{ box: { x: 0, y: 0, w: 10, h: 10 } }]);
  assert.equal(r.tp, 1);
  assert.ok(r.ious[0] > 0.8);
});

test('aggregate computes precision / recall / F1 from counts only', () => {
  const s = aggregate([
    { tp: 8, fp: 2, fn: 2, duplicates: 1, merges: 0, ious: [], jawCorrect: 7, jawTotal: 8, countGT: 10, countDet: 10 },
    { tp: 5, fp: 0, fn: 5, duplicates: 0, merges: 2, ious: [], jawCorrect: 5, jawTotal: 5, countGT: 10, countDet: 5 },
  ]);
  assert.equal(s.tp, 13);
  assert.ok(Math.abs(s.precision - 13 / 15) < 1e-9);
  assert.ok(Math.abs(s.recall - 13 / 20) < 1e-9);
  assert.equal(s.exactCountRate, 0.5);
  assert.equal(s.countMAE, 2.5);
  assert.equal(aggregate([]), null, 'no frames -> no metrics, not zeros');
});

// ------------------------------------------------------------ tracker v2
const det = (u, v, conf = 0.8, arch = 'upper', w = 0.09, h = 0.11) => ({
  center: { u, v }, box: { u: u - w / 2, v: v - h / 2, w, h }, contour: [], confidence: conf, arch,
});
const row = (n, { v = -0.1, conf = 0.8, arch = 'upper', du = 0 } = {}) =>
  Array.from({ length: n }, (_, i) => det(-0.3 + i * 0.1 + du, v, conf, arch));

test('NMS removes a duplicate report of the same tooth', () => {
  const tr = new ToothTracker();
  const kept = tr.nms([det(0, 0, 0.9), det(0.005, 0, 0.6), det(0.2, 0, 0.8)]);
  assert.equal(kept.length, 2);
  assert.equal(kept[0].confidence, 0.9);
});

test('weak detections keep a tooth alive but never create new IDs', () => {
  const tr = new ToothTracker({ highConf: 0.5, lowConf: 0.15 });
  for (let f = 0; f < 4; f++) tr.update(row(3));
  const ids = tr.visibleTracks().map((t) => t.id);
  // a dim frame: all detections weak, plus a weak spurious one elsewhere
  tr.update([...row(3, { conf: 0.3 }), det(0.4, 0.2, 0.3)]);
  assert.deepEqual(tr.visibleTracks().map((t) => t.id), ids, 'same teeth, same IDs');
  assert.equal(tr.tracks.length, 3, 'the weak spurious detection spawned nothing');
});

test('jaw-relative matching keeps lower-tooth IDs when the mouth opens fast', () => {
  const mk = (jawRef) => {
    const tr = new ToothTracker({ maxCentroidDist: 0.5 });
    for (let f = 0; f < 4; f++) tr.update(row(4, { v: 0.05, arch: 'lower' }), { jawRef });
    return tr;
  };
  const before = { upper: -0.05, lower: 0.02 };
  const after = { upper: -0.05, lower: 0.32 };     // lower lip dropped 0.30 mouth widths
  const opened = row(4, { v: 0.35, arch: 'lower' });

  const withRef = mk(before);
  const idsA = withRef.visibleTracks().map((t) => t.id);
  withRef.update(opened, { jawRef: after });
  assert.deepEqual(withRef.visibleTracks().map((t) => t.id), idsA, 'IDs kept with jaw reference');

  const noRef = mk(null);
  const idsB = noRef.visibleTracks().map((t) => t.id);
  noRef.update(opened);
  noRef.update(opened);
  const kept = noRef.visibleTracks().filter((t) => idsB.includes(t.id)).length;
  assert.ok(kept < 4, 'without it, the same jump breaks identity (shows the reference matters)');
});

test('two tracks that converge on one tooth are merged, keeping the older ID', () => {
  const tr = new ToothTracker({ nmsIoU: 0.99, nmsContain: 1.01 });
  for (let f = 0; f < 5; f++) tr.update([det(0, 0)]);
  const oldId = tr.visibleTracks()[0].id;
  for (let f = 0; f < 3; f++) tr.update([det(0, 0), det(0.3, 0)]);
  for (let f = 0; f < 3; f++) tr.update([det(0, 0), det(0.01, 0)]);   // second drifts onto the first
  const vis = tr.visibleTracks();
  assert.equal(vis.filter((t) => Math.abs(t.center.u) < 0.05).length, 1);
  assert.ok(vis.some((t) => t.id === oldId));
});

test('stability is measured from ID continuity, and drops when IDs churn', () => {
  const steady = new ToothTracker();
  for (let f = 0; f < 20; f++) steady.update(row(5));
  assert.ok(steady.stats().stability > 0.95);

  const churn = new ToothTracker({ minHitsToShow: 1, maxMissing: 0 });
  for (let f = 0; f < 20; f++) churn.update(row(5, { du: (f % 2) * 0.5 }));
  assert.ok(churn.stats().stability < 0.5, `got ${churn.stats().stability}`);
});

// ------------------------------------------------------------ decoder
/** Synthetic maps: `k` touching teeth in a row, with boundary ridges. */
function synth(k, { W = 64, H = 32, withCenters = true } = {}) {
  const n = W * H;
  const teeth = new Float32Array(n), center = new Float32Array(n), boundary = new Float32Array(n);
  const tw = 12, x0 = 4, y0 = 8, y1 = 24;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x0 + k * tw; x++) teeth[y * W + x] = 0.95;
  for (let t = 1; t < k; t++) {
    const bx = x0 + t * tw;
    for (let y = y0; y < y1; y++) { boundary[y * W + bx] = 0.9; boundary[y * W + bx - 1] = 0.6; }
  }
  if (withCenters) {
    for (let t = 0; t < k; t++) {
      const cx = x0 + t * tw + tw / 2 - 0.5, cy = (y0 + y1) / 2 - 0.5;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5 ** 2));
        center[y * W + x] = Math.max(center[y * W + x], v);
      }
    }
  }
  return { maps: { teeth, center, boundary }, W, H };
}

test('decoder splits touching teeth along the predicted boundary', () => {
  const { maps, W, H } = synth(4);
  const { instances } = decodeToothMaps(maps, W, H, null, { minArea: 10 });
  assert.equal(instances.length, 4);
  const xs = instances.map((t) => t.cx).sort((a, b) => a - b);
  xs.forEach((x, i) => assert.ok(Math.abs(x - (4 + i * 12 + 6)) < 1.5, `tooth ${i} centre ${x}`));
  instances.forEach((t) => assert.ok(t.contour.length >= 4));
});

test('enamel with no centre peak still becomes a (lower-confidence) tooth', () => {
  const { maps, W, H } = synth(1, { withCenters: false });
  const { instances } = decodeToothMaps(maps, W, H, null, { minArea: 10 });
  assert.equal(instances.length, 1);
  assert.equal(instances[0].seeded, false);
  assert.ok(instances[0].confidence < 0.8);
});

test('decoder reports nothing when the model sees no enamel', () => {
  const W = 40, H = 30, n = W * H;
  const z = () => new Float32Array(n);
  const { instances } = decodeToothMaps({ teeth: z(), center: z(), boundary: z() }, W, H, null);
  assert.equal(instances.length, 0);
});

test('findPeaks yields one peak per plateau', () => {
  const W = 5, H = 1, m = new Float32Array([0, 1, 1, 0, 0]);
  const p = findPeaks(m, W, H, new Uint8Array(5).fill(1), 0.5, 1);
  assert.equal(p.length, 1);
});

test('jaws: two bands split at the gap; a single band follows the midline', () => {
  const W = 40, H = 40;
  const ap = new Uint8Array(W * H).fill(255);
  const two = [{ cx: 10, cy: 10 }, { cx: 20, cy: 11 }, { cx: 12, cy: 30 }, { cx: 22, cy: 29 }];
  assert.deepEqual(assignJaws(two, W, H, ap), ['upper', 'upper', 'lower', 'lower']);
  const onlyTop = [{ cx: 10, cy: 8 }, { cx: 20, cy: 9 }];
  assert.deepEqual(assignJaws(onlyTop, W, H, ap), ['upper', 'upper']);
});

// ------------------------------------------------------------ recording
test('recording file names follow dental_tracking_YYYYMMDD_HHMMSS', () => {
  assert.equal(recordingBaseName(new Date(2026, 8, 11, 10, 15, 30)), 'dental_tracking_20260911_101530');
});

test('metadata logger records only what the pipeline reported', () => {
  const log = new MetadataLogger();
  log.begin({ app: 'test' });
  const anchor = {
    isValid: () => true,
    localToScreen: ({ x, y }) => ({ x: 100 + 200 * x, y: 50 + 200 * y }),
  };
  const track = {
    id: 7, arch: 'upper', status: 'stable', hits: 9,
    smoothed: {
      center: { u: 0, v: -0.1 }, box: { u: -0.05, v: -0.15, w: 0.1, h: 0.1 },
      contour: [{ u: -0.05, v: -0.15 }, { u: 0.05, v: -0.15 }, { u: 0.05, v: -0.05 }], confidence: 0.9,
    },
  };
  log.log({ t_ms: 33.3, fps: 30, face: true, mouth: true, opening: 0.4,
            stats: { count: 1, avgConfidence: 0.9, status: 'stable' }, tracks: [track], anchor });
  log.log({ t_ms: 66.6, fps: 30, face: false, mouth: false, tracks: [] });
  const doc = log.end();
  assert.equal(doc.header.schema, METADATA_SCHEMA);
  assert.equal(doc.frames.length, 2);
  assert.deepEqual(doc.frames[0].teeth[0].center, [100, 30]);
  assert.deepEqual(doc.frames[0].teeth[0].bbox, [90, 20, 20, 20]);
  assert.equal(doc.frames[1].n, 0);
  assert.equal(doc.summary.unique_tooth_ids, 1);
});

// ------------------------------------------------------------ pipeline (async)
test('a stale async inference is discarded after reset / detector switch', async () => {
  const { ToothPipeline } = await import('../src/core/ToothPipeline.js');
  const pipe = new ToothPipeline({ detector: 'classical' });
  let release;
  pipe.detector = {
    isAsync: true, ready: true, name: 'fake', isLearnedModel: true,
    detectAsync: () => new Promise((r) => { release = r; }),
  };
  pipe.roi = {
    computeBounds: () => true, extract: () => ({ width: 1, height: 1 }), apertureMask: () => null,
    snapshot: () => ({ jawRef: { upper: 0, lower: 0 } }), jawRef: { upper: 0, lower: 0 }, bounds: {},
  };
  const anchor = { getPose: () => ({ origin: { x: 0, y: 0, z: 0 }, basis: {}, scale: 100 }) };
  const mouth = { opening: { ratio: 0.4 } };
  pipe.update(null, [{}], mouth, anchor, 0, 640, 480);      // launches inference
  pipe.reset();                                              // user hits Reset Tracking
  release([{ center: { u: 0, v: 0 }, box: { u: -0.05, v: -0.05, w: 0.1, h: 0.1 }, confidence: 0.9, arch: 'upper' }]);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(pipe._pending, null, 'result from before the reset must be dropped');
});

test('a merged band of teeth is re-split from weak centre peaks', () => {
  // 5 teeth wide band, boundaries present but centre peaks weak (0.15 < ctrThr)
  // except the first; plus two normal isolated teeth so a "typical" width exists.
  const W = 120, H = 40, n = W * H;
  const teeth = new Float32Array(n), center = new Float32Array(n), boundary = new Float32Array(n);
  const blob = (xa, xb, peak) => {
    for (let y = 10; y < 30; y++) for (let x = xa; x < xb; x++) teeth[y * W + x] = 0.9;
    const cx = (xa + xb) / 2 - 0.5;
    for (let y = 0; y < H; y++) for (let x = xa; x < xb; x++) {
      center[y * W + x] = Math.max(center[y * W + x], peak * Math.exp(-((x - cx) ** 2 + (y - 19.5) ** 2) / 8));
    }
  };
  blob(2, 12, 0.9); blob(16, 26, 0.9);                     // two separate normal teeth
  for (let k = 0; k < 5; k++) blob(30 + k * 10, 40 + k * 10, k === 0 ? 0.9 : 0.15);
  for (let k = 1; k < 5; k++) for (let y = 10; y < 30; y++) boundary[y * W + 30 + k * 10] = 0.5;
  const merged = decodeToothMaps({ teeth, center, boundary }, W, H, null, { minArea: 10 });
  const split = decodeToothMaps({ teeth, center, boundary }, W, H, null, { minArea: 10, splitWide: true });
  assert.equal(merged.instances.length, 3, 'without re-splitting the band is one tooth');
  assert.equal(split.instances.length, 7, 'with re-splitting every tooth is separate');
});

test('re-splitting never invents teeth without centre evidence', () => {
  const W = 80, H = 30, n = W * H;
  const teeth = new Float32Array(n), center = new Float32Array(n), boundary = new Float32Array(n);
  for (let y = 8; y < 22; y++) for (let x = 5; x < 75; x++) teeth[y * W + x] = 0.9;
  center[15 * W + 10] = 0.9;
  const { instances } = decodeToothMaps({ teeth, center, boundary }, W, H, null, { minArea: 10, splitWide: true });
  assert.equal(instances.length, 1);
});
