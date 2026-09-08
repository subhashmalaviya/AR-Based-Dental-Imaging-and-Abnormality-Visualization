/**
 * Step 3 verification: tooth tracking, smoothing and segmentation logic.
 *
 * The detector's accuracy on real mouths is measured separately against real
 * footage (tools/prototype_tooth_seg.py) and end-to-end in a browser fed that
 * same video as a fake camera. What is tested here is everything that must be
 * correct regardless of the detector: identity stability, occlusion handling,
 * smoothing behaviour, and that the segmenter responds to real image structure
 * rather than inventing detections.
 *
 * Run: npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { ToothTracker } from '../src/core/ToothTracker.js';
import { TrackingSmoother } from '../src/core/TrackingSmoother.js';
import { ToothSegmenter } from '../src/core/ToothSegmenter.js';

// ------------------------------------------------------------------ helpers
const det = (u, v, w = 0.09, h = 0.11, arch = 'upper', conf = 0.8) => ({
  center: { u, v },
  box: { u: u - w / 2, v: v - h / 2, w, h },
  contour: [
    { u: u - w / 2, v: v - h / 2 }, { u: u + w / 2, v: v - h / 2 },
    { u: u + w / 2, v: v + h / 2 }, { u: u - w / 2, v: v + h / 2 },
  ],
  area: w * h,
  confidence: conf,
  arch,
});

/** A row of teeth, as the segmenter would report for one arch. */
const row = (n, dx = 0, arch = 'upper') =>
  Array.from({ length: n }, (_, i) => det(-0.3 + i * 0.1 + dx, -0.1, 0.09, 0.11, arch));

// ------------------------------------------------------------------- tracker
test('tooth IDs stay stable across frames while teeth remain visible', () => {
  const tr = new ToothTracker();
  tr.update(row(6));
  const first = tr.visibleTracks().length === 0 ? null : null;
  tr.update(row(6));
  const ids = tr.visibleTracks().map((t) => t.id);
  assert.equal(ids.length, 6, 'all six teeth visible after minHits');

  // small head-relative wobble, well within a tooth width
  for (let f = 0; f < 30; f++) {
    tr.update(row(6, 0.004 * Math.sin(f / 3)));
  }
  const after = tr.visibleTracks().map((t) => t.id);
  assert.deepEqual(after, ids, 'IDs must not churn frame to frame');
});

test('IDs survive a brief occlusion (mouth closes then reopens)', () => {
  const tr = new ToothTracker({ maxMissing: 8 });
  for (let f = 0; f < 10; f++) tr.update(row(5));
  const before = tr.visibleTracks().map((t) => t.id);
  assert.equal(before.length, 5);

  for (let f = 0; f < 5; f++) tr.update([]);        // mouth shut
  assert.equal(tr.visibleTracks().length, 0, 'nothing drawn while occluded');

  for (let f = 0; f < 3; f++) tr.update(row(5));    // reopened
  const after = tr.visibleTracks().map((t) => t.id);
  assert.deepEqual(after, before, 'must reuse the same IDs, not renumber');
});

test('tracks are retired after a sustained absence', () => {
  const tr = new ToothTracker({ maxMissing: 4 });
  for (let f = 0; f < 10; f++) tr.update(row(3));
  assert.equal(tr.tracks.length, 3);
  for (let f = 0; f < 6; f++) tr.update([]);
  assert.equal(tr.tracks.length, 0, 'stale tracks must not accumulate');
});

test('an upper tooth is never matched to a lower one', () => {
  const tr = new ToothTracker();
  for (let f = 0; f < 6; f++) tr.update(row(4, 0, 'upper'));
  const upperIds = tr.visibleTracks().map((t) => t.id);
  // lower arch appears at the same u positions but a different v
  for (let f = 0; f < 6; f++) {
    tr.update([...row(4, 0, 'upper'), ...row(4, 0, 'lower').map((d) => ({
      ...d, center: { u: d.center.u, v: 0.15 }, box: { ...d.box, v: 0.09 },
    }))]);
  }
  const vis = tr.visibleTracks();
  assert.equal(vis.length, 8, 'both arches tracked');
  const stillUpper = vis.filter((t) => upperIds.includes(t.id));
  assert.ok(stillUpper.every((t) => t.arch === 'upper'),
    'original IDs must remain on the upper arch');
});

test('neighbouring teeth do not swap IDs when the row shifts', () => {
  const tr = new ToothTracker();
  for (let f = 0; f < 8; f++) tr.update(row(6));
  const byU = tr.visibleTracks()
    .sort((a, b) => a.center.u - b.center.u).map((t) => t.id);
  // shift the whole row by a third of a tooth width, as a head turn would
  for (let f = 0; f < 8; f++) tr.update(row(6, 0.03));
  const afterU = tr.visibleTracks()
    .sort((a, b) => a.center.u - b.center.u).map((t) => t.id);
  assert.deepEqual(afterU, byU, 'left-to-right ID order must be preserved');
});

test('stats report count, average confidence and stability', () => {
  const tr = new ToothTracker();
  for (let f = 0; f < 8; f++) {
    tr.update(row(4).map((d, i) => ({ ...d, confidence: 0.6 + i * 0.1 })));
  }
  const s = tr.stats();
  assert.equal(s.count, 4);
  assert.ok(Math.abs(s.avgConfidence - 0.75) < 1e-6, `avg ${s.avgConfidence}`);
  assert.equal(s.status, 'stable');
});

test('reset clears tracks and restarts numbering', () => {
  const tr = new ToothTracker();
  for (let f = 0; f < 5; f++) tr.update(row(3));
  assert.ok(tr.tracks[0].id >= 1);
  tr.reset();
  assert.equal(tr.tracks.length, 0);
  tr.update(row(2));
  assert.equal(tr.tracks[0].id, 1, 'IDs restart from 1 after reset');
});

test('pick() hit-tests in mouth-local coordinates', () => {
  const tr = new ToothTracker();
  for (let f = 0; f < 6; f++) tr.update(row(5));
  const target = tr.visibleTracks()[2];
  const hit = tr.pick(target.center.u, target.center.v);
  assert.ok(hit, 'centre of a tooth must hit it');
  assert.equal(hit.id, target.id);
  assert.equal(tr.pick(5, 5), null, 'far outside must hit nothing');
});

// ----------------------------------------------------------------- smoothing
test('smoothing reduces per-tooth jitter', () => {
  const mk = (preset) => {
    const tr = new ToothTracker();
    const sm = new TrackingSmoother(preset);
    const xs = [];
    let seed = 3;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff) * 2 - 1; };
    for (let f = 0; f < 200; f++) {
      const noisy = row(4).map((d) => {
        const u = d.center.u + 0.006 * rnd();
        return { ...d, center: { u, v: d.center.v }, box: { ...d.box, u: u - d.box.w / 2 } };
      });
      const tracks = tr.update(noisy);
      sm.apply(tracks, f / 60);
      if (f > 120) xs.push(tracks[0].smoothed.center.u);
    }
    return xs;
  };
  const sd = (a) => {
    const m = a.reduce((p, c) => p + c, 0) / a.length;
    return Math.sqrt(a.reduce((p, c) => p + (c - m) ** 2, 0) / a.length);
  };
  const responsive = sd(mk('responsive'));
  const smooth = sd(mk('smooth'));
  assert.ok(smooth < responsive, `smooth ${smooth.toFixed(5)} < responsive ${responsive.toFixed(5)}`);
});

test('smoothing does not lag behind sustained motion', () => {
  const tr = new ToothTracker();
  const sm = new TrackingSmoother('balanced');
  let tracks;
  for (let f = 0; f < 90; f++) {
    tracks = tr.update(row(4, f * 0.002));
    sm.apply(tracks, f / 60);
  }
  const t0 = tracks.sort((a, b) => a.center.u - b.center.u)[0];
  const lag = Math.abs(t0.smoothed.center.u - t0.center.u);
  assert.ok(lag < 0.02, `steady-state lag ${lag.toFixed(4)} mouth-widths must stay small`);
});

test('smoother releases filter state for retired tracks', () => {
  const tr = new ToothTracker({ maxMissing: 2 });
  const sm = new TrackingSmoother('balanced');
  for (let f = 0; f < 6; f++) sm.apply(tr.update(row(4)), f / 60);
  assert.equal(sm.banks.size, 4);
  for (let f = 6; f < 14; f++) sm.apply(tr.update([]), f / 60);
  assert.equal(sm.banks.size, 0, 'filter banks must not leak over a long session');
});

// ---------------------------------------------------------------- segmenter
/**
 * Build a synthetic ROI shaped like a real open mouth: dark cavity, a bright
 * upper arch, a red tongue in the middle, and (optionally) a lower arch beneath
 * it — which is the arrangement the arch extraction is designed for.
 */
function syntheticRoi(W, H, { nTeeth = 5, tongue = true, lower = true } = {}) {
  const data = new Uint8ClampedArray(W * H * 4);
  const aperture = new Uint8Array(W * H);
  const put = (x, y, r, g, b) => {
    const i = (y * W + x) * 4;
    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
  };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      put(x, y, 35, 18, 20);                                  // dark cavity
      if (x > 8 && x < W - 8 && y > 8 && y < H - 8) aperture[y * W + x] = 255;
    }
  }
  const toothW = Math.floor((W - 40) / nTeeth);
  const drawRow = (yTop, yBot) => {
    for (let t = 0; t < nTeeth; t++) {
      const x0 = 20 + t * toothW + 2;
      const x1 = 20 + (t + 1) * toothW - 2;
      for (let y = yTop; y < yBot; y++) {
        for (let x = x0; x < x1; x++) put(x, y, 232, 230, 225);
      }
    }
  };
  drawRow(14, 14 + Math.floor(H * 0.22));                     // upper arch
  // saturated red tongue in the middle: must NOT be reported as a tooth
  if (tongue) {
    for (let y = Math.floor(H * 0.42); y < Math.floor(H * 0.68); y++) {
      for (let x = 25; x < W - 25; x++) put(x, y, 205, 70, 80);
    }
  }
  if (lower) drawRow(H - 14 - Math.floor(H * 0.20), H - 14); // lower arch
  return { image: { data, width: W, height: H }, aperture };
}

const fakeRoi = (W, H) => ({
  width: W, height: H,
  bounds: { u0: -0.5, u1: 0.5, v0: -0.4, v1: 0.4 },
  roiToLocal(px, py) {
    return { u: -0.5 + (px / W) * 1.0, v: -0.4 + (py / H) * 0.8 };
  },
});

test('segmenter finds both arches and splits them into teeth', () => {
  const W = 192, H = 144;
  const { image, aperture } = syntheticRoi(W, H, { nTeeth: 5 });
  const out = new ToothSegmenter().detect(image, aperture, fakeRoi(W, H));
  assert.ok(out.length >= 6, `expected both arches split, got ${out.length}`);
  assert.ok(out.every((d) => d.confidence > 0 && d.confidence <= 1), 'confidence in range');
  assert.ok(out.every((d) => d.contour.length >= 4), 'each tooth has a contour');
  assert.ok(out.some((d) => d.arch === 'upper'), 'upper arch found');
  assert.ok(out.some((d) => d.arch === 'lower'), 'lower arch found');
  // the top row must be labelled upper, the bottom row lower
  for (const d of out) {
    if (d.arch === 'upper') assert.ok(d.center.v < 0, `upper at v=${d.center.v}`);
    else assert.ok(d.center.v > 0, `lower at v=${d.center.v}`);
  }
});

test('segmenter rejects the tongue', () => {
  const W = 192, H = 144;
  const roi = fakeRoi(W, H);
  const seg = new ToothSegmenter();
  const withT = syntheticRoi(W, H, { nTeeth: 5, tongue: true });
  const without = syntheticRoi(W, H, { nTeeth: 5, tongue: false });
  const a = seg.detect(withT.image, withT.aperture, roi);
  const b = seg.detect(without.image, without.aperture, roi);
  assert.ok(Math.abs(a.length - b.length) <= 1,
    `tongue changed the count ${b.length} -> ${a.length}`);
  // Nothing may be reported in the middle band where the tongue lives.
  const mid = a.filter((d) => d.center.v > -0.06 && d.center.v < 0.10);
  assert.equal(mid.length, 0, 'nothing should be reported in the tongue band');
});

test('a tongue with no lower teeth is not promoted to a lower arch', () => {
  // The hard case: with no lower arch present the tongue *is* the bottom-most
  // bright thing, so only the relative whiteness floor can reject it.
  const W = 192, H = 144;
  const { image, aperture } = syntheticRoi(W, H, { nTeeth: 5, tongue: true, lower: false });
  const out = new ToothSegmenter().detect(image, aperture, fakeRoi(W, H));
  assert.ok(out.every((d) => d.arch === 'upper'),
    `tongue must not become a lower arch: ${JSON.stringify(out.map((d) => d.arch))}`);
});

test('segmenter reports nothing for an empty (dark) mouth', () => {
  const W = 192, H = 144;
  const data = new Uint8ClampedArray(W * H * 4);
  const aperture = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      data[i] = 30; data[i + 1] = 16; data[i + 2] = 18; data[i + 3] = 255;
      if (x > 8 && x < W - 8 && y > 8 && y < H - 8) aperture[y * W + x] = 255;
    }
  }
  const out = new ToothSegmenter().detect({ data, width: W, height: H }, aperture,
    fakeRoi(W, H));
  assert.equal(out.length, 0,
    'a dark mouth must yield no detections — the whiteness floor must hold');
});

test('split sensitivity is tunable and monotonic', () => {
  const W = 192, H = 144;
  const { image, aperture } = syntheticRoi(W, H, { nTeeth: 7 });
  const roi = fakeRoi(W, H);
  const low = new ToothSegmenter({ splitRatio: 0.86 }).detect(image, aperture, roi);
  const high = new ToothSegmenter({ splitRatio: 0.99 }).detect(image, aperture, roi);
  assert.ok(high.length >= low.length,
    `higher sensitivity must not find fewer teeth (${low.length} -> ${high.length})`);
});

// ------------------------------------------------- regression on real data
/**
 * Replays 258 frames of *real* detections, exported from mouthtestvideo.mp4 by
 * tools/prototype_tooth_seg.py, through the tracker. This is the regression
 * guard for identity stability: it measures the property the spec actually
 * cares about (IDs persist frame to frame, and we do not mint a new one every
 * time a crown boundary wobbles) on real data rather than synthetic input.
 */
test('tracker holds identity across real recorded detections', async () => {
  const { readFileSync } = await import('node:fs');
  const { fileURLToPath } = await import('node:url');
  const path = fileURLToPath(new URL('./fixtures_real_detections.json', import.meta.url));
  const frames = JSON.parse(readFileSync(path, 'utf8'));
  assert.ok(frames.length > 100, 'fixture should cover a real sequence');

  const tr = new ToothTracker();
  let prev = null, persist = [], ids = new Set();
  const counts = [];
  for (const f of frames) {
    tr.update(f ?? []);
    const vis = tr.visibleTracks().map((t) => t.id);
    for (const i of vis) ids.add(i);
    if (f && f.length) counts.push(vis.length);
    if (prev && prev.length && vis.length) {
      const a = new Set(prev);
      persist.push(vis.filter((x) => a.has(x)).length / vis.length);
    }
    prev = (f && f.length) ? vis : null;
  }
  const avg = (a) => a.reduce((p, c) => p + c, 0) / a.length;
  const persistence = avg(persist);
  const meanVisible = avg(counts);

  assert.ok(persistence > 0.80,
    `ID persistence ${persistence.toFixed(3)} must stay high on real data`);
  assert.ok(meanVisible > 5,
    `should track several teeth per frame, got ${meanVisible.toFixed(2)}`);
  // A fresh ID for every tooth on every frame would be ~2000; anything near
  // that means identity has collapsed.
  assert.ok(ids.size < 140,
    `created ${ids.size} IDs over ${frames.length} frames — too much churn`);
});
