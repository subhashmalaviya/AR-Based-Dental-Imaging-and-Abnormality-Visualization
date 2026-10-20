import test from 'node:test';
import assert from 'node:assert/strict';
import { ToothPipeline } from '../src/core/ToothPipeline.js';
import { TrackingSmoother } from '../src/core/TrackingSmoother.js';

function pendingPipeline() {
  const pipe = new ToothPipeline({ detector: 'classical' });
  let release, reject;
  pipe.detector = { isAsync: true, ready: true,
    detectAsync: () => new Promise((yes, no) => { release = yes; reject = no; }) };
  pipe.roi = { computeBounds: () => true, extract: () => ({ width: 1, height: 1 }),
    apertureMask: () => null, snapshot: () => ({ jawRef: null }), jawRef: null };
  const anchor = { getPose: () => ({ origin: { x: 0, y: 0, z: 0 }, basis: {}, scale: 100 }) };
  const frame = (mouth = { opening: { ratio: 0.4 } }, landmarks = [{}]) => pipe.update(null, landmarks, mouth, anchor, 0, 640, 480);
  frame();
  return { pipe, frame, release: x => release(x), reject: x => reject(x) };
}

test('closing the mouth, losing landmarks or invalid ROI rejects in-flight teeth', async () => {
  for (const interrupt of [
    s => s.frame({ opening: { ratio: 0.01 } }),
    s => s.frame(null, null),
    s => { s.pipe.roi.computeBounds = () => false; s.frame(); },
  ]) {
    const s = pendingPipeline(), job = s.pipe._inflight;
    s.pipe.lastComparisonFrame = { old: true };
    interrupt(s);
    assert.equal(s.pipe.lastComparisonFrame, null);
    assert.strictEqual(s.pipe._inflight, job, 'never start concurrent inference on the same detector');
    s.release([{ confidence: 0.99 }]); await job;
    assert.equal(s.pipe._pending, null);
    assert.equal(s.pipe.lastComparisonFrame, null);
    assert.equal(s.pipe._inflight, null);
  }
});

test('completed pending results and obsolete errors cannot leak across a mouth closure', async () => {
  const s = pendingPipeline(); s.release([]); await s.pipe._inflight;
  assert.ok(s.pipe._pending);
  s.frame({ opening: { ratio: 0 } }); assert.equal(s.pipe._pending, null);
  s.frame(); const job = s.pipe._inflight;
  s.frame(null, null); s.reject(new Error('obsolete inference')); await job;
  assert.equal(s.pipe.lastError, null);
  s.frame(); s.release([]); await s.pipe._inflight;
  assert.ok(s.pipe._pending, 'fresh observations work after reacquisition');
});

function smoothSequence(displaySteps) {
  const smoother = new TrackingSmoother();
  const t = { id: 1, hits: 1, center: { u: 0, v: 0 }, box: { u: -0.05, v: -0.05, w: 0.1, h: 0.1 },
    confidence: 0.9, contour: [{ u: 0, v: 0 }] };
  const output = [];
  for (let k = 0; k < 10; k++) {
    t.hits = k + 1; t.center.u = k % 2 ? 0.01 : -0.01;
    t.box.u = t.center.u - 0.05;
    smoother.apply([t], k / 10);
    output.push(structuredClone(t.smoothed));
    const cached = t.smoothed;
    for (let j = 1; j < displaySteps; j++) smoother.apply([t], k / 10 + j / (10 * displaySteps));
    assert.strictEqual(t.smoothed, cached, 'cached detections are not repeated filter measurements');
  }
  return output;
}

test('smoothing is independent of 30 vs 120 Hz rendering for the same 10 Hz detections', () => {
  assert.deepEqual(smoothSequence(3), smoothSequence(12));
});
