import test from 'node:test';
import assert from 'node:assert/strict';
import { alignReferenceMask, maskBounds, scoreReferenceOverlap } from '../src/eval/liveReference.js';
import { ToothPipeline } from '../src/core/ToothPipeline.js';
import { GroundTruthComparison } from '../src/ui/GroundTruthComparison.js';

test('reference fits aperture independently of tooth predictions', () => {
  const source = Uint8Array.from([1, 0, 1, 1]);
  const aperture = new Uint8Array(36);
  for (let y = 1; y < 5; y++) for (let x = 1; x < 5; x++) aperture[y * 6 + x] = 1;
  const result = alignReferenceMask(source, 2, 2, aperture, 6, 6);
  assert.deepEqual(maskBounds(result, 6, 6), [1, 1, 4, 4]);
  assert.equal(result.reduce((a, b) => a + b, 0), 12);
  assert.equal(result[1 * 6 + 4], 0);
  const flipped = alignReferenceMask(source, 2, 2, aperture, 6, 6, { flip: true });
  assert.equal(flipped[1 * 6 + 1], 0);
  assert.equal(flipped[1 * 6 + 4], 1);
});

test('manual alignment moves pixels and never wraps offscreen labels', () => {
  const source = Uint8Array.from([1, 0, 1, 1]);
  const aperture = Uint8Array.from([1, 1, 1, 1]);
  assert.deepEqual(Array.from(alignReferenceMask(source, 2, 2, aperture, 2, 2,
    { offsetX: 0.5 })), [0, 1, 0, 1]);
  assert.deepEqual(Array.from(alignReferenceMask(source, 2, 2, aperture, 2, 2,
    { rotation: 180 })), [1, 1, 0, 1]);
  assert.deepEqual(Array.from(alignReferenceMask(source, 2, 2, new Uint8Array(4), 2, 2)), [0, 0, 0, 0]);
});

test('live scores count negative decoder labels as background', () => {
  assert.deepEqual(scoreReferenceOverlap([1, 1, 0, 0], [2, -1, 1, 0]),
    { tp: 1, fp: 1, fn: 1, tn: 1, iou: 1/3, dice: 0.5, precision: 0.5, recall: 0.5 });
});

test('completed live sample preserves its input and copied labels across next inference', async () => {
  const pipe = new ToothPipeline({ detector: 'classical' });
  let release;
  const image = { width: 2, height: 2, data: new Uint8ClampedArray(16) };
  const labels = Int32Array.from([1, -1, 0, 1]);
  pipe.detector = { isAsync: true, ready: true, lastDebug: { labels },
    detectAsync: () => new Promise(resolve => { release = resolve; }) };
  pipe.roi = {
    computeBounds: () => true, extract: () => image,
    apertureMask: () => new Uint8Array(4).fill(255),
    snapshot: () => ({ width: 2, height: 2, bounds: { u0: 0, u1: 1, v0: 0, v1: 1 } }),
    jawRef: null, bounds: {},
  };
  const anchor = { getPose: () => ({ origin: { x: 0, y: 0, z: 0 }, basis: {}, scale: 100 }) };
  const mouth = { opening: { ratio: 0.4 } };
  pipe.update(null, [{}], mouth, anchor, 0, 640, 480);
  assert.equal(pipe.lastComparisonFrame, null);
  release([]);
  await pipe._inflight;
  assert.equal(pipe.lastComparisonFrame.image, image);
  assert.deepEqual(Array.from(pipe.lastComparisonFrame.labels), [1, -1, 0, 1]);
  labels.fill(9);
  pipe.lastRoiImage = { width: 3, height: 3 };
  assert.equal(pipe.lastComparisonFrame.image, image);
  assert.deepEqual(Array.from(pipe.lastComparisonFrame.labels), [1, -1, 0, 1]);
  pipe.update(null, [{}], mouth, anchor, 0.1, 640, 480);
  pipe.reset();
  release([]);
  await pipe._inflight;
  assert.equal(pipe.lastComparisonFrame, null, 'reset also rejects late comparison samples');
});

test('live UI uses fresh samples, freezes capture, and clears stale/missing-mouth results', () => {
  // Exercise the actual controller transition logic without a camera or browser.
  const controller = Object.create(GroundTruthComparison.prototype);
  Object.assign(controller, { mode: 'live', frozen: false, lastDraw: -Infinity });
  let rendered = 0, cleared = 0;
  controller.renderLive = () => { rendered++; };
  controller.clearLive = () => { cleared++; controller.liveFrame = null; };
  const frame = { completedAt: 100 };
  controller.updateLive(frame, null, 200);
  assert.equal(rendered, 1);
  controller.frozen = true;
  controller.updateLive({ completedAt: 300 }, null, 400);
  assert.equal(controller.liveFrame, frame);
  controller.frozen = false;
  controller.updateLive(frame, 'no mouth', 500);
  assert.equal(cleared, 1);
  controller.updateLive(frame, null, 3000);
  assert.equal(cleared, 2);
  controller.mode = 'photo';
  controller.updateLive({ completedAt: 4000 }, null, 4100);
  assert.equal(rendered, 1, 'live frames cannot overwrite the photo test');
});
