import test from 'node:test';
import assert from 'node:assert/strict';
import { PHOTO_GT, expandCropMask, scorePhotoMasks } from '../src/eval/photoGroundTruth.js';

const blank = () => new Uint8Array(PHOTO_GT.width * PHOTO_GT.height);
test('photo confusion counts exclude pixels outside scoring cards', () => {
  const gt = blank(), pred = blank();
  gt[0] = 1; pred[0] = 1; // Outside the test region.
  gt[880 + 1] = 1; pred[880 + 1] = 1;
  gt[880 + 2] = 1;
  pred[880 + 3] = 1;
  const s = scorePhotoMasks(gt, pred, [[1, 1, 4, 1]]);
  assert.deepEqual([s.tp, s.fp, s.fn, s.tn], [1, 1, 1, 1]);
  assert.equal(s.iou, 1 / 3);
  assert.equal(s.dice, 0.5);
  assert.equal(s.precision, 0.5);
  assert.equal(s.recall, 0.5);
});

test('crop expansion preserves origin, axes and nearest-neighbor labels', () => {
  const output = blank();
  expandCropMask(output, [1, -1, 0, 2], 2, 2, [10, 20, 4, 4]);
  assert.equal(output[20 * 880 + 10], 1);
  assert.equal(output[21 * 880 + 11], 1);
  assert.equal(output[20 * 880 + 12], 0);
  assert.equal(output[23 * 880 + 13], 1);
  assert.equal(output[24 * 880 + 13], 0);
  assert.equal(output.reduce((a, b) => a + b, 0), 8);
});

test('empty masks do not manufacture perfect foreground scores', () => {
  const s = scorePhotoMasks(blank(), blank());
  assert.equal(s.iou, null);
  assert.equal(s.precision, null);
  assert.equal(s.tn, 333432);
});

test('mismatched dimensions fail before scoring', () => {
  assert.throws(() => scorePhotoMasks(blank(), new Uint8Array(1)), /880x450/);
  assert.throws(() => expandCropMask(blank(), [1], 2, 2, [0, 0, 2, 2]), /dimensions/);
});
