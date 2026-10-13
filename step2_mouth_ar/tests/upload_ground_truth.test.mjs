import test from 'node:test';
import assert from 'node:assert/strict';
import { extractUploadTruth, resizeBinaryMask } from '../src/eval/uploadGroundTruth.js';

function fixture(color = [0, 200, 255]) {
  const W = 24, H = 24, clean = new Uint8ClampedArray(W * H * 4), marked = clean.slice();
  for (let i = 0; i < W * H; i++) clean[i * 4 + 3] = marked[i * 4 + 3] = 255;
  for (let y = 5; y <= 18; y++) for (let x = 5; x <= 18; x++) {
    if (x === 5 || x === 18 || y === 5 || y === 18) marked.set([...color, 255], (y * W + x) * 4);
  }
  return { clean, marked, W, H };
}

test('closed colored outlines become filled regions, not just stroke pixels', () => {
  const { clean, marked, W, H } = fixture();
  const out = extractUploadTruth(clean, marked, W, H, { gap: 0 });
  assert.equal(out.regions, 1);
  assert.equal(out.mask[12 * W + 12], 1);
  assert.equal(out.mask[2 * W + 2], 0);
  assert.equal(out.strokes[12 * W + 12], 0);
});

test('unchanged colored image content does not become annotation', () => {
  const { marked, W, H } = fixture();
  assert.equal(extractUploadTruth(marked, marked, W, H).regions, 0);
});

test('small-gap repair closes a broken outline; an open outline is not silently filled', () => {
  const { clean, marked, W, H } = fixture();
  marked.set([0, 0, 0, 255], (5 * W + 12) * 4);
  assert.equal(extractUploadTruth(clean, marked, W, H, { gap: 0 }).regions, 0);
  assert.equal(extractUploadTruth(clean, marked, W, H, { gap: 1 }).regions, 1);
});

test('custom hue permits red annotations while default preset excludes them', () => {
  const { clean, marked, W, H } = fixture([255, 0, 0]);
  assert.equal(extractUploadTruth(clean, marked, W, H).regions, 0);
  assert.equal(extractUploadTruth(clean, marked, W, H, { color: 'custom', hue: 0 }).regions, 1);
});

test('binary masks use white foreground and exclude transparent white pixels', () => {
  const clean = new Uint8ClampedArray(12);
  const marked = Uint8ClampedArray.from([255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 0]);
  assert.deepEqual(Array.from(extractUploadTruth(clean, marked, 3, 1, { format: 'binary' }).mask), [1, 0, 0]);
});

test('size mismatch is rejected, and model mask resizing preserves negative background', () => {
  assert.throws(() => extractUploadTruth(new Uint8Array(4), new Uint8Array(8), 1, 1), /dimensions/);
  assert.deepEqual(Array.from(resizeBinaryMask([2, -1], 2, 1, 4, 2)), [1, 1, 0, 0, 1, 1, 0, 0]);
});
