import test from 'node:test';
import assert from 'node:assert/strict';
import { extractUploadTruth } from '../src/eval/uploadGroundTruth.js';
import { compareInstances, binaryFromLabels, paintLabelStroke, encodeLabels, decodeLabels, resizeInstanceLabels } from '../src/eval/toothInstances.js';
import { scoreReferenceOverlap } from '../src/eval/liveReference.js';
import { ToothMaskEditor } from '../src/ui/ToothMaskEditor.js';

function boxes({ dividerGap = 0, outerGap = 0 } = {}) {
  const W = 64, H = 32, clean = new Uint8ClampedArray(W * H * 4), marked = clean.slice();
  for (let i = 0; i < W * H; i++) clean[i * 4 + 3] = marked[i * 4 + 3] = 255;
  for (let y = 5; y <= 26; y++) for (let x = 5; x <= 58; x++) {
    if (x === 5 || x === 58 || y === 5 || y === 26 || x === 32) marked.set([0, 80, 255, 255], (y * W + x) * 4);
  }
  for (let y = 14; y < 14 + dividerGap; y++) marked.set([0, 0, 0, 255], (y * W + 32) * 4);
  for (let x = 14; x < 14 + outerGap; x++) marked.set([0, 0, 0, 255], (5 * W + x) * 4);
  return { W, H, clean, marked };
}

test('adjacent boxes of the same blue color retain separate tooth IDs', () => {
  const { W, H, clean, marked } = boxes();
  const result = extractUploadTruth(clean, marked, W, H, { gap: 1 });
  assert.equal(result.regions, 2);
  assert.ok(result.labels[15 * W + 16] > 0);
  assert.notEqual(result.labels[15 * W + 16], result.labels[15 * W + 45]);
  assert.deepEqual(result.mask, binaryFromLabels(result.labels));
});

test('progressive repair recovers a missed outer loop and splits a broken divider', () => {
  for (const spec of [{ outerGap: 5 }, { dividerGap: 5 }]) {
    const { W, H, clean, marked } = boxes(spec);
    const result = extractUploadTruth(clean, marked, W, H, { gap: 1 });
    assert.equal(result.regions, 2, JSON.stringify(spec));
    assert.notEqual(result.labels[15 * W + 16], result.labels[15 * W + 45]);
  }
});

test('merged model region keeps perfect overall coverage but cannot match two teeth', () => {
  const gt = Int32Array.from([1, 1, 2, 2]), pred = Int32Array.from([7, 7, 7, 7]);
  assert.equal(scoreReferenceOverlap(gt, pred).iou, 1);
  const comparison = compareInstances(gt, pred, 4);
  assert.equal(comparison.rows.filter(r => r.prediction != null).length, 1);
  assert.equal(comparison.rows.filter(r => r.status === 'Missed').length, 1);
  assert.equal(comparison.rows.find(r => r.prediction === 7).iou, 0.5);
  assert.deepEqual(comparison.rows.find(r => r.prediction === 7).mergedWith, [1, 2]);
});

test('split predictions and extra detections are exposed, never hidden', () => {
  const result = compareInstances([1, 1, 1, 1, 0], [3, 3, 4, 4, 5], 5);
  assert.deepEqual(result.rows[0].splitAcross, [3, 4]);
  assert.equal(result.extraPredictions.length, 2);
  assert.equal(compareInstances([1, 1], [0, 0], 2).rows[0].recall, 0);
  assert.equal(compareInstances([0, 0], [9, 9], 2).extraPredictions[0].id, 9);
});

test('paint/erase preserve other IDs and undo restores pixel-exact label map', () => {
  const editor = Object.create(ToothMaskEditor.prototype);
  editor.labels = new Int32Array(100); editor.labels[99] = 2; editor.history = [];
  editor.saveUndo();
  paintLabelStroke(editor.labels, 10, 10, [1, 1], [8, 1], 1, 3);
  assert.equal(editor.labels[1 * 10 + 5], 3, 'long strokes have no holes');
  assert.equal(editor.labels[99], 2);
  paintLabelStroke(editor.labels, 10, 10, [4, 1], [4, 1], 1, 0);
  assert.equal(editor.labels[14], 0);
  editor.labels = editor.history.pop().labels;
  assert.equal(editor.labels[15], 0); assert.equal(editor.labels[99], 2);
});

test('instance resizing and saved editable labels retain IDs, including touching teeth', () => {
  const original = resizeInstanceLabels([4, 8, -1, 0], 2, 2, 4, 4);
  assert.deepEqual([...original.slice(0, 4)], [4, 4, 8, 8]);
  assert.deepEqual(decodeLabels(encodeLabels(original), original.length), original);
  assert.throws(() => decodeLabels([[1, 999]], 4), /Invalid/);
  assert.throws(() => decodeLabels([[1, 1]], 4), /Incomplete/);
});
