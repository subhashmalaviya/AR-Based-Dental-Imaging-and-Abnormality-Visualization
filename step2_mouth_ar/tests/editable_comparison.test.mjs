import test from 'node:test';
import assert from 'node:assert/strict';
import { drawPredictionOutline } from '../src/ui/ToothMaskEditor.js';
import { UploadedPhotoComparison } from '../src/ui/UploadedPhotoComparison.js';

// Minimal pixel canvas/DOM: verify rendering coordinates and controller wiring
// without a browser or loading a model. Text drawing is irrelevant to mask data.
function element() {
  return { value: '', checked: false, style: {}, events: {}, children: [],
    addEventListener(name, fn) { this.events[name] = fn; },
    append(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    querySelectorAll() { return []; } };
}
function canvas(w = 0, h = 0, value = 0) {
  const c = Object.assign(element(), { width: w, height: h });
  let pixels = new Uint8ClampedArray(w * h * 4).fill(value);
  const ctx = {
    drawImage(source, x, y) {
      assert.equal(x, 0); assert.equal(y, 0);
      assert.equal(source.width, c.width); assert.equal(source.height, c.height);
      pixels = source.getContext('2d').getImageData().data.slice();
    },
    getImageData() { return { data: pixels.slice() }; },
    putImageData(image, x, y) { assert.equal(x, 0); assert.equal(y, 0); pixels = image.data.slice(); },
    fillRect() {
      pixels = new Uint8ClampedArray(c.width * c.height * 4);
      for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
    },
    strokeText() {}, fillText() {},
  };
  c.getContext = () => ctx;
  return c;
}
const rgba = c => c.getContext('2d').getImageData().data;

test('model outline stays on original boundary pixels, including touching instances and crop edges', () => {
  const labels = new Int32Array(9 * 7);
  for (let y = 1; y < 6; y++) for (let x = 1; x < 8; x++) labels[y * 9 + x] = x < 4 ? 1 : 2;
  labels[0] = 3;
  const saved = labels.slice(), c = canvas(9, 7, 17);
  drawPredictionOutline(c, labels);
  const pixel = (x, y) => [...rgba(c).slice((y * 9 + x) * 4, (y * 9 + x) * 4 + 4)];
  for (const [x, y] of [[0, 0], [1, 3], [3, 3], [4, 3], [7, 3]]) {
    assert.deepEqual(pixel(x, y), [255, 90, 54, 255]);
  }
  for (const [x, y] of [[0, 3], [2, 3], [5, 3], [8, 3]]) assert.deepEqual(pixel(x, y), [17, 17, 17, 17]);
  assert.deepEqual(labels, saved, 'display must not modify model labels');
  assert.throws(() => drawPredictionOutline(c, [1]), /dimensions/);
});

test('editor, results and scores share exact edited labels across background changes and undo', t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    createElement: tag => tag === 'canvas' ? canvas() : element(),
  } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'document', previous);
    else delete globalThis.document;
  });
  const elements = new Map();
  const get = selector => {
    if (!elements.has(selector)) {
      const el = selector.endsWith('Canvas') ? canvas() : element();
      el.querySelector = get; elements.set(selector, el);
    }
    return elements.get(selector);
  };
  get('#toothEditBackground').value = 'marked';
  get('#toothEditOpacity').value = '0.55';
  const app = new UploadedPhotoComparison({ root: { querySelector: get }, setBusy() {} });
  const clean = canvas(5, 3, 20), marked = canvas(5, 3, 90);
  const gt = new Int32Array(15), pred = new Int32Array(15);
  gt[6] = pred[6] = pred[7] = 1;
  const originalPrediction = pred.slice();
  Object.assign(app, { input: clean, clean, markedCrop: marked, truthLabels: gt,
    predictionLabels: pred, crop: [10, 20, 5, 3], manualEdits: 0 });
  app.editor.setData(gt, clean, marked);
  app.updateComparison();
  assert.equal(app.result.scores.iou, 0.5);
  assert.deepEqual(rgba(get('#toothEditCanvas')), rgba(get('#uploadGtCanvas')));
  assert.equal(rgba(get('#uploadGtCanvas'))[0], 90, 'results use marked background just like the editor');

  app.editor.saveUndo();
  app.editor.labels[7] = 1;
  app.editor.changed();
  assert.strictEqual(app.truthLabels, app.editor.labels);
  assert.equal(app.result.scores.iou, 1);
  assert.equal(app.result.perTooth.rows[0].iou, 1);
  assert.deepEqual([...app.truth], [...app.editor.labels]);
  const savedResult = app.result;
  for (const background of ['clean', 'mask', 'marked']) {
    get('#toothEditBackground').value = background;
    get('#toothEditBackground').events.input();
    assert.strictEqual(app.result, savedResult, 'display settings never recompute evaluation');
    assert.strictEqual(app.truthLabels, gt, 'display settings never transform GT labels');
    assert.deepEqual(rgba(get('#toothEditCanvas')), rgba(get('#uploadGtCanvas')));
    assert.equal(rgba(get('#uploadGtCanvas'))[0], background === 'clean' ? 20 : background === 'mask' ? 0 : 90);
  }
  get('#toothEditOpacity').value = '0';
  get('#toothEditOpacity').events.input();
  assert.deepEqual(rgba(get('#uploadGtCanvas')), rgba(marked));
  assert.equal(app.result.scores.iou, 1, 'invisible mask still scores the edited labels');
  get('#toothEditPrediction').checked = true;
  get('#toothEditPrediction').events.change();
  assert.deepEqual([...rgba(get('#toothEditCanvas')).slice(24, 28)], [255, 90, 54, 255]);
  assert.deepEqual(pred, originalPrediction);
  get('#toothEditUndo').events.click();
  assert.notStrictEqual(app.truthLabels, gt, 'undo replaces the label array');
  assert.strictEqual(app.truthLabels, app.editor.labels, 'scores follow the replacement array');
  assert.equal(app.result.scores.iou, 0.5);
  assert.equal(app.result.perTooth.rows[0].iou, 0.5);
});
