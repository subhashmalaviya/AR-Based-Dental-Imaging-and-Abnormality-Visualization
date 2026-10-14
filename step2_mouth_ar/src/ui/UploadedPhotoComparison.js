import { LearnedToothDetector } from '../core/LearnedToothDetector.js';
import { ToothSegmenter } from '../core/ToothSegmenter.js';
import { extractUploadTruth, rgbHue } from '../eval/uploadGroundTruth.js';
import { scoreReferenceOverlap } from '../eval/liveReference.js';
import { binaryFromLabels, resizeInstanceLabels, compareInstances, encodeLabels, decodeLabels, instanceStats } from '../eval/toothInstances.js';
import { ToothMaskEditor, drawInstances } from './ToothMaskEditor.js';

const canvas = (w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h });
const percent = n => n == null ? '—' : `${(n * 100).toFixed(2)}%`;
const readFile = async file => {
  if (file.size > 25 * 1024 * 1024) throw new Error('Use an image file smaller than 25 MB.');
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (image.naturalWidth * image.naturalHeight > 4_000_000) throw new Error('Use images up to 4 megapixels; resize both identically before selecting them.');
    const c = canvas(image.naturalWidth, image.naturalHeight);
    c.getContext('2d').drawImage(image, 0, 0);
    return c;
  } finally { URL.revokeObjectURL(url); }
};

export class UploadedPhotoComparison {
  constructor({ root, modelUrl, getDetector, setBusy }) {
    Object.assign(this, { root, modelUrl, getDetector, setBusy });
    this.el = id => root.querySelector(`#${id}`);
    this.revision = 0;
    this.editor = new ToothMaskEditor(this.el('uploadEditor'), labels => {
      this.truthLabels = labels; this.manualEdits = (this.manualEdits ?? 0) + 1;
      this.updateComparison();
      this.el('uploadStatus').textContent = 'Ground truth updated. Overall and per-tooth scores use your edited labels. Detect & compare keeps these edits.';
    }, () => this.drawComparison());
    this.el('uploadLabelsJson').addEventListener('click', () => {
      if (!this.truthLabels) return;
      this.download(new Blob([JSON.stringify({ schema: 'tooth-labels/1', cleanImage: this.cleanName,
        crop: this.crop, width: this.crop[2], height: this.crop[3], labels: encodeLabels(this.truthLabels) })],
      { type: 'application/json' }), 'editable-tooth-labels.json');
    });
    this.el('uploadRestoreLabels').addEventListener('change', async e => {
      const file = e.target.files[0]; if (!file || !this.input) return;
      try {
        if (file.size > 40_000_000) throw new Error('Label file is too large');
        const doc = JSON.parse(await file.text());
        if (doc.schema !== 'tooth-labels/1' || doc.cleanImage !== this.cleanName || JSON.stringify(doc.crop) !== JSON.stringify(this.crop)
          || doc.width !== this.input.width || doc.height !== this.input.height) throw new Error('Saved labels must match this clean image and selected crop');
        const labels = decodeLabels(doc.labels, this.input.width * this.input.height);
        if (instanceStats(labels, this.input.width).length > 256) throw new Error('Too many tooth labels');
        this.truthLabels = labels; this.manualEdits = (this.manualEdits ?? 0) + 1;
        this.editor.setData(labels, this.input, this.markedCrop); this.updateComparison();
        this.el('uploadStatus').textContent = 'Saved tooth labels restored.';
      } catch (error) { this.el('uploadStatus').textContent = `Could not restore labels: ${error.message}`; }
      e.target.value = '';
    });
    for (const id of ['uploadClean', 'uploadMarked']) this.el(id).addEventListener('change', () => this.loadPair());
    for (const id of ['uploadFormat', 'uploadColor', 'uploadHue', 'uploadDifference', 'uploadGap']) {
      this.el(id).addEventListener('input', () => this.invalidate());
    }
    this.el('uploadPreview').addEventListener('click', () => this.run(false));
    this.el('uploadRun').addEventListener('click', () => this.run(true));
    this.el('uploadFull').addEventListener('click', () => {
      if (!this.clean) return;
      this.crop = [0, 0, this.clean.width, this.clean.height]; this.invalidate(); this.drawSelection();
    });
    this.el('uploadJson').addEventListener('click', () => {
      if (this.result) this.download(new Blob([JSON.stringify(this.result, null, 2)], { type: 'application/json' }), 'uploaded-photo-scores.json');
    });
    for (const [id, kind] of [['uploadGtPng', 'truth'], ['uploadPredPng', 'prediction']]) {
      this.el(id).addEventListener('click', () => this.downloadMask(kind));
    }
    const c = this.el('uploadSelection');
    const point = e => {
      const r = c.getBoundingClientRect();
      return [Math.max(0, Math.min(c.width, (e.clientX - r.left) * c.width / r.width)),
        Math.max(0, Math.min(c.height, (e.clientY - r.top) * c.height / r.height))];
    };
    c.addEventListener('pointerdown', e => {
      if (!this.clean || this.busy) return;
      this.drag = point(e); c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', e => {
      if (!this.drag) return;
      const p = point(e), x = Math.floor(Math.min(p[0], this.drag[0])), y = Math.floor(Math.min(p[1], this.drag[1]));
      const w = Math.floor(Math.max(p[0], this.drag[0])) - x, h = Math.floor(Math.max(p[1], this.drag[1])) - y;
      if (w >= 8 && h >= 8) { this.crop = [x, y, w, h]; this.drawSelection(); }
    });
    const finish = () => { if (this.drag) { this.drag = null; this.invalidate(); } };
    c.addEventListener('pointerup', finish); c.addEventListener('pointercancel', finish);
  }

  invalidate() {
    this.result = null; this.truth = null; this.prediction = null;
    this.truthLabels = null; this.predictionLabels = null; this.input = null; this.runInfo = null; this.manualEdits = 0;
    this.el('uploadEditor').hidden = true;
    this.el('uploadResults').hidden = true;
    this.el('uploadMasks').hidden = true;
    this.el('uploadStatus').textContent = this.clean && this.marked
      ? 'Ready. Select the mouth area, then preview the ground truth or Detect & compare.' : 'Choose both matching images.';
  }

  async loadPair() {
    const rev = ++this.revision;
    this.clean = null; this.marked = null; this.invalidate();
    this.el('uploadInputPreview').hidden = true;
    this.el('uploadPreview').disabled = true; this.el('uploadRun').disabled = true;
    const cleanFile = this.el('uploadClean').files[0], markedFile = this.el('uploadMarked').files[0];
    if (!cleanFile || !markedFile) return;
    try {
      this.el('uploadStatus').textContent = 'Loading images…';
      const [clean, marked] = await Promise.all([readFile(cleanFile), readFile(markedFile)]);
      if (rev !== this.revision) return;
      if (clean.width !== marked.width || clean.height !== marked.height) {
        throw new Error(`Image dimensions differ: clean ${clean.width}×${clean.height}, ground truth ${marked.width}×${marked.height}. Use aligned copies of the same size.`);
      }
      Object.assign(this, { clean, marked, cleanName: cleanFile.name, markedName: markedFile.name,
        crop: [0, 0, clean.width, clean.height] });
      this.drawSelection();
      const preview = this.el('uploadMarkedPreview');
      preview.width = marked.width; preview.height = marked.height; preview.getContext('2d').drawImage(marked, 0, 0);
      this.el('uploadInputPreview').hidden = false;
      this.el('uploadPreview').disabled = false; this.el('uploadRun').disabled = false;
      this.invalidate();
    } catch (error) { if (rev === this.revision) this.el('uploadStatus').textContent = error.message; }
  }

  drawSelection() {
    const c = this.el('uploadSelection'); c.width = this.clean.width; c.height = this.clean.height;
    const ctx = c.getContext('2d'); ctx.drawImage(this.clean, 0, 0);
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = Math.max(2, c.width / 250);
    ctx.strokeRect(...this.crop);
    this.el('uploadCropLabel').textContent = `Selected region: x=${this.crop[0]}, y=${this.crop[1]}, ${this.crop[2]} × ${this.crop[3]} pixels. Drag on the clean image to change it.`;
  }

  async run(infer) {
    if (this.busy || !this.clean || !this.marked) return;
    this.busy = true; this.setBusy(true); this.editor.disabled = true;
    this.root.querySelectorAll('input, select, button').forEach(el => { el.disabled = true; });
    let detector;
    try {
      this.el('uploadStatus').textContent = this.truthLabels && infer ? 'Using your current edited ground truth…' : 'Extracting separate ground-truth teeth…';
      await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
      const rect = this.crop.slice(), [x, y, w, h] = rect;
      // Re-running inference must never discard manually corrected labels.
      if (!this.truthLabels || !infer) {
        const cleanPixels = this.clean.getContext('2d').getImageData(x, y, w, h);
        const markedPixels = this.marked.getContext('2d').getImageData(x, y, w, h);
        const hex = this.el('uploadHue').value;
        this.extractionOptions = { format: this.el('uploadFormat').value, color: this.el('uploadColor').value,
          hue: rgbHue(parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)),
          difference: Number(this.el('uploadDifference').value), gap: Number(this.el('uploadGap').value) };
        const extraction = extractUploadTruth(cleanPixels.data, markedPixels.data, w, h, this.extractionOptions);
        if (extraction.regions > 256) throw new Error('More than 256 regions found. Refine the crop or outline color to exclude noise.');
        this.truthLabels = extraction.labels; this.manualEdits = 0;
        this.input = canvas(w, h); this.input.getContext('2d').putImageData(cleanPixels, 0, 0);
        this.markedCrop = canvas(w, h); this.markedCrop.getContext('2d').putImageData(markedPixels, 0, 0);
        this.editor.setData(this.truthLabels, this.input, this.markedCrop);
        this.el('uploadExtractionNote').textContent = `${extraction.regions} separate regions extracted; ${extraction.repaired ?? 0} recovered by additional gap repair. Check every box. Add missing teeth or split merged regions with the editor. Binary masks separate only disconnected regions.`;
      }
      this.el('uploadMasks').hidden = false;
      this.el('uploadEditor').hidden = false;
      this.updateComparison();
      if (!infer) {
        this.el('uploadStatus').textContent = 'Each colored T label is one ground-truth tooth. Edit missing pixels or add a tooth, then Detect & compare.';
        return;
      }
      const selection = this.getDetector();
      this.el('uploadStatus').textContent = `Running ${selection.key} detector on the clean image…`;
      detector = selection.key === 'learned' ? new LearnedToothDetector({ modelUrl: this.modelUrl }) : new ToothSegmenter();
      await detector.init?.(); detector.setParams?.(selection.params);
      const W = detector.inputWidth ?? 192, H = detector.inputHeight ?? 144;
      const smallCanvas = canvas(W, H), ctx = smallCanvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(this.input, 0, 0, W, H);
      const roi = { width: W, height: H, bounds: { u0: 0, u1: 1, v0: 0, v1: 1 }, roiToLocal: (px, py) => ({ u: px / W, v: py / H }) };
      const image = ctx.getImageData(0, 0, W, H), aperture = new Uint8Array(W * H).fill(255);
      let small;
      if (detector.isAsync) { await detector.detectAsync(image, aperture, roi); small = detector.lastDebug.labels; }
      else {
        const detections = detector.detect(image, aperture, roi);
        small = new Int32Array(W * H);
        for (const [index, d] of detections.entries()) {
          ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff';
          ctx.beginPath(); d.contour.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](p.u * W, p.v * H)); ctx.closePath(); ctx.fill();
          const pixels = ctx.getImageData(0, 0, W, H).data;
          for (let i = 0; i < small.length; i++) if (pixels[i * 4 + 3] >= 128) small[i] = index + 1;
        }
      }
      this.predictionLabels = resizeInstanceLabels(small, W, H, w, h);
      this.runInfo = { detector: selection.key, parameters: detector.params ?? selection.params,
        modelInput: [W, H], timestamp: new Date().toISOString() };
      this.updateComparison();
      this.el('uploadStatus').textContent = `Finished using ${selection.key}. Compared ${w} × ${h} pixels. Ground-truth edits update scores immediately without changing the model prediction.`;
    } catch (error) { this.el('uploadStatus').textContent = `Comparison failed: ${error.message}`; }
    finally {
      detector?.dispose?.(); this.busy = false; this.setBusy(false); this.editor.disabled = false;
      this.root.querySelectorAll('input, select, button').forEach(el => { el.disabled = false; });
      if (this.truthLabels) this.editor.refresh();
    }
  }

  updateComparison() {
    if (!this.truthLabels || !this.input) return;
    // The editor label array is the source of truth: no extraction, resizing or
    // registration is applied between editing, scoring and rendering results.
    this.truth = binaryFromLabels(this.truthLabels);
    this.editor.setPrediction(this.predictionLabels);
    this.drawComparison();
    this.el('uploadMasks').hidden = false;
    this.el('uploadPredictionFigure').hidden = !this.predictionLabels;
    if (!this.predictionLabels) { this.el('uploadResults').hidden = true; return; }
    this.prediction = binaryFromLabels(this.predictionLabels);
    const scores = scoreReferenceOverlap(this.truth, this.prediction);
    const perTooth = compareInstances(this.truthLabels, this.predictionLabels, this.input.width);
    for (const k of ['iou', 'dice', 'precision', 'recall']) this.el(`upload_${k}`).textContent = percent(scores[k]);
    this.result = { cleanImage: this.cleanName, groundTruthImage: this.markedName,
      originalSize: [this.clean.width, this.clean.height], evaluatedRegion: this.crop.slice(), scores, perTooth,
      ...this.runInfo, extraction: this.extractionOptions, manualEditOperations: this.manualEdits,
      scope: 'Selected crop only. T/P IDs are annotation/detection IDs, not anatomical tooth numbers. Per-tooth scores use one-to-one instance matching; overall scores use binary unions.' };
    this.el('uploadToothSummary').textContent = `${perTooth.groundTruthCount} GT teeth · ${perTooth.predictionCount} model teeth · ${perTooth.matchedAt50} matched at IoU ≥ 50% · ${perTooth.extraPredictions.length} unmatched predictions`;
    const body = this.el('uploadToothRows'); body.replaceChildren();
    for (const row of perTooth.rows) {
      const notes = [row.status];
      if (row.mergedWith.length) notes.push(`model merged T${row.mergedWith.join(', T')}`);
      if (row.splitAcross.length) notes.push(`GT split across P${row.splitAcross.join(', P')}`);
      const tr = document.createElement('tr');
      for (const value of [`T${row.groundTruth}`, row.prediction == null ? '—' : `P${row.prediction}`,
        percent(row.iou), percent(row.dice), percent(row.precision), percent(row.recall), notes.join('; ')]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      body.append(tr);
    }
    for (const p of perTooth.extraPredictions) {
      const tr = document.createElement('tr');
      for (const value of ['—', `P${p.id}`, '0%', '0%', '0%', '—', 'Unmatched prediction']) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      body.append(tr);
    }
    this.el('uploadResults').hidden = false;
  }

  drawComparison() {
    if (!this.truthLabels || !this.input) return;
    // Match the editor's background and opacity, including the marked-image
    // view. Inference always uses this.input (the clean crop) independently.
    const { background, opacity } = this.editor.getView();
    drawInstances(this.el('uploadGtCanvas'), background, this.truthLabels, 'T', opacity);
    if (this.predictionLabels) {
      drawInstances(this.el('uploadPredCanvas'), background, this.predictionLabels, 'P', opacity);
      this.paint(this.el('uploadErrorCanvas'), background, this.truthLabels, this.predictionLabels);
    }
  }

  paint(c, input, truth, pred) {
    c.width = input.width; c.height = input.height;
    const ctx = c.getContext('2d'); ctx.drawImage(input, 0, 0);
    const image = ctx.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < c.width * c.height; i++) {
      const color = truth && pred ? truth[i] && pred[i] ? [65, 210, 65] : pred[i] ? [245, 45, 45] : truth[i] ? [35, 110, 245] : null
        : truth?.[i] ? [245, 210, 65] : pred?.[i] ? [56, 189, 248] : null;
      if (color) for (let k = 0; k < 3; k++) image.data[i * 4 + k] = 0.3 * image.data[i * 4 + k] + 0.7 * color[k];
    }
    ctx.putImageData(image, 0, 0);
  }

  downloadMask(kind) {
    const mask = this[kind]; if (!mask) return;
    const [, , w, h] = this.crop, c = canvas(w, h), ctx = c.getContext('2d');
    const image = ctx.createImageData(w, h);
    for (let i = 0; i < mask.length; i++) {
      image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = mask[i] ? 255 : 0;
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    c.toBlob(blob => { if (blob) this.download(blob, `uploaded-${kind}-crop.png`); });
  }

  download(blob, filename) {
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
