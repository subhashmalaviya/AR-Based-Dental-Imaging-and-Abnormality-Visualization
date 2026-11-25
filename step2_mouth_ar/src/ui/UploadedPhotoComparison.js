import { downloadErrorReport } from './downloadErrorReport.js';
import { getBoundaryOptions, onBoundaryOptions } from './boundaryMetricSettings.js';
import { formatDistance, formatCoverage, formatPass, summarizeBoundaryMetrics } from '../eval/boundaryOptions.js';
import { readImageFile, drawImagePreview, checkEditCrop } from './imageInput.js';
import { evaluationDetector } from '../eval/inferCanvas.js';
import { extractUploadTruth, rgbHue } from '../eval/uploadGroundTruth.js';
import { scoreReferenceOverlap } from '../eval/liveReference.js';
import { binaryFromLabels, resizeInstanceLabels, compareInstances, encodeLabels, decodeLabels, instanceStats } from '../eval/toothInstances.js';
import { ToothMaskEditor, drawInstances } from './ToothMaskEditor.js';
import { traceLabelContours, vectorBoundaryDistance, validateVectorContours } from '../eval/vectorBoundary.js';
import { importGroundTruth } from '../eval/importGroundTruth.js';

const canvas = (w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h });
const formatError = n => n == null ? '—' : n === 0 ? '0' : n.toPrecision(4);

export class UploadedPhotoComparison {
  constructor({ root, modelUrl, getDetector, setBusy }) {
    Object.assign(this, { root, modelUrl, getDetector, setBusy });
    this.el = id => root.querySelector(`#${id}`);
    this.el('uploadExcel').addEventListener('click', () => downloadErrorReport(this.result, this.el('uploadExcel')));
    this.el('uploadFrameDownload').addEventListener('click', () => this.downloadFrame());
    this.el('uploadUseDetection').addEventListener('click', () => this.useDetectionAsTruth());
    this.revision = 0;
    onBoundaryOptions(() => { if (!this.busy) this.updateComparison(); });
    this.editor = new ToothMaskEditor(this.el('uploadEditor'), labels => {
      this.truthLabels = labels; this.manualEdits = (this.manualEdits ?? 0) + 1;
      this.updateComparison();
      this.el('uploadStatus').textContent = 'Ground truth updated. Overall and per-tooth scores use your edited labels. Detect & compare keeps these edits.';
    }, () => this.drawComparison());
    this.el('uploadLabelsJson').addEventListener('click', () => {
      if (!this.truthLabels) return;
      this.download(new Blob([JSON.stringify({ schema: 'tooth-labels/1', cleanImage: this.cleanName,
        crop: this.crop, width: this.crop[2], height: this.crop[3], labels: encodeLabels(this.truthLabels), vectorContours: this.editor.vectors, groundTruthSource: this.editor.groundTruthSource, frame: this.frameContext ?? undefined })],
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
        const vectors = validateVectorContours(doc.vectorContours ?? [], this.input.width, this.input.height, labels);
        this.truthLabels = labels; this.manualEdits = (this.manualEdits ?? 0) + 1;
        this.editor.setData(labels, this.input, this.markedCrop);
        this.editor.groundTruthSource = doc.groundTruthSource ?? null;
        if (vectors.length) this.editor.restoreVectors(vectors);
        this.updateComparison();
        this.el('uploadStatus').textContent = 'Saved tooth labels restored.';
      } catch (error) { this.el('uploadStatus').textContent = `Could not restore labels: ${error.message}`; }
      e.target.value = '';
    });
    this.el('uploadTruthJson').addEventListener('change', async e => {
      const file = e.target.files[0];
      if (file) await this.loadTruthJson(file);
      e.target.value = '';
    });
    for (const id of ['uploadClean', 'uploadMarked']) this.el(id).addEventListener('change', () => this.loadPair(id));
    for (const id of ['uploadFormat', 'uploadColor', 'uploadHue', 'uploadDifference', 'uploadGap']) {
      this.el(id).addEventListener('input', () => this.invalidate());
    }
    this.el('uploadManual').addEventListener('click', () => this.startManual());
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
      return [Math.max(0, Math.min(this.clean.width, (e.clientX - r.left) * this.clean.width / r.width)),
        Math.max(0, Math.min(this.clean.height, (e.clientY - r.top) * this.clean.height / r.height))];
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

  loadAnnotationFrame(clean, name, frameContext, detection=null) {
    if (this.busy) throw new Error('Wait for photo inference to finish');
    checkEditCrop([0, 0, clean.width, clean.height]);
    if(detection && detection.labels?.length!==clean.width*clean.height)throw new Error('Video prediction must match the extracted frame dimensions');
    this.frameSource={clean,name,frameContext};
    this.el('uploadClean').value='';this.el('uploadMarked').value='';
    this.revision++; this.clean=clean; this.marked=clean; this.cleanName=name; this.markedName='Manual video-frame annotation';
    this.crop=[0,0,clean.width,clean.height]; this.invalidate(); this.frameContext=frameContext;
    this.input=clean;this.markedCrop=clean;this.truthLabels=new Int32Array(clean.width*clean.height);
    this.el('toothEditBackground').value='clean';this.el('toothEditTool').value='curve';
    this.editor.setData(this.truthLabels,clean,clean);
    if(detection){
      this.predictionLabels=detection.labels.slice();this.runInfo=structuredClone(detection.runInfo);
      this.el('toothEditPrediction').checked=true;
    }
    this.el('uploadInputPreview').hidden=false;this.drawSelection();
    const preview=this.el('uploadMarkedPreview');preview.width=clean.width;preview.height=clean.height;preview.getContext('2d').drawImage(clean,0,0);
    this.el('uploadEditor').hidden=false;this.el('uploadRun').disabled=false;this.el('uploadPreview').disabled=true;
    this.updateComparison();this.el('uploadStatus').textContent=detection
      ? 'Video detection frame loaded. Download the clean frame, or choose Use detected teeth as editable GT to start correcting the saved detection. Orange borders keep the original prediction; Detect & compare explicitly runs a new prediction.'
      : 'Clean video frame loaded. Draw each tooth with the freehand border, restore saved labels, or upload a matching hand-marked image above. Then Detect & compare to calculate the same boundary error as for photos.';
  }

  syncFrameActions() {
    const hasFrame = !!this.clean && Number.isFinite(this.frameContext?.timeMs);
    const hasPrediction = !!this.input && !!this.predictionLabels;
    this.el('uploadFrameDownload').hidden = !hasFrame;
    this.el('uploadFrameDownload').disabled = !!this.busy || !!this.frameDownloadBusy || !hasFrame;
    this.el('uploadUseDetection').hidden = !hasPrediction;
    this.el('uploadUseDetection').disabled = !!this.busy || !hasPrediction;
    const seeded = !!this.truthLabels && this.editor.groundTruthSource?.kind === 'model-assisted';
    this.el('uploadSeedNote').hidden = !hasPrediction && !seeded;
    this.el('uploadSeedNote').textContent = seeded
      ? 'Model-assisted ground truth: review and correct every tooth. An unchanged copy gives zero error against the same prediction; it is not independent accuracy. Move border points, paint/erase, or add missing teeth. Undo can restore the previous mask.'
      : 'Use detected teeth as editable GT replaces the editable mask with a copy of the current detection. Undo restores your previous mask. The original prediction stays unchanged.';
  }

  async downloadFrame() {
    if (this.busy || this.frameDownloadBusy || !this.clean || !Number.isFinite(this.frameContext?.timeMs)) return;
    const clean = this.clean, name = this.cleanName;
    this.frameDownloadBusy = true; this.syncFrameActions();
    try {
      // Export the extracted clean input, never the preview with borders/zoom.
      const blob = await new Promise(resolve => clean.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('Could not encode the extracted frame');
      this.download(blob, `${name.replace(/\.[^.]+$/, '')}.png`);
      this.el('uploadStatus').textContent = `Downloaded ${name} at ${clean.width} × ${clean.height} pixels, without overlays.`;
    } catch (error) { this.el('uploadStatus').textContent = `Frame download failed: ${error.message}`; }
    finally { this.frameDownloadBusy = false; this.syncFrameActions(); }
  }

  useDetectionAsTruth() {
    if (this.busy || !this.input || !this.predictionLabels) return;
    try {
      const copied = this.editor.usePrediction(this.predictionLabels, {
        kind: 'model-assisted', detector: this.runInfo?.detector ?? 'unknown',
        predictionTimestamp: this.runInfo?.timestamp ?? null, frame: this.frameContext ?? null,
        note: 'Initialized from model prediction; requires manual review. Not independent ground truth.'
      });
      if (!copied) return;
      this.el('toothEditBackground').value = 'clean';
      this.editor.draw(); this.drawComparison();
      this.el('uploadEditor').hidden = false;
      this.el('uploadStatus').textContent = 'Detected teeth copied into editable ground truth. Select a tooth and move its border points, paint/erase pixels, or add missing teeth. Multipart teeth retain their exact pixel masks. Undo and Redo include this copy operation.';
    } catch (error) { this.el('uploadStatus').textContent = error.message; }
  }

  invalidate() {
    this.result = null; this.truth = null; this.prediction = null;
    this.truthLabels = null; this.predictionLabels = null; this.input = null; this.runInfo = null; this.manualEdits = 0;
    this.el('uploadEditor').hidden = true;
    this.el('uploadResults').hidden = true;
    this.el('uploadMasks').hidden = true;
    this.extractionOptions = null;
    this.syncFrameActions();
    this.el('uploadExtractionNote').textContent = '';
    this.el('uploadManual').disabled = !this.clean;
    this.el('uploadTruthJson').disabled = !this.clean;
    this.el('uploadTruthJson').value = '';
    this.el('uploadPreview').disabled = !this.marked;
    this.el('uploadRun').disabled = !this.marked;
    this.el('uploadStatus').textContent = this.clean
      ? this.marked ? 'Ready. Preview the ground truth or Detect & compare the whole image. Selecting a mouth region is optional.'
        : 'Draw ground truth on the whole image or optionally select a mouth region. You can also upload a matching marked image.'
      : 'Choose a clean photo, with an optional matching marked image.';
  }

  async loadTruthJson(file) {
    if (this.busy || !this.clean) return;
    const revision = this.revision, clean = this.clean;
    this.busy = true; this.setBusy(true); this.editor.disabled = true;
    const controls = [...this.root.querySelectorAll('input, select, button')].map(el => [el, el.disabled]);
    for (const [el] of controls) el.disabled = true;
    let loaded = false;
    try {
      this.el('uploadStatus').textContent = 'Loading ground-truth JSON and creating editable tooth masks…';
      if (file.size > 40_000_000) throw new Error('Label file is too large (maximum 40 MB)');
      const doc = JSON.parse(await file.text());
      if (revision !== this.revision || clean !== this.clean) return;
      const imported = importGroundTruth(doc, clean.width, clean.height, this.cleanName);
      const sameRegion = JSON.stringify(this.crop) === JSON.stringify(imported.crop);
      // Keep a captured/model prediction only when its coordinate system agrees.
      if (!sameRegion) { this.predictionLabels = null; this.runInfo = null; }
      this.crop = imported.crop.slice(); this.input = this.cropImage(clean, this.crop);
      this.marked = null; this.markedName = file.name; this.markedCrop = this.input;
      this.el('uploadMarked').value = '';
      this.truthLabels = imported.labels; this.manualEdits = 0;
      this.extractionOptions = { format: 'json', source: file.name };
      this.el('toothEditBackground').value = 'clean';
      this.el('toothEditTool').value = imported.vectors.length ? 'vector-edit' : 'paint';
      this.editor.setData(this.truthLabels, this.input, this.input);
      this.editor.groundTruthSource = doc.groundTruthSource ?? null;
      if (imported.vectors.length) this.editor.restoreVectors(imported.vectors);
      this.drawSelection(); drawImagePreview(this.el('uploadMarkedPreview'), clean);
      this.el('uploadInputPreview').hidden = false; this.el('uploadEditor').hidden = false;
      this.updateComparison();
      const count = instanceStats(this.truthLabels, this.input.width).length;
      this.el('uploadExtractionNote').textContent = `${count} teeth imported from ${file.name}. Each tooth has a separate editable label.`
        + (imported.pixelOnlyTeeth ? ` ${imported.pixelOnlyTeeth} multipart or overlapping teeth are kept as editable pixel masks; use paint/erase to refine them.` : ' Use Move border points to adjust the imported contours.');
      this.el('uploadStatus').textContent = 'JSON ground truth loaded. Edit the mask below, then Detect & compare. Your edits are preserved when inference runs.';
      loaded = true;
    } catch (error) { this.el('uploadStatus').textContent = `Could not load ground-truth JSON: ${error.message}`; }
    finally {
      this.busy = false; this.setBusy(false); this.editor.disabled = false;
      for (const [el, disabled] of controls) el.disabled = disabled;
      if (loaded) { this.el('uploadRun').disabled = false; this.el('uploadPreview').disabled = true; }
      if (this.truthLabels) this.editor.refresh();
    }
  }

  async loadCleanFile(file) {
    if (this.busy) throw new Error('Wait for photo inference to finish');
    // A captured photo follows the same crop-first path as a normal upload.
    this.el('uploadClean').value = ''; this.el('uploadMarked').value = '';
    const revision = ++this.revision;
    const clean = await readImageFile(file);
    if (revision !== this.revision) return;
    this.frameSource = { clean, name: file.name, frameContext: { source: 'camera capture' } };
    this.clean = clean; this.marked = null; this.cleanName = file.name;
    this.frameContext = this.frameSource.frameContext;
    this.crop = [0, 0, clean.width, clean.height]; this.invalidate(); this.showInputs();
  }

  showInputs() {
    this.drawSelection();
    drawImagePreview(this.el('uploadMarkedPreview'), this.marked ?? this.clean);
    this.el('uploadInputPreview').hidden = false;
    this.el('uploadManual').disabled = false;
    this.el('uploadPreview').disabled = !this.marked; this.el('uploadRun').disabled = !this.marked;
    this.el('uploadStatus').textContent = `${this.clean.width} × ${this.clean.height} original pixels loaded. Compare the whole image or optionally select the mouth, then ${this.marked ? 'extract outlines or Detect & compare' : 'Draw ground truth on selected region, or upload a matching marked image'}. Editing and comparison keep native resolution; cropping is optional.`;
  }

  startManual() {
    if (this.busy || !this.clean) return;
    try {
      checkEditCrop(this.crop);
      this.invalidate();
      const [x, y, w, h] = this.crop;
      this.input = this.cropImage(this.clean, [x, y, w, h]);
      this.markedCrop = this.input; this.truthLabels = new Int32Array(w * h);
      this.extractionOptions = { format: 'manual-vector' };
      this.el('toothEditBackground').value = 'clean'; this.el('toothEditTool').value = 'curve';
      this.editor.setData(this.truthLabels, this.input, this.input);
      this.el('uploadEditor').hidden = false; this.el('uploadRun').disabled = false;
      this.updateComparison();
      this.el('uploadStatus').textContent = 'Draw each tooth border on this native-resolution crop, then Detect & compare. Copying detected teeth into ground truth requires choosing Use detected teeth as editable GT.';
    } catch (e) { this.el('uploadStatus').textContent = e.message; }
  }

  async loadPair(changedId) {
    if (this.busy) return;
    const frameSource=changedId==='uploadMarked'?this.frameSource:null;this.frameSource=frameSource;
    const rev = ++this.revision;
    this.frameContext = null; this.clean = null; this.marked = null; this.invalidate();
    this.el('uploadInputPreview').hidden = true;
    this.el('uploadPreview').disabled = true; this.el('uploadRun').disabled = true;
    const cleanFile = this.el('uploadClean').files[0], markedFile = this.el('uploadMarked').files[0];
    this.el('uploadManual').disabled = true;
    if (!cleanFile && !frameSource) return;
    try {
      this.el('uploadStatus').textContent = 'Loading images…';
      const [clean, marked] = await Promise.all([frameSource ? frameSource.clean : readImageFile(cleanFile), markedFile ? readImageFile(markedFile) : null]);
      if (rev !== this.revision) return;
      if (marked && (clean.width !== marked.width || clean.height !== marked.height)) {
        throw new Error(`Image dimensions differ: clean ${clean.width}×${clean.height}, ground truth ${marked.width}×${marked.height}. Use aligned copies of the same size.`);
      }
      Object.assign(this, { clean, marked, cleanName: frameSource?.name ?? cleanFile.name, markedName: markedFile?.name ?? null,
        crop: [0, 0, clean.width, clean.height] });
      this.frameContext=frameSource?.frameContext ?? null;
      this.invalidate(); this.showInputs();
    } catch (error) { if (rev === this.revision) this.el('uploadStatus').textContent = error.message; }
  }

  drawSelection() {
    const c = this.el('uploadSelection'); drawImagePreview(c, this.clean);
    const ctx = c.getContext('2d');
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2;
    const [x, y, w, h] = this.crop;
    ctx.strokeRect(x * c.width / this.clean.width, y * c.height / this.clean.height, w * c.width / this.clean.width, h * c.height / this.clean.height);
    this.el('uploadCropLabel').textContent = `Selected region: x=${this.crop[0]}, y=${this.crop[1]}, ${this.crop[2]} × ${this.crop[3]} pixels. Drag on the clean image to change it.`;
  }

  cropImage(source, [x, y, w, h]) {
    if (x === 0 && y === 0 && w === source.width && h === source.height) return source;
    const crop = canvas(w, h);
    crop.getContext('2d').drawImage(source, x, y, w, h, 0, 0, w, h);
    return crop;
  }

  async run(infer) {
    if (this.busy || !this.clean || (!this.marked && !this.truthLabels)) return;
    this.busy = true; this.setBusy(true); this.editor.disabled = true;
    this.root.querySelectorAll('input, select, button').forEach(el => { el.disabled = true; });
    let detector;
    try {
      this.el('uploadStatus').textContent = this.truthLabels && infer ? 'Using your current edited ground truth…' : 'Extracting separate ground-truth teeth…';
      await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
      checkEditCrop(this.crop);
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
        this.input = this.cropImage(this.clean, rect);
        this.markedCrop = this.cropImage(this.marked, rect);
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
      detector = evaluationDetector(selection, this.modelUrl);
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
      this.el('uploadPreview').disabled = !this.marked;
      this.el('uploadRun').disabled = !this.marked && !this.truthLabels;
      if (this.truthLabels) this.editor.refresh();
      this.syncFrameActions();
    }
  }

  updateComparison() {
    this.syncFrameActions();
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
    const gtContours = this.editor.getContours(), predContours = traceLabelContours(this.predictionLabels, this.input.width, this.input.height);
    const boundary = vectorBoundaryDistance(gtContours, predContours, this.input.width, this.input.height, getBoundaryOptions());
    for (const row of perTooth.rows) row.boundary = vectorBoundaryDistance(
      gtContours.filter(c => c.id === row.groundTruth), predContours.filter(c => c.id === row.prediction), this.input.width, this.input.height, getBoundaryOptions());
    for (const row of perTooth.rows) row.normalizedBoundaryError = row.boundary?.normalizedAssd ?? null;
    const matchedRows = perTooth.rows.filter(r => r.prediction != null && r.boundary?.hausdorffPx != null);
    const matchedAvgPx = matchedRows.length ? matchedRows.reduce((s, r) => s + (r.boundary.averageErrorPx ?? 0), 0) / matchedRows.length : null;
    const matchedMaxHdPx = matchedRows.length ? Math.max(...matchedRows.map(r => r.boundary.hausdorffPx)) : null;
    const matchedMeanHdPx = matchedRows.length ? matchedRows.reduce((s, r) => s + r.boundary.hausdorffPx, 0) / matchedRows.length : null;

    if (this.el('uploadMatchedAvgError')) this.el('uploadMatchedAvgError').textContent = formatDistance(matchedAvgPx);
    if (this.el('uploadMatchedMaxHausdorff')) this.el('uploadMatchedMaxHausdorff').textContent = formatDistance(matchedMaxHdPx);
    this.el('uploadHausdorff').textContent = formatDistance(boundary.hausdorffPx);
    if (this.el('uploadAverageError')) this.el('uploadAverageError').textContent = formatDistance(boundary.averageErrorPx);
    if (this.el('uploadNormalizedHausdorff')) this.el('uploadNormalizedHausdorff').textContent = formatError(boundary.normalizedHausdorff);
    if (this.el('uploadHausdorff95')) this.el('uploadHausdorff95').textContent = formatDistance(boundary.hausdorff95Px);
    this.el('uploadSdf').textContent = `${formatCoverage(boundary.sdfCoverage)} within ${boundary.tolerancePx} px`;
    this.el('uploadSdfPass').textContent = `${formatPass(boundary.sdfPass)} (requires ${formatCoverage(boundary.requiredCoverage)})`;
    this.el('uploadBoundaryError').textContent = formatError(boundary.normalizedAssd);
    this.el('uploadBoundaryStatus').textContent = boundary.status === 'ok' ? ''
      : `Distance unavailable: ${boundary.status}. Missing boundaries are not a perfect match.`;

    let diagnosis = '';
    if (perTooth.extraPredictions.length > 0) {
      const topExtra = perTooth.extraPredictions.map(p => `P${p.id} at (${Math.round(p.x)}, ${Math.round(p.y)})`).join(', ');
      diagnosis = `⚠️ Notice: Detector found ${perTooth.extraPredictions.length} unmatched prediction(s) outside ground-truth teeth (${topExtra}). Distance from these background detections to the dental arch elevates full-canvas Max Hausdorff to ${formatDistance(boundary.hausdorffPx)} px and 95% HD to ${formatDistance(boundary.hausdorff95Px)} px. On actual matched teeth, mean error is ${formatDistance(matchedAvgPx)} px and max Hausdorff is ${formatDistance(matchedMaxHdPx)} px.`;
    } else if (perTooth.rows.some(r => r.prediction == null)) {
      const missed = perTooth.rows.filter(r => r.prediction == null).map(r => `T${r.groundTruth}`).join(', ');
      diagnosis = `⚠️ Notice: Teeth ${missed} were missed by the model. Overall Hausdorff measures the distance across the arch gap to the nearest detection. On matched teeth, mean error is ${formatDistance(matchedAvgPx)} px and max Hausdorff is ${formatDistance(matchedMaxHdPx)} px.`;
    }
    if (this.el('uploadBoundaryDiagnosis')) this.el('uploadBoundaryDiagnosis').textContent = diagnosis;

    this.result = { cleanImage: this.cleanName, groundTruthImage: this.markedName,
      originalSize: [this.clean.width, this.clean.height], evaluatedRegion: this.crop.slice(), scores, perTooth, boundary,
      boundaryDefinition: boundary.definition,
      matchedAverageErrorPx: matchedAvgPx,
      matchedMaxHausdorffPx: matchedMaxHdPx,
      matchedMeanHausdorffPx: matchedMeanHdPx,
      normalizedBoundaryError: boundary.normalizedAssd,
      averageErrorPx: boundary.averageErrorPx,
      normalizedAverageError: boundary.normalizedAverageError,
      hausdorff95Px: boundary.hausdorff95Px,
      normalizedHausdorff95: boundary.normalizedHausdorff95,
      truthBoundaryPoints: boundary.truthBoundaryPoints,
      predictionBoundaryPoints: boundary.predictionBoundaryPoints,
      totalBoundaryPoints: boundary.totalBoundaryPoints,
      ...this.runInfo, extraction: this.extractionOptions, manualEditOperations: this.manualEdits,
      groundTruthSource: this.editor.groundTruthSource ?? null, frame: this.frameContext ?? null,
      scope: 'Selected crop only. T/P IDs are annotation/detection IDs, not anatomical tooth numbers. Per-tooth scores use one-to-one instance matching; overall boundary scores retain all tooth borders; area scores use binary unions.' };
    this.el('uploadToothSummary').textContent = `${perTooth.groundTruthCount} GT teeth · ${perTooth.predictionCount} model teeth · ${perTooth.rows.filter(row => row.prediction != null).length} compared pairs · ${perTooth.extraPredictions.length} unmatched predictions`;
    const body = this.el('uploadToothRows'); body.replaceChildren();
    for (const row of perTooth.rows) {
      const notes = [row.prediction == null ? 'Missed' : 'Compared'];
      if (row.mergedWith.length) notes.push(`model merged T${row.mergedWith.join(', T')}`);
      if (row.splitAcross.length) notes.push(`GT split across P${row.splitAcross.join(', P')}`);
      const tr = document.createElement('tr');
      for (const value of [`T${row.groundTruth}`, row.prediction == null ? '—' : `P${row.prediction}`,
        formatError(row.normalizedBoundaryError), formatDistance(row.boundary.hausdorffPx), formatCoverage(row.boundary.sdfCoverage), formatPass(row.boundary.sdfPass), notes.join('; ')]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      body.append(tr);
    }
    for (const p of perTooth.extraPredictions) {
      const tr = document.createElement('tr');
      for (const value of ['—', `P${p.id}`, '—', '—', '—', 'Unavailable', `Unmatched prediction at (${Math.round(p.x)}, ${Math.round(p.y)})`]) {
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
