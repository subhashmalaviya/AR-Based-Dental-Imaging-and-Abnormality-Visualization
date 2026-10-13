import { LearnedToothDetector } from '../core/LearnedToothDetector.js';
import { ToothSegmenter } from '../core/ToothSegmenter.js';
import { PHOTO_GT, expandCropMask, scorePhotoMasks } from '../eval/photoGroundTruth.js';
import { alignReferenceMask, scoreReferenceOverlap } from '../eval/liveReference.js';
import { UploadedPhotoComparison } from './UploadedPhotoComparison.js';

const percent = value => value == null ? '—' : `${(100 * value).toFixed(2)}%`;
const makeCanvas = (w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h });
const loadImage = async url => {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
};

/** Reference display is independent of the live pipeline and its model session. */
export class GroundTruthComparison {
  constructor({ root, modelUrl, getDetector, startCamera }) {
    this.root = root;
    this.modelUrl = modelUrl;
    this.getDetector = getDetector;
    this.mode = 'live';
    this.busy = false;
    this.lastDraw = -Infinity;
    this.base = `${import.meta.env.BASE_URL}ground-truth/`;
    this.el = id => root.querySelector(`#${id}`);
    this.el('gtReferenceImage').src = `${this.base}annotated.jpg`;
    this.el('gtDownloadMask').href = `${this.base}mask.png`;
    this.el('gtPhotoBtn').addEventListener('click', () => this.testPhoto());
    this.el('gtLiveBtn').addEventListener('click', () => this.showLive());
    this.el('gtExportBtn').addEventListener('click', () => this.exportResult());
    this.uploads = new UploadedPhotoComparison({ root: this.el('gtUploadPanel'), modelUrl, getDetector,
      setBusy: busy => {
        this.busy = busy;
        for (const id of ['gtLiveBtn', 'gtPhotoBtn', 'gtUploadBtn']) this.el(id).disabled = busy;
      } });
    this.el('gtUploadBtn').addEventListener('click', () => {
      this.mode = 'upload';
      this.el('gtBuiltinContent').hidden = true;
      this.el('gtUploadPanel').hidden = false;
      this.el('gtUploadBtn').setAttribute('aria-pressed', 'true');
      this.el('gtLiveBtn').setAttribute('aria-pressed', 'false');
      this.el('gtPhotoBtn').setAttribute('aria-pressed', 'false');
    });
    this.el('gtStartCamera').addEventListener('click', startCamera);
    this.el('gtCaptureBtn').addEventListener('click', () => {
      if (!this.liveFrame) return;
      this.frozen = true;
      this.el('gtCaptureBtn').disabled = true;
      this.el('gtResumeBtn').hidden = false;
      this.renderLive();
    });
    this.el('gtResumeBtn').addEventListener('click', () => {
      this.frozen = false; this.liveFrame = null;
      this.el('gtResumeBtn').hidden = true;
      this.clearLive('Waiting for the next completed camera prediction…');
    });
    for (const id of ['gtView', 'gtScaleX', 'gtScaleY', 'gtOffsetX', 'gtOffsetY', 'gtRotation', 'gtFlip']) {
      this.el(id).addEventListener('input', () => this.renderLive());
    }
    this.el('gtResetAlignment').addEventListener('click', () => {
      for (const id of ['gtScaleX', 'gtScaleY']) this.el(id).value = '1';
      for (const id of ['gtOffsetX', 'gtOffsetY', 'gtRotation']) this.el(id).value = '0';
      this.el('gtFlip').checked = false;
      this.renderLive();
    });
    this.showLive();
  }

  showLive() {
    this.el('gtBuiltinContent').hidden = false;
    this.el('gtUploadPanel').hidden = true;
    this.el('gtUploadBtn').setAttribute('aria-pressed', 'false');
    this.mode = 'live';
    this.frozen = false;
    this.liveFrame = null;
    this.result = null;
    this.el('gtLiveControls').hidden = false;
    this.el('gtResumeBtn').hidden = true;
    this.el('gtLiveReference').hidden = false;
    this.el('gtReferenceImage').hidden = true;
    this.el('gtReferenceTitle').textContent = 'Your ground truth aligned to this mouth';
    this.el('gtPhotoResults').hidden = true;
    this.el('gtLiveBtn').setAttribute('aria-pressed', 'true');
    this.el('gtPhotoBtn').setAttribute('aria-pressed', 'false');
    this.el('gtPredictionTitle').textContent = 'Live mouth frame + model prediction';
    this.clearLive('Start the camera and open your mouth to compare with the selected ground-truth view.');
    this.loadAssets().then(() => { if (this.mode === 'live') this.renderLive(); }).catch(error => {
      if (this.mode === 'live') this.el('gtStatus').textContent = `Could not load reference: ${error.message}. Click Live reference to retry.`;
    });
  }

  clearLive(message) {
    this.liveFrame = null;
    this.result = null;
    this.el('gtCaptureBtn').disabled = true;
    this.el('gtPhotoResults').hidden = true;
    if (this.el('gtStatus').textContent !== message) this.el('gtStatus').textContent = message;
    const c = this.el('gtPrediction');
    c.width = 440; c.height = 225;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#94a3b8'; ctx.font = '15px system-ui';
    ctx.fillText('Waiting for camera…', 20, 110);
    const reference = this.el('gtLiveReference');
    reference.getContext('2d').clearRect(0, 0, reference.width, reference.height);
  }

  updateLive(frame, reason, now) {
    if (this.mode !== 'live' || this.frozen || now - this.lastDraw < 125) return;
    this.lastDraw = now;
    if (reason || !frame || now - frame.completedAt > 1500) {
      this.clearLive(reason === 'disabled' ? 'Enable tooth detection to compare.' :
        reason === 'mouth closed' ? 'Open your mouth so the teeth can be compared.' :
          'Waiting for a detected mouth and a completed prediction…');
      return;
    }
    if (frame === this.liveFrame) return;
    this.liveFrame = frame;
    this.renderLive();
  }

  cameraStopped() {
    if (this.mode === 'live' && !this.frozen) this.showLive();
  }

  renderLive() {
    if (this.mode !== 'live' || !this.liveFrame || !this.assets) return;
    const frame = this.liveFrame, { image, roi } = frame;
    const W = image.width, H = image.height;
    const input = makeCanvas(W, H), ctx = input.getContext('2d');
    ctx.putImageData(image, 0, 0);
    let prediction;
    if (frame.labels) prediction = Uint8Array.from(frame.labels, n => n > 0 ? 1 : 0);
    else {
      const mask = makeCanvas(W, H), mctx = mask.getContext('2d');
      mctx.fillStyle = '#fff';
      for (const detection of frame.detections) {
        mctx.beginPath();
        detection.contour.forEach((p, i) => mctx[i ? 'lineTo' : 'moveTo'](
          (p.u - roi.bounds.u0) / (roi.bounds.u1 - roi.bounds.u0) * W,
          (p.v - roi.bounds.v0) / (roi.bounds.v1 - roi.bounds.v0) * H));
        mctx.closePath(); mctx.fill();
      }
      const pixels = mctx.getImageData(0, 0, W, H).data;
      prediction = Uint8Array.from({ length: W * H }, (_, i) => pixels[i * 4 + 3] >= 128 ? 1 : 0);
    }
    const view = Number(this.el('gtView').value), [x, y, sw, sh] = PHOTO_GT.crops[view];
    const source = new Uint8Array(sw * sh);
    for (let row = 0; row < sh; row++) source.set(this.assets.truth.subarray((y + row) * PHOTO_GT.width + x,
      (y + row) * PHOTO_GT.width + x + sw), row * sw);
    const alignment = { scaleX: Number(this.el('gtScaleX').value), scaleY: Number(this.el('gtScaleY').value),
      offsetX: Number(this.el('gtOffsetX').value), offsetY: Number(this.el('gtOffsetY').value),
      rotation: Number(this.el('gtRotation').value), flip: this.el('gtFlip').checked };
    const reference = alignReferenceMask(source, sw, sh, frame.aperture, W, H, alignment);
    const scores = scoreReferenceOverlap(reference, prediction);
    this.paintMask(this.el('gtPrediction'), input, reference, prediction, false);
    this.paintMask(this.el('gtLiveReference'), input, reference, reference, false, [245, 210, 65]);
    this.paintMask(this.el('gtErrors'), input, reference, prediction, true);
    for (const key of ['iou', 'dice', 'precision', 'recall']) this.el(`gtMetric_${key}`).textContent = percent(scores[key]);
    this.el('gtPredictionTitle').textContent = `${this.frozen ? 'Captured' : 'Live'} mouth + ${frame.detector} prediction (cyan)`;
    this.el('gtReferenceTitle').textContent = `Aligned ${view ? 'right' : 'left'} photo ground truth (yellow)`;
    this.el('gtErrorsTitle').textContent = 'Live-frame differences against aligned reference';
    this.el('gtScoreScope').textContent = 'Reference overlap — geometric comparison, not frame-annotated accuracy. Scores cover the visible mouth crop; moving labels outside it clips them.';
    this.el('gtPhotoResults').hidden = false;
    this.el('gtCaptureBtn').disabled = !!this.frozen;
    const status = this.frozen ? 'Captured frame held. Adjust reference alignment or Resume live.' :
      'Comparing completed camera predictions with your aligned ground truth. Capture comparison to hold a frame.';
    if (this.el('gtStatus').textContent !== status) this.el('gtStatus').textContent = status;
    this.result = { mode: 'live-reference', detector: frame.detector, scores,
      referenceView: view, alignment, width: W, height: H, capturedAtMs: frame.capturedAt,
      scope: 'Photo GT fitted to lip aperture plus manual alignment; not frame-specific accuracy',
      predictionMask: Array.from(prediction), alignedReferenceMask: Array.from(reference) };
  }

  async loadAssets() {
    if (this.assets) return this.assets;
    const [clean, mask] = await Promise.all(['clean.jpg', 'mask.png'].map(name => loadImage(this.base + name)));
    if ([clean, mask].some(im => im.width !== PHOTO_GT.width || im.height !== PHOTO_GT.height)) {
      throw new Error('Reference images must be 880 × 450');
    }
    const c = makeCanvas(PHOTO_GT.width, PHOTO_GT.height), ctx = c.getContext('2d');
    ctx.drawImage(mask, 0, 0);
    const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
    const truth = Uint8Array.from({ length: c.width * c.height }, (_, i) => rgba[i * 4] >= 128 ? 1 : 0);
    this.assets = { clean, truth };
    return this.assets;
  }

  async testPhoto() {
    if (this.busy) return;
    this.busy = true;
    this.el('gtBuiltinContent').hidden = false;
    this.el('gtUploadPanel').hidden = true;
    this.el('gtUploadBtn').setAttribute('aria-pressed', 'false');
    this.el('gtUploadBtn').disabled = true;
    this.el('gtPhotoBtn').disabled = true;
    this.el('gtLiveBtn').disabled = true;
    this.mode = 'photo';
    this.el('gtLiveControls').hidden = true;
    this.el('gtLiveReference').hidden = true;
    this.el('gtReferenceImage').hidden = false;
    this.el('gtReferenceTitle').textContent = 'Your hand-marked ground truth';
    this.el('gtErrorsTitle').textContent = 'Pixel differences on the matching photo';
    this.el('gtScoreScope').textContent = 'Matching-photo evaluation against your hand-marked regions.';
    this.el('gtPhotoResults').hidden = true;
    this.result = null;
    this.el('gtLiveBtn').setAttribute('aria-pressed', 'false');
    this.el('gtPhotoBtn').setAttribute('aria-pressed', 'true');
    this.el('gtPredictionTitle').textContent = 'Running detector on clean reference photo…';
    const canvas = this.el('gtPrediction');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    const selection = this.getDetector(); // Freeze detector choice/parameters for this run.
    let detector;
    try {
      this.el('gtStatus').textContent = `Loading ${selection.key} detector and your reference…`;
      const { clean, truth } = await this.loadAssets();
      detector = selection.key === 'learned'
        ? new LearnedToothDetector({ modelUrl: this.modelUrl }) : new ToothSegmenter();
      if (detector.init) await detector.init();
      detector.setParams?.(selection.params);
      const W = detector.inputWidth ?? 192, H = detector.inputHeight ?? 144;
      const prediction = new Uint8Array(PHOTO_GT.width * PHOTO_GT.height);
      const cropCanvas = makeCanvas(W, H), context = cropCanvas.getContext('2d', { willReadFrequently: true });
      const roi = { width: W, height: H, bounds: { u0: 0, u1: 1, v0: 0, v1: 1 },
        roiToLocal: (x, y) => ({ u: x / W, v: y / H }) };
      for (const [index, rect] of PHOTO_GT.crops.entries()) {
        this.el('gtStatus').textContent = `Testing ${selection.key}: photo ${index + 1} of 2…`;
        context.drawImage(clean, ...rect, 0, 0, W, H);
        let input = context.getImageData(0, 0, W, H);
        // Match the offline benchmark's INTER_AREA preprocessing exactly for
        // the shipped model. These are clean RGB pixels, with no GT labels.
        if (selection.key === 'learned' && W === 160 && H === 120) {
          const response = await fetch(`${this.base}${index ? 'right' : 'left'}.rgb`);
          if (!response.ok) throw new Error('Clean reference crop is missing');
          const rgb = new Uint8Array(await response.arrayBuffer());
          if (rgb.length !== W * H * 3) throw new Error('Clean reference crop has incorrect dimensions');
          for (let i = 0; i < W * H; i++) {
            input.data.set(rgb.subarray(i * 3, i * 3 + 3), i * 4);
            input.data[i * 4 + 3] = 255;
          }
        }
        const aperture = new Uint8Array(W * H).fill(255);
        let small;
        if (detector.isAsync) {
          await detector.detectAsync(input, aperture, roi);
          small = detector.lastDebug.labels;
        } else {
          const detections = detector.detect(input, aperture, roi);
          context.clearRect(0, 0, W, H);
          context.fillStyle = '#fff';
          for (const d of detections) {
            context.beginPath();
            d.contour.forEach((p, i) => context[i ? 'lineTo' : 'moveTo'](p.u * W, p.v * H));
            context.closePath(); context.fill();
          }
          const pixels = context.getImageData(0, 0, W, H).data;
          small = Uint8Array.from({ length: W * H }, (_, i) => pixels[i * 4 + 3] >= 128 ? 1 : 0);
        }
        expandCropMask(prediction, small, W, H, rect);
      }
      const scores = scorePhotoMasks(truth, prediction);
      this.result = { source: 'photo_area_clean.jpg', groundTruth: 'photo_area.jpg filled outlines',
        detector: selection.key, parameters: detector.params ?? selection.params, timestamp: new Date().toISOString(),
        scope: 'Fixed photo crops; detector only, no live tracking or landmark gating',
        scores, perCard: PHOTO_GT.cards.map(rect => scorePhotoMasks(truth, prediction, [rect])) };
      this.paintMask(this.el('gtPrediction'), clean, truth, prediction, false);
      this.paintMask(this.el('gtErrors'), clean, truth, prediction, true);
      for (const key of ['iou', 'dice', 'precision', 'recall']) {
        this.el(`gtMetric_${key}`).textContent = percent(scores[key]);
      }
      this.el('gtPredictionTitle').textContent = `${selection.key === 'learned' ? 'Learned model' : 'Classical detector'} on clean reference photo`;
      this.el('gtPhotoResults').hidden = false;
      this.el('gtStatus').textContent = 'Finished. Scores compare filled masks at 880 × 450, within both photo cards. Unmarked pixels count as background. Camera tracking is excluded.';
    } catch (error) {
      this.el('gtStatus').textContent = `Comparison failed: ${error.message}. You can retry or return to Live reference.`;
    } finally {
      detector?.dispose?.();
      this.busy = false;
      this.el('gtPhotoBtn').disabled = false;
      this.el('gtLiveBtn').disabled = false;
      this.el('gtUploadBtn').disabled = false;
    }
  }

  paintMask(canvas, clean, truth, prediction, errors, tint = [56, 189, 248]) {
    canvas.width = clean.width; canvas.height = clean.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(clean, 0, 0);
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < truth.length; i++) {
      const color = errors
        ? truth[i] && prediction[i] ? [65, 210, 65] : prediction[i] ? [245, 45, 45] : truth[i] ? [35, 110, 245] : null
        : prediction[i] ? tint : null;
      if (color) for (let k = 0; k < 3; k++) image.data[i * 4 + k] = 0.3 * image.data[i * 4 + k] + 0.7 * color[k];
    }
    ctx.putImageData(image, 0, 0);
  }

  exportResult() {
    if (!this.result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(this.result, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `${this.mode}-ground-truth-comparison.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
