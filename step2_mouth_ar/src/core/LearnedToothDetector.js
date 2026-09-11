/**
 * LearnedToothDetector.js — per-tooth instance segmentation with a trained CNN.
 *
 * Model: a small U-Net (≈0.1 M parameters) trained by
 * tools/train/train_tooth_model.py on two openly licensed datasets:
 *
 *   DentalAI     2,495 intraoral photographs, 22,731 per-tooth polygons
 *                (CC BY 4.0, P. Valluri, Kaggle / Dataset Ninja)
 *                -> teaches where one tooth ends and the next begins
 *   EasyPortrait selfie-style portraits with a TEETH mask class
 *                (CC BY-SA 4.0 variant, Kvanchiani et al.)
 *                -> teaches what teeth look like to a phone front camera,
 *                   cropped with the app's own mouth rectification
 *
 * It predicts teeth / tooth-centre / interdental-boundary maps over the
 * rectified mouth ROI; toothDecode.js turns those into instances. Inference
 * runs entirely on the device with ONNX Runtime Web (WebAssembly); no frame
 * is ever sent anywhere.
 *
 * Upper/lower jaw is assigned geometrically (see assignJaws) — the training
 * data has no jaw labels, and the UI does not pretend otherwise.
 */
import { ToothDetector, registerDetector } from './ToothDetector.js';
import { assignJaws, decodeToothMaps } from './toothDecode.js';

const IS_NODE = typeof process !== 'undefined' && !!process.versions?.node
  && typeof window === 'undefined';

let ortPromise = null;
async function loadOrt(wasmBase) {
  if (!ortPromise) {
    // In Node (evaluation harness) the package's own Node build is used; in
    // the browser, the WebAssembly-only build (no WebGL/WebGPU code shipped).
    const load = IS_NODE ? import(/* @vite-ignore */ 'onnxruntime-web') : import('onnxruntime-web/wasm');
    ortPromise = load.then((m) => {
      const ort = m.default ?? m;
      // Browser: always load the runtime from the vendored, same-origin copy
      // in public/ort (scripts/vendor-ort.mjs), as an absolute URL.
      const base = wasmBase ?? (IS_NODE ? null : new URL('ort/', window.location.href).href);
      if (base) ort.env.wasm.wasmPaths = base;
      const iso = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
      const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
      // Threads need cross-origin isolation (SharedArrayBuffer); without it,
      // one thread is the only option and is still fast enough at 160x120.
      ort.env.wasm.numThreads = iso ? Math.min(4, cores) : 1;
      return ort;
    });
  }
  return ortPromise;
}

// Node-only module, loaded only by the evaluation harness / tests. Kept out
// of the browser bundle's static graph.
const NODE_FS = 'node:fs/promises';

async function readModel(url) {
  if (IS_NODE) {
    const fs = await import(/* @vite-ignore */ NODE_FS);
    return new Uint8Array(await fs.readFile(url));
  }
  const r = await fetch(url);
  // Static hosts / dev servers often answer a missing file with index.html
  // (HTTP 200, SPA fallback). Treat an HTML answer as "not found", not as a
  // corrupt model.
  const type = r.headers.get('content-type') ?? '';
  if (!r.ok || type.includes('text/html')) {
    throw new Error(`tooth model not found at ${url}`);
  }
  return new Uint8Array(await r.arrayBuffer());
}

async function readJson(url) {
  try {
    if (IS_NODE) {
      const fs = await import(/* @vite-ignore */ NODE_FS);
      return JSON.parse(await fs.readFile(url, 'utf8'));
    }
    const r = await fetch(url);
    if (!r.ok || (r.headers.get('content-type') ?? '').includes('text/html')) return null;
    return await r.json();
  } catch { return null; }
}

export class LearnedToothDetector extends ToothDetector {
  constructor({ modelUrl, infoUrl, wasmBase } = {}) {
    super();
    this.modelUrl = modelUrl;
    this.infoUrl = infoUrl ?? modelUrl?.replace(/\.onnx$/, '.json');
    this.wasmBase = wasmBase;
    this.info = null;
    this.session = null;
    this.inputWidth = 160;
    this.inputHeight = 120;
    this.mean = [0.5, 0.5, 0.5];
    this.std = [0.25, 0.25, 0.25];
    this.params = {};
    this.lastDebug = null;
    this.lastInferenceMs = 0;
    this.ready = false;
  }

  get name() { return 'Learned tooth segmenter (U-Net)'; }
  get isLearnedModel() { return true; }
  get isAsync() { return true; }
  /** ROI size and geometry this model expects (from its model card). */
  get preferredRoiSize() {
    return {
      width: this.inputWidth,
      height: this.inputHeight,
      padding: this.info?.roi?.padding ?? 0.16,
      padTop: this.info?.roi?.padTop ?? this.info?.roi?.padding ?? 0.16,
    };
  }

  setParams(p) { Object.assign(this.params, p); }

  async init() {
    if (this.ready) return;
    this.info = await readJson(this.infoUrl);
    if (this.info) {
      this.inputWidth = this.info.input?.width ?? this.inputWidth;
      this.inputHeight = this.info.input?.height ?? this.inputHeight;
      this.mean = this.info.input?.mean ?? this.mean;
      this.std = this.info.input?.std ?? this.std;
      Object.assign(this.params, this.info.decode ?? {});
    }
    const ort = await loadOrt(this.wasmBase);
    this.ort = ort;
    this.session = await ort.InferenceSession.create(await readModel(this.modelUrl), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
    this.ready = true;
  }

  _tensor(image) {
    const W = this.inputWidth, H = this.inputHeight;
    if (image.width !== W || image.height !== H) {
      throw new Error(`ROI is ${image.width}x${image.height}, model expects ${W}x${H}`);
    }
    const n = W * H;
    const buf = new Float32Array(3 * n);
    const d = image.data;
    const [m0, m1, m2] = this.mean, [s0, s1, s2] = this.std;
    this.lastBrightness = this._p97(d, n);
    const g = this.params.autoGain
      ? Math.min(3, Math.max(1, (0.8 * 255) / Math.max(this.lastBrightness, 1))) : 1;
    this.lastGain = g;
    for (let i = 0; i < n; i++) {
      buf[i] = (Math.min(1, (d[i * 4] / 255) * g) - m0) / s0;
      buf[n + i] = (Math.min(1, (d[i * 4 + 1] / 255) * g) - m1) / s1;
      buf[2 * n + i] = (Math.min(1, (d[i * 4 + 2] / 255) * g) - m2) / s2;
    }
    return new this.ort.Tensor('float32', buf, [1, 3, H, W]);
  }

  /**
   * Auto-gain for dim mouths: scale so the ROI's 97th-percentile brightness
   * reaches ~0.8 (enamel is the brightest thing in a mouth), capped at 3x.
   * Never darkens. Cheap: one histogram over 19k pixels.
   */
  _p97(d, n) {
    // 97th-percentile brightness of the mouth crop: where the enamel sits.
    // Also drives the app's low-light warning.
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) hist[Math.max(d[i * 4], d[i * 4 + 1], d[i * 4 + 2])]++;
    let acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= 0.97 * n) return v; }
    return 255;
  }

  /** Raw probability maps for one ROI (exposed for the debug view / tests). */
  async predictMaps(image) {
    const t0 = performance.now();
    const out = await this.session.run({ [this.inputName]: this._tensor(image) });
    this.lastInferenceMs = performance.now() - t0;
    const t = out[this.outputName];
    const n = this.inputWidth * this.inputHeight;
    const data = t.data;
    return {
      teeth: data.subarray(0, n),
      center: data.subarray(n, 2 * n),
      boundary: data.subarray(2 * n, 3 * n),
    };
  }

  /**
   * @param {ImageData} image rectified ROI at inputWidth x inputHeight
   * @param {Uint8Array} aperture lip aperture mask at the same size
   * @param {object} roi geometry (roiToLocal) captured when `image` was cut
   */
  async detectAsync(image, aperture, roi) {
    if (!this.ready || !image || !roi?.bounds) return [];
    const W = this.inputWidth, H = this.inputHeight;
    const maps = await this.predictMaps(image);
    const t1 = performance.now();
    const { instances, labels } = decodeToothMaps(maps, W, H, aperture, this.params);
    const jaws = assignJaws(instances, W, H, aperture);

    const areas = instances.map((t) => t.area).sort((a, b) => a - b);
    const medianArea = areas.length ? areas[areas.length >> 1] : 0;

    const dets = instances.map((t, k) => {
      const c = roi.roiToLocal(t.cx, t.cy);
      const p0 = roi.roiToLocal(t.bbox.x0, t.bbox.y0);
      const p1 = roi.roiToLocal(t.bbox.x1, t.bbox.y1);
      const partial = t.touchesEdge || t.area < 0.45 * medianArea;
      return {
        center: { u: c.u, v: c.v },
        box: { u: p0.u, v: p0.v, w: p1.u - p0.u, h: p1.v - p0.v },
        contour: t.contour.map(([x, y]) => roi.roiToLocal(x, y)),
        area: t.area,
        confidence: t.confidence,
        arch: jaws[k],
        visibility: partial ? 'partial' : 'full',
        seeded: t.seeded,
        // Instance mask in ROI pixels (label map + this instance's label),
        // kept so a later stage can crop exactly this tooth's pixels.
        mask: { roiWidth: W, roiHeight: H, label: t.label, bbox: t.bbox },
      };
    });

    this.lastDebug = {
      width: W, height: H,
      maps, labels,
      toothCount: dets.length,
      brightness: this.lastBrightness,
      inferenceMs: this.lastInferenceMs,
      decodeMs: performance.now() - t1,
    };
    return dets;
  }

  detect() {
    throw new Error('LearnedToothDetector is asynchronous; use detectAsync()');
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
    this.ready = false;
  }
}

registerDetector('learned', (opts = {}) => new LearnedToothDetector(opts), {
  label: 'Learned U-Net (DentalAI + EasyPortrait)',
  learned: true,
  note: 'Trained tooth instance segmentation, on-device via ONNX Runtime Web.',
});
