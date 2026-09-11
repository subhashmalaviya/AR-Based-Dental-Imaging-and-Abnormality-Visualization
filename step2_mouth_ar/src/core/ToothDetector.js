/**
 * ToothDetector.js — the pluggable detector interface.
 *
 * ===========================================================================
 * WHY THIS INTERFACE EXISTS
 * ===========================================================================
 * Two implementations are registered:
 *
 *   'learned'    LearnedToothDetector.js — a CNN trained on openly licensed
 *                tooth data (DentalAI per-tooth polygons + EasyPortrait selfie
 *                teeth masks), run on-device with ONNX Runtime Web. Default
 *                when its model file is present.
 *   'classical'  ToothSegmenter.js — the original hand-designed method
 *                (whiteness threshold + interdental split). Not a neural
 *                network; kept as a fallback and as the measured baseline.
 *
 * Most published tooth models are trained on CBCT or panoramic X-rays and are
 * useless on a phone camera image; the one directly relevant RGB model found
 * (SegmentAnyTooth) releases weights only under a signed non-commercial
 * agreement. This interface is the seam: a model obtained that way, or a
 * custom-trained one, drops in by implementing detect()/detectAsync().
 * ===========================================================================
 *
 * Contract
 * --------
 * detect(roiImageData, apertureMask, roi) -> ToothDetection[]          (sync)
 * detectAsync(roiImageData, apertureMask, roi) -> Promise<ToothDetection[]>
 *   (learned models; `isAsync` is true and the pipeline never blocks on it)
 *
 * Detections are returned in **mouth-local coordinates** (the Step-2 anchor
 * contract: 1.0 = mouth width, +X towards the subject's right corner, +Y
 * towards the chin). Working in that frame — not in pixels — is what lets the
 * tracker match teeth across frames while the head moves, and is what Step 4
 * will consume.
 *
 * @typedef {object} ToothDetection
 * @property {{u:number,v:number}} center     centroid, mouth-local
 * @property {{u:number,v:number,w:number,h:number}} box  bbox, mouth-local
 * @property {Array<{u:number,v:number}>} contour         outline, mouth-local
 * @property {number} area        contour area in mouth-local units^2
 * @property {number} confidence  0..1 (see each detector for its definition)
 * @property {'upper'|'lower'} arch
 * @property {'full'|'partial'} [visibility]
 * @property {object} [mask]       instance mask reference (learned detector)
 */

export class ToothDetector {
  /** Human-readable name shown in the UI. */
  get name() { return 'abstract'; }

  /** True for methods that are genuinely learned models. */
  get isLearnedModel() { return false; }

  /** Optional async setup (model download, warm-up). */
  async init() {}

  /**
   * @returns {ToothDetection[]}
   */
  detect(_roiImageData, _apertureMask, _roi) {
    throw new Error('ToothDetector.detect() must be implemented');
  }

  dispose() {}
}

/**
 * Registry so the UI can offer whatever detectors are available and a future
 * ONNX/TFJS model can be registered without editing main.js.
 */
const registry = new Map();

export function registerDetector(key, factory, meta = {}) {
  registry.set(key, { factory, meta });
}

export function listDetectors() {
  return [...registry.entries()].map(([key, { meta }]) => ({ key, ...meta }));
}

export function createDetector(key, opts = {}) {
  const entry = registry.get(key);
  if (!entry) throw new Error(`unknown tooth detector: ${key}`);
  return entry.factory(opts);
}
