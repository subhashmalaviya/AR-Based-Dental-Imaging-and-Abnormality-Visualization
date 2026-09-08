/**
 * ToothDetector.js — the pluggable detector interface.
 *
 * ===========================================================================
 * WHY THIS INTERFACE EXISTS
 * ===========================================================================
 * No suitable pretrained tooth model exists for this input. The tooth
 * segmentation models that are publicly available (railNet, DENTEX and
 * similar) are trained on **CBCT volumes and panoramic radiographs**, not on
 * RGB photographs of a mouth from a phone camera, and none ship in a form that
 * runs in a browser. Running an X-ray model on a selfie is a domain mismatch
 * that produces noise, so this project does not do that.
 *
 * The shipped detector is therefore a **classical computer-vision segmenter**
 * (ToothSegmenter.js) that measures real pixels from the live camera. It is a
 * genuine CV method — no hardcoded coordinates, no predefined rectangles, no
 * canned results — but it is explicitly *not* a neural network, and the UI
 * labels it accordingly so nothing here is passed off as an AI result.
 *
 * This interface is the seam: implement `detect()` and a learned model drops
 * in without touching the ROI, tracking, smoothing or rendering layers.
 * ===========================================================================
 *
 * Contract
 * --------
 * detect(roiImageData, apertureMask, roi, ctx) -> ToothDetection[]
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
 * @property {number} confidence  0..1, see ToothSegmenter for its definition
 * @property {'upper'|'lower'} arch
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

export function createDetector(key) {
  const entry = registry.get(key);
  if (!entry) throw new Error(`unknown tooth detector: ${key}`);
  return entry.factory();
}
