/**
 * ToothPipeline.js — orchestrates Step 3: ROI -> detect -> track -> smooth.
 *
 * Kept separate from main.js so the Step-2 render loop stays a thin wiring
 * layer and the tooth stage can be disabled wholesale without touching face or
 * mouth tracking.
 *
 * FRAME SKIPPING (§11)
 * --------------------
 * Face and mouth tracking run every frame; **detection** runs every
 * `detectEveryN` frames, and the tracker + smoother run every frame in between.
 * This is safe here specifically because detections live in mouth-local
 * coordinates: between detections the teeth do not move in that frame at all,
 * so the anchor alone carries the overlay correctly and there is nothing to
 * extrapolate. The saving is real but modest — the segmenter works on a
 * 192x144 ROI, so it is far cheaper than the face model — which is why the
 * default is 1 (detect every frame) and skipping is offered for slower devices
 * rather than being forced on everyone.
 */
import { MouthROI } from './MouthROI.js';
import { ToothTracker } from './ToothTracker.js';
import { TrackingSmoother } from './TrackingSmoother.js';
import { createDetector } from './ToothDetector.js';
import './ToothSegmenter.js';   // registers the default detector

export class ToothPipeline {
  constructor({ detector = 'classical', smoothing = 'balanced',
                 detectEveryN = 1 } = {}) {
    this.roi = new MouthROI();
    this.detector = createDetector(detector);
    this.tracker = new ToothTracker();
    this.smoother = new TrackingSmoother(smoothing);

    this.enabled = true;
    this.detectEveryN = detectEveryN;
    this._frame = 0;
    this.lastDetections = [];
    this.lastRoiImage = null;
    this.selectedId = null;

    this.timing = { detect: 0, track: 0, total: 0 };
    this.reason = null;          // why nothing was detected, for the HUD
  }

  async init() { await this.detector.init?.(); }

  setEnabled(v) {
    this.enabled = !!v;
    if (!v) this.reset();
  }

  setSmoothing(name) { this.smoother.setPreset(name); }
  setDetectEveryN(n) { this.detectEveryN = Math.max(1, n | 0); }

  reset() {
    this.tracker.reset();
    this.smoother.reset();
    this.lastDetections = [];
    this.selectedId = null;
    this._frame = 0;
  }

  /**
   * @param {HTMLVideoElement} video
   * @param {Array} landmarks raw MediaPipe landmarks (null when no face)
   * @param {object} mouth MouthTracker output (null when untracked)
   * @param {MouthARAnchor} anchor
   * @param {number} tSec
   */
  update(video, landmarks, mouth, anchor, tSec, frameW, frameH) {
    const t0 = performance.now();
    this.reason = null;

    if (!this.enabled) { this.reason = 'disabled'; return this._empty(t0); }
    const pose = anchor?.getPose();
    if (!landmarks || !mouth || !pose) {
      this.reason = 'no mouth';
      // Keep tracks alive briefly rather than wiping them on a single dropout.
      const tracks = this.tracker.update([]);
      this.smoother.apply(tracks, tSec);
      return this._finish(t0, 0);
    }

    // Teeth are simply not visible below a small opening; reporting that is
    // more honest than segmenting lip highlights and calling them teeth.
    if ((mouth.opening?.ratio ?? 0) < 0.10) {
      this.reason = 'mouth closed';
      const tracks = this.tracker.update([]);
      this.smoother.apply(tracks, tSec);
      return this._finish(t0, 0);
    }

    if (!this.roi.computeBounds(landmarks, pose, frameW, frameH)) {
      this.reason = 'roi failed';
      const tracks = this.tracker.update([]);
      this.smoother.apply(tracks, tSec);
      return this._finish(t0, 0);
    }

    let detectMs = 0;
    const shouldDetect = (this._frame % this.detectEveryN) === 0;
    if (shouldDetect) {
      const d0 = performance.now();
      const image = this.roi.extract(video);
      if (image) {
        this.lastRoiImage = image;
        const aperture = this.roi.apertureMask();
        this.lastDetections = this.detector.detect(image, aperture, this.roi) ?? [];
      } else {
        this.reason = 'roi extract failed';
        this.lastDetections = [];
      }
      detectMs = performance.now() - d0;
    }
    this._frame += 1;

    const tracks = this.tracker.update(shouldDetect ? this.lastDetections : []);
    this.smoother.apply(tracks, tSec);
    return this._finish(t0, detectMs);
  }

  _empty(t0) {
    this.timing = { detect: 0, track: 0, total: performance.now() - t0 };
    return { tracks: [], stats: this.tracker.stats() };
  }

  _finish(t0, detectMs) {
    const total = performance.now() - t0;
    // Exponentially smoothed so the HUD numbers are readable rather than
    // flickering on every frame.
    const a = 0.2;
    this.timing.detect = this.timing.detect * (1 - a) + detectMs * a;
    this.timing.track = this.timing.track * (1 - a) + Math.max(0, total - detectMs) * a;
    this.timing.total = this.timing.total * (1 - a) + total * a;
    return { tracks: this.tracker.visibleTracks(), stats: this.tracker.stats() };
  }

  /** Tap-to-select, from a canvas-space point. */
  selectAt(canvasPoint, anchor) {
    if (!anchor?.isValid()) return null;
    const local = anchor.screenToLocal(canvasPoint);
    if (!local) return null;
    const hit = this.tracker.pick(local.x, local.y);
    this.selectedId = hit ? hit.id : null;
    return hit;
  }

  getSelected() {
    return this.tracker.tracks.find((t) => t.id === this.selectedId) ?? null;
  }
}
