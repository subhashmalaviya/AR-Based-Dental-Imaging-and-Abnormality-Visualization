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
import { ToothPoseEstimator, intrinsicsForFrame } from './ToothPoseEstimator.js';
import { Tooth3DAnchorSet } from './Tooth3DAnchor.js';
import './ToothSegmenter.js';   // registers the default detector

export class ToothPipeline {
  constructor({ detector = 'classical', smoothing = 'balanced',
                 detectEveryN = 1 } = {}) {
    this.roi = new MouthROI();
    this.detector = createDetector(detector);
    this.tracker = new ToothTracker();
    this.smoother = new TrackingSmoother(smoothing);
    // Step 3b — one independent 3D spatial anchor per tracked tooth.
    this.poseEstimator = new ToothPoseEstimator();
    this.anchors3D = new Tooth3DAnchorSet();
    this.intrinsics = null;

    this.enabled = true;
    this.detectEveryN = detectEveryN;
    this._frame = 0;
    this.lastDetections = [];
    this.lastRoiImage = null;
    this.selectedId = null;

    this.timing = { detect: 0, track: 0, total: 0 };
    this.reason = null;          // why nothing was detected, for the HUD

    // Detection FPS is measured separately from camera FPS: with frame
    // skipping they are not the same number, and conflating them hides
    // whether the detector is actually keeping up.
    this._detectTimes = [];
    this.detectFps = 0;
    this._logNextFrame = false;
  }

  /** Ask the pipeline to dump one frame's full detection state to the console. */
  logNextFrame() { this._logNextFrame = true; }

  /** The segmenter's intermediate masks, for the debug view. */
  get debugData() { return this.detector.lastDebug ?? null; }

  _dumpFrame(tracks) {
    const dbg = this.debugData;
    const rows = tracks.map((t) => {
      const s = t.smoothed ?? t;
      return {
        id: t.id, arch: t.arch, status: t.status,
        confidence: +s.confidence.toFixed(3),
        center_u: +s.center.u.toFixed(4), center_v: +s.center.v.toFixed(4),
        box_w: +s.box.w.toFixed(4), box_h: +s.box.h.toFixed(4),
        contourPoints: s.contour?.length ?? 0,
        hits: t.hits, missing: t.missing,
      };
    });
    /* eslint-disable no-console */
    console.log(
      `%c[Step3] TEETH DETECTED THIS FRAME: ${tracks.length}`,
      'font-weight:bold;font-size:14px;color:#7cf6b0',
    );
    console.log(`  detector      : ${this.detector.name}`);
    console.log(`  neural network: ${this.detector.isLearnedModel ? 'YES' : 'NO — classical CV'}`);
    console.log(`  inference     : ${this.timing.detect.toFixed(2)} ms`);
    console.log(`  detection FPS : ${this.detectFps.toFixed(1)}`);
    if (dbg) {
      console.log(`  ROI           : ${dbg.width}x${dbg.height} px`);
      console.log(`  aperture px   : ${dbg.aperturePx}`);
      console.log(`  whiteness thr : ${dbg.threshold.toFixed(1)}`);
      console.log(`  candidate px  : ${dbg.candidatePx}  (pixels passing the enamel threshold)`);
      console.log(`  arch px       : ${dbg.archPx}  (kept after upper/lower arch extraction)`);
    }
    console.table(rows);

    const anchors = this.anchors3D.list();
    console.log(`%c[Step3] 3D ANCHORS: ${anchors.length} (one transform per tooth)`,
      'font-weight:bold;color:#6fd2ff');
    console.log('  provenance — position_xy/scale_xy: MEASURED (viewing ray + '
      + 'contour size); orientation: TRACKED (MediaPipe head matrix); '
      + 'position_z/scale_z: ESTIMATED (dental-arch prior); '
      + 'metric scale: ASSUMED (average 50 mm mouth). Not medical-grade.');
    if (anchors.length) {
      console.log(`  mouth distance: ${(anchors[0].pose.mouthDepthM * 100).toFixed(1)} cm `
        + `via ${anchors[0].pose.depthSource}`);
    }
    console.table(anchors.map((a) => {
      const t = a.getTransform();
      return {
        id: a.id, arch: a.arch,
        x_cm: +(t.position.x * 100).toFixed(2),
        y_cm: +(t.position.y * 100).toFixed(2),
        z_cm: +(t.position.z * 100).toFixed(2),
        rx: +t.rotation.rx.toFixed(1), ry: +t.rotation.ry.toFixed(1), rz: +t.rotation.rz.toFixed(1),
        w_mm: +(t.scale.x * 1000).toFixed(1), h_mm: +(t.scale.y * 1000).toFixed(1),
      };
    }));
    /* eslint-enable no-console */
    return rows;
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
    this.anchors3D.clear();
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
  update(video, landmarks, mouth, anchor, tSec, frameW, frameH, headMatrix = null) {
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

      const now = performance.now();
      this._detectTimes.push(now);
      while (this._detectTimes.length > 30) this._detectTimes.shift();
      const n = this._detectTimes.length;
      if (n >= 2) {
        const span = (this._detectTimes[n - 1] - this._detectTimes[0]) / 1000;
        this.detectFps = span > 0 ? (n - 1) / span : 0;
      }
    }
    this._frame += 1;

    const tracks = this.tracker.update(shouldDetect ? this.lastDetections : []);
    this.smoother.apply(tracks, tSec);

    // ---- 3D anchors -------------------------------------------------
    // Built every frame from the smoothed 2D tracks plus the head pose, so an
    // anchor follows its own tooth rather than the mouth bounding box.
    this.intrinsics = intrinsicsForFrame(frameW, frameH);
    const visible = this.tracker.visibleTracks();
    const poses = [];
    for (const t of visible) {
      const p = this.poseEstimator.estimate(t, anchor.getPose(), headMatrix, this.intrinsics);
      if (p) poses.push(p);
    }
    this.anchors3D.update(poses);

    if (this._logNextFrame) {
      this._logNextFrame = false;
      this.lastFrameDump = this._dumpFrame(visible);
    }
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
    const t = this.tracker.tracks.find((x) => x.id === this.selectedId) ?? null;
    if (!t) return null;
    // Hand the HUD the tooth's own 3D anchor alongside the 2D track, so the
    // panel can show a real transform rather than screen coordinates.
    return { ...t, anchor3D: this.anchors3D.get(t.id) };
  }

  /** Summary of the 3D stage for the HUD. */
  anchor3DInfo() {
    const list = this.anchors3D.list();
    const first = list[0]?.pose ?? null;
    return {
      count: list.length,
      mouthDepthM: first?.mouthDepthM ?? null,
      depthSource: first?.depthSource ?? null,
    };
  }
}
