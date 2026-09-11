/**
 * ToothPipeline.js — orchestrates Step 3: ROI -> detect -> track -> smooth -> 3D.
 *
 * Kept separate from main.js so the Step-2 render loop stays a thin wiring
 * layer and the tooth stage can be disabled wholesale without touching face or
 * mouth tracking.
 *
 * DETECTION SCHEDULING
 * --------------------
 * Face and mouth tracking run every frame. Detection runs every `detectEveryN`
 * frames; with an asynchronous (learned) detector it also never blocks the
 * frame loop: one inference is in flight at a time, and while it runs the
 * tracker and smoother carry the teeth. This is sound because detections live
 * in mouth-local coordinates, where teeth barely move between frames — a
 * result computed from a frame ~30 ms old is still in the right place. On a
 * slow phone this degrades gracefully into "detect as often as the device
 * can, track in between" instead of dropping camera frames.
 *
 * The tracker is only updated on frames that produced a detection result.
 * (Feeding it an empty list on skipped frames, as v1 did, made every track
 * "missing" on those frames — invisible flicker whenever N > 1.)
 */
import { MouthROI } from './MouthROI.js';
import { ToothTracker } from './ToothTracker.js';
import { TrackingSmoother } from './TrackingSmoother.js';
import { createDetector } from './ToothDetector.js';
import { ToothPoseEstimator, intrinsicsForFrame } from './ToothPoseEstimator.js';
import { Tooth3DAnchorSet } from './Tooth3DAnchor.js';
import './ToothSegmenter.js';        // registers 'classical'
import './LearnedToothDetector.js';  // registers 'learned'

/** Tracker thresholds depend on what the detector's confidence means. */
const TRACKER_PROFILE = {
  classical: { highConf: 0.30, lowConf: 0.10 },   // heuristic quality score
  learned: { highConf: 0.45, lowConf: 0.20 },     // model probability blend
};

export class ToothPipeline {
  constructor({ detector = 'classical', smoothing = 'balanced', detectEveryN = 1 } = {}) {
    this.roi = new MouthROI();
    this.detectorKey = detector;
    this.detector = createDetector(detector);
    this.tracker = new ToothTracker(TRACKER_PROFILE[detector] ?? {});
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
    this.lastError = null;

    this.timing = { detect: 0, track: 0, total: 0 };
    this.reason = null;          // why nothing was detected, for the HUD

    this._detectTimes = [];
    this.detectFps = 0;
    this._logNextFrame = false;
    this._inflight = null;
    this._pending = null;
    // Bumped on reset / detector switch: an inference still in flight from
    // before must not feed its (stale) teeth into the fresh tracker.
    this._gen = 0;
  }

  /**
   * Switch detector at runtime. Resolves once the new detector is ready; the
   * ROI is resized to the detector's preferred input so no resampling is lost.
   */
  async setDetector(key, opts = {}) {
    const det = createDetector(key, opts);
    await det.init?.();
    this.detector?.dispose?.();
    this.detector = det;
    this.detectorKey = key;
    const size = det.preferredRoiSize;
    this.roi = size ? new MouthROI(size) : new MouthROI();
    this.tracker.setParams(TRACKER_PROFILE[key] ?? {});
    this.reset();
    this._inflight = null;
    return det;
  }

  /** Ask the pipeline to dump one frame's full detection state to the console. */
  logNextFrame() { this._logNextFrame = true; }

  /** The detector's intermediate data, for the debug view. */
  get debugData() { return this.detector.lastDebug ?? null; }

  _dumpFrame(tracks) {
    const dbg = this.debugData;
    const rows = tracks.map((t) => {
      const s = t.smoothed ?? t;
      return {
        id: t.id, jaw: t.arch, status: t.status, visibility: t.visibility ?? '',
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
    console.log(`  neural network: ${this.detector.isLearnedModel ? 'YES — trained model' : 'NO — classical CV'}`);
    console.log(`  inference     : ${this.timing.detect.toFixed(2)} ms`);
    console.log(`  detection FPS : ${this.detectFps.toFixed(1)}`);
    if (dbg?.aperturePx != null) {
      console.log(`  ROI           : ${dbg.width}x${dbg.height} px`);
      console.log(`  aperture px   : ${dbg.aperturePx}`);
      console.log(`  whiteness thr : ${dbg.threshold.toFixed(1)}`);
    } else if (dbg?.maps) {
      console.log(`  ROI           : ${dbg.width}x${dbg.height} px`);
      console.log(`  model / decode: ${dbg.inferenceMs.toFixed(1)} ms / ${dbg.decodeMs.toFixed(1)} ms`);
    }
    const ts = this.tracker.lastFrameStats;
    console.log(`  tracker       : suppressed ${ts.suppressed} dup detections, merged ${ts.merged} dup tracks,`
      + ` ${ts.created} new, ${ts.lowRescued} kept alive by weak detections`);
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
    this._pending = null;
    this._gen += 1;
    this._frame = 0;
  }

  _noteDetection(ms) {
    const now = performance.now();
    this._detectTimes.push(now);
    while (this._detectTimes.length > 30) this._detectTimes.shift();
    const n = this._detectTimes.length;
    if (n >= 2) {
      const span = (this._detectTimes[n - 1] - this._detectTimes[0]) / 1000;
      this.detectFps = span > 0 ? (n - 1) / span : 0;
    }
    this.lastDetectMs = ms;
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
      this.tracker.update([]);
      this.smoother.apply(this.tracker.tracks, tSec);
      return this._finish(t0, 0);
    }

    // Teeth are simply not visible below a small opening; reporting that is
    // more honest than segmenting lip highlights and calling them teeth.
    if ((mouth.opening?.ratio ?? 0) < 0.10) {
      this.reason = 'mouth closed';
      this.tracker.update([]);
      this.smoother.apply(this.tracker.tracks, tSec);
      return this._finish(t0, 0);
    }

    if (!this.roi.computeBounds(landmarks, pose, frameW, frameH)) {
      this.reason = 'roi failed';
      this.tracker.update([]);
      this.smoother.apply(this.tracker.tracks, tSec);
      return this._finish(t0, 0);
    }

    let detectMs = 0;
    let fresh = null;           // detections to feed the tracker this frame
    let jawRef = this.roi.jawRef;
    const due = (this._frame % this.detectEveryN) === 0;
    this._frame += 1;

    if (this.detector.isAsync) {
      if (due && !this._inflight && this.detector.ready !== false) {
        const image = this.roi.extract(video);
        if (image) {
          this.lastRoiImage = image;
          const aperture = this.roi.apertureMask();
          const snap = this.roi.snapshot();
          const started = performance.now();
          const gen = this._gen;
          const job = this.detector.detectAsync(image, aperture, snap)
            .then((dets) => {
              if (gen !== this._gen) return;      // reset / switched meanwhile
              this._pending = { dets: dets ?? [], jawRef: snap.jawRef };
              this._noteDetection(performance.now() - started);
            })
            .catch((err) => { this.lastError = err; })
            .finally(() => { if (this._inflight === job) this._inflight = null; });
          this._inflight = job;
        }
      }
      if (this._pending) {
        fresh = this._pending.dets;
        jawRef = this._pending.jawRef;
        detectMs = this.lastDetectMs ?? 0;
        this._pending = null;
      }
    } else if (due) {
      const d0 = performance.now();
      const image = this.roi.extract(video);
      if (image) {
        this.lastRoiImage = image;
        const aperture = this.roi.apertureMask();
        fresh = this.detector.detect(image, aperture, this.roi) ?? [];
      } else {
        this.reason = 'roi extract failed';
        fresh = [];
      }
      detectMs = performance.now() - d0;
      this._noteDetection(detectMs);
    }

    if (fresh) {
      this.lastDetections = fresh;
      this.tracker.update(fresh, { jawRef });
    }
    this.smoother.apply(this.tracker.tracks, tSec);

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
    // flickering on every frame. For an async detector `detectMs` is the
    // inference time of the result consumed this frame (it ran off-thread of
    // the loop's own budget), and `track` excludes it.
    const a = 0.2;
    if (detectMs > 0) this.timing.detect = this.timing.detect * (1 - a) + detectMs * a;
    const own = this.detector.isAsync ? total : Math.max(0, total - detectMs);
    this.timing.track = this.timing.track * (1 - a) + own * a;
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
    return {
      ...t, status: t.status, visibility: t.visibility, anchor3D: this.anchors3D.get(t.id),
    };
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
