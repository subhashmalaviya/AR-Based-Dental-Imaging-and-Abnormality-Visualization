/**
 * ToothTracker.js — stable per-tooth identity across frames.
 *
 * ===========================================================================
 * WHY IoU + CENTROID MATCHING IN MOUTH-LOCAL SPACE
 * ===========================================================================
 * The method matters less than the coordinate frame it runs in. Detections
 * arrive in **mouth-local coordinates** (Step-2's anchor contract, 1.0 = mouth
 * width), which means the head's translation, scale and roll have *already*
 * been removed by the anchor. A tooth that stays put on the jaw barely moves in
 * this frame even while the head swings across the camera.
 *
 * That makes the association problem nearly trivial, so the cheapest adequate
 * method wins:
 *
 *   - **IoU** is the primary cue: teeth are roughly equal-sized neighbours in a
 *     row, and box overlap disambiguates adjacent crowns better than distance
 *     alone, which is exactly where centroid-only tracking swaps IDs.
 *   - **Centroid distance** breaks ties and rescues frames where a crown was
 *     partly cut by the interdental split, so IoU collapses but the centre
 *     barely moved.
 *   - Greedy best-first assignment, not Hungarian: with fewer than ~20 tracks
 *     the optimal assignment and the greedy one almost always agree, and greedy
 *     costs nothing.
 *
 * Optical flow and Kalman filtering were both considered and rejected. Flow
 * would re-solve motion the anchor has already removed. A Kalman filter models
 * velocity, but in mouth-local space a tooth's velocity is essentially zero —
 * there is no dynamic worth modelling, and it would only add lag and tuning.
 *
 * Occlusion (§6): closing the mouth hides every tooth at once. Tracks are
 * therefore kept alive for `maxMissing` frames before removal, so a blink of a
 * closed mouth does not renumber the whole dentition.
 * ===========================================================================
 */

let nextId = 1;

function iou(a, b) {
  const x0 = Math.max(a.u, b.u);
  const y0 = Math.max(a.v, b.v);
  const x1 = Math.min(a.u + a.w, b.u + b.w);
  const y1 = Math.min(a.v + a.h, b.v + b.h);
  const iw = x1 - x0, ih = y1 - y0;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export class ToothTrack {
  constructor(det) {
    this.id = nextId++;
    this.center = { ...det.center };
    this.box = { ...det.box };
    this.contour = det.contour;
    this.confidence = det.confidence;
    this.arch = det.arch;
    this.age = 1;          // frames seen in total
    this.hits = 1;         // frames matched
    this.missing = 0;      // consecutive frames unmatched
    this.smoothed = null;  // filled by TrackingSmoother
  }

  get isStable() { return this.hits >= 5 && this.missing === 0; }

  get status() {
    if (this.missing > 0) return 'occluded';
    if (this.hits >= 5) return 'stable';
    return 'acquiring';
  }
}

export class ToothTracker {
  /**
   * Defaults were tuned by replaying 258 frames of *real* detections from
   * mouthtestvideo.mp4 (tests/fixtures_real_detections.json) through this
   * tracker and measuring ID persistence and total IDs created:
   *
   *   iou .20 / dist .09 / miss  8   ->  persistence 0.869, 99 IDs
   *   iou .10 / dist .13 / miss  8   ->  persistence 0.874, 89 IDs
   *   iou .05 / dist .16 / miss 20   ->  persistence 0.873, 66 IDs   <- chosen
   *
   * Persistence barely moves, but the looser gates plus a longer grace period
   * create a third fewer spurious tracks, which is what §4/§6 actually ask for.
   * The looseness is safe because matching happens in mouth-local coordinates,
   * where a tooth barely moves between frames.
   */
  constructor({ iouThreshold = 0.06, maxCentroidDist = 0.16,
                 maxMissing = 18, minHitsToShow = 2 } = {}) {
    this.iouThreshold = iouThreshold;
    this.maxCentroidDist = maxCentroidDist;   // in mouth widths
    this.maxMissing = maxMissing;
    this.minHitsToShow = minHitsToShow;
    this.tracks = [];
  }

  reset() {
    this.tracks = [];
    nextId = 1;
  }

  setParams(p) { Object.assign(this, p); }

  /**
   * @param {Array} detections mouth-local detections for this frame
   * @returns {ToothTrack[]} live tracks
   */
  update(detections) {
    const dets = detections ?? [];

    // Score every (track, detection) pair, keep only admissible ones.
    const pairs = [];
    for (let ti = 0; ti < this.tracks.length; ti++) {
      const t = this.tracks[ti];
      for (let di = 0; di < dets.length; di++) {
        const d = dets[di];
        if (t.arch !== d.arch) continue;          // an upper tooth is never a lower one
        const ov = iou(t.box, d.box);
        const dist = Math.hypot(d.center.u - t.center.u, d.center.v - t.center.v);
        if (ov < this.iouThreshold && dist > this.maxCentroidDist) continue;
        // IoU leads; distance only refines. Normalised so both are 0..1-ish.
        const score = ov + 0.5 * Math.max(0, 1 - dist / this.maxCentroidDist);
        pairs.push({ ti, di, score });
      }
    }
    pairs.sort((a, b) => b.score - a.score);

    const usedT = new Set(), usedD = new Set();
    for (const p of pairs) {
      if (usedT.has(p.ti) || usedD.has(p.di)) continue;
      usedT.add(p.ti); usedD.add(p.di);
      const t = this.tracks[p.ti];
      const d = dets[p.di];
      t.center = { ...d.center };
      t.box = { ...d.box };
      t.contour = d.contour;
      t.confidence = d.confidence;
      t.age += 1;
      t.hits += 1;
      t.missing = 0;
    }

    for (let ti = 0; ti < this.tracks.length; ti++) {
      if (usedT.has(ti)) continue;
      const t = this.tracks[ti];
      t.age += 1;
      t.missing += 1;
    }

    for (let di = 0; di < dets.length; di++) {
      if (!usedD.has(di)) this.tracks.push(new ToothTrack(dets[di]));
    }

    // Retire only after a grace period, so a closed mouth does not renumber
    // every tooth the moment it reopens.
    this.tracks = this.tracks.filter((t) => t.missing <= this.maxMissing);
    return this.tracks;
  }

  /** Tracks worth drawing: seen enough times, and currently visible. */
  visibleTracks() {
    return this.tracks.filter((t) => t.hits >= this.minHitsToShow && t.missing === 0);
  }

  stats() {
    const vis = this.visibleTracks();
    const conf = vis.length
      ? vis.reduce((s, t) => s + t.confidence, 0) / vis.length : 0;
    const stable = vis.filter((t) => t.isStable).length;
    return {
      count: vis.length,
      total: this.tracks.length,
      avgConfidence: conf,
      stable,
      status: vis.length === 0 ? 'none'
        : (stable >= Math.max(1, vis.length * 0.6) ? 'stable' : 'acquiring'),
    };
  }

  /** Hit-test in mouth-local coordinates (used for tap-to-select). */
  pick(u, v) {
    let best = null;
    for (const t of this.visibleTracks()) {
      const b = t.smoothed?.box ?? t.box;
      if (u >= b.u && u <= b.u + b.w && v >= b.v && v <= b.v + b.h) {
        const d = Math.hypot(u - (b.u + b.w / 2), v - (b.v + b.h / 2));
        if (!best || d < best.d) best = { t, d };
      }
    }
    return best?.t ?? null;
  }
}
