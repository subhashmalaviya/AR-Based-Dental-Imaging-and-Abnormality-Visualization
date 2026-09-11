/**
 * ToothTracker.js — stable per-tooth identity across frames.
 *
 * ===========================================================================
 * DESIGN (v2)
 * ===========================================================================
 * The coordinate frame does most of the work. Detections arrive in
 * mouth-local coordinates (1.0 = mouth width), so head translation, scale and
 * roll are already removed. v2 goes one step further and matches in
 * **jaw-relative** coordinates: an upper tooth's vertical position is measured
 * from the upper inner lip, a lower tooth's from the lower inner lip. Opening
 * the mouth moves the whole lower arch down in mouth-local space — the single
 * largest source of motion left — but barely moves it relative to the lower
 * lip, so this removes it before association instead of fighting it.
 *
 * Association, per frame:
 *   1. Duplicate suppression (NMS) on the raw detections: a detection that
 *      overlaps a more confident one of the same jaw (IoU > 0.5, or >80% of
 *      its area contained in it) is the same tooth reported twice.
 *   2. ByteTrack-style two-stage matching with the **Hungarian** algorithm:
 *        stage 1  confident detections  vs  every live track
 *        stage 2  weak detections       vs  confirmed tracks left unmatched
 *      A weak detection may keep an existing tooth alive through a dim frame,
 *      but can never create a new ID — which is where spurious IDs came from.
 *      Hungarian (optimal) rather than greedy: greedy is exactly what swaps
 *      the IDs of two adjacent crowns that both overlap one track.
 *   3. Cost = (1 − IoU) + 0.5 · centroid distance / tooth width, gated so
 *      implausible pairs are never forced together. Lost tracks keep
 *      participating (with a gate that widens slightly while they are missing),
 *      so a tooth that re-appears gets its old ID back.
 *   4. Track lifecycle: tentative → confirmed after `minHitsToShow` hits; a
 *      tentative track that misses is dropped immediately (SORT); a confirmed
 *      one survives `maxMissing` frames of occlusion (a closed mouth).
 *   5. Duplicate-track removal: two confirmed tracks that have converged onto
 *      the same tooth are merged, keeping the older ID.
 *
 * Why no Kalman filter / optical flow: in jaw-relative mouth-local space a
 * tooth's true velocity is ~0, so a motion model would only add lag and
 * tuning; the one-euro smoother downstream handles measurement jitter.
 * Optical flow would re-solve motion the anchor already removed.
 *
 * Stability is *measured*, not asserted: the mean Jaccard overlap of the set
 * of visible IDs between consecutive frames over the last 30 frames.
 * ===========================================================================
 */
import { FORBIDDEN, assign } from './math/hungarian.js';

let nextId = 1;

export function iou(a, b) {
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

/** Fraction of the smaller box that lies inside the larger one. */
export function containment(a, b) {
  const x0 = Math.max(a.u, b.u);
  const y0 = Math.max(a.v, b.v);
  const x1 = Math.min(a.u + a.w, b.u + b.w);
  const y1 = Math.min(a.v + a.h, b.v + b.h);
  const iw = x1 - x0, ih = y1 - y0;
  if (iw <= 0 || ih <= 0) return 0;
  const small = Math.min(a.w * a.h, b.w * b.h);
  return small > 0 ? (iw * ih) / small : 0;
}

const shiftBox = (b, dv) => ({ u: b.u, v: b.v - dv, w: b.w, h: b.h });

export class ToothTrack {
  constructor(det, ref) {
    this.id = nextId++;
    this.age = 1;          // frames since creation
    this.hits = 1;         // frames matched
    this.missing = 0;      // consecutive frames unmatched
    this.confirmed = false;
    this.smoothed = null;  // filled by TrackingSmoother
    this._take(det, ref);
  }

  _take(det, ref) {
    this.center = { ...det.center };
    this.box = { ...det.box };
    this.contour = det.contour;
    this.mask = det.mask ?? null;
    this.confidence = det.confidence;
    this.arch = det.arch;
    this.visibility = det.visibility ?? null;
    // jaw-relative copies, used only for association
    this.rbox = shiftBox(det.box, ref);
    this.rcenter = { u: det.center.u, v: det.center.v - ref };
  }

  get isStable() { return this.hits >= 5 && this.missing === 0; }

  get status() {
    if (this.missing > 0) return 'occluded';
    if (this.hits >= 5) return 'stable';
    return 'acquiring';
  }
}

export class ToothTracker {
  constructor({
    iouThreshold = 0.05,     // minimum IoU for a pair to be admissible...
    maxCentroidDist = 0.8,   // ...or centroid within this many tooth widths
    maxMissing = 18,         // frames a confirmed track survives unmatched
    minHitsToShow = 2,       // hits before a track is confirmed and drawn
    highConf = 0.45,         // stage-1 detections (may create tracks)
    lowConf = 0.15,          // stage-2 detections (may only extend tracks)
    nmsIoU = 0.5,
    nmsContain = 0.8,
    mergeIoU = 0.55,
  } = {}) {
    Object.assign(this, {
      iouThreshold, maxCentroidDist, maxMissing, minHitsToShow,
      highConf, lowConf, nmsIoU, nmsContain, mergeIoU,
    });
    this.tracks = [];
    this._visHistory = [];
    this.lastFrameStats = { suppressed: 0, merged: 0, created: 0, lowRescued: 0 };
  }

  reset() {
    this.tracks = [];
    this._visHistory = [];
    nextId = 1;
  }

  setParams(p) { Object.assign(this, p); }

  /** Drop detections that are the same tooth reported twice. */
  nms(dets) {
    const order = dets.map((d, i) => i).sort((a, b) => dets[b].confidence - dets[a].confidence);
    const keep = [];
    for (const i of order) {
      const d = dets[i];
      const dup = keep.some((k) => {
        const e = dets[k];
        if (e.arch !== d.arch) return false;
        return iou(e.box, d.box) > this.nmsIoU || containment(e.box, d.box) > this.nmsContain;
      });
      if (!dup) keep.push(i);
    }
    return keep.sort((a, b) => a - b).map((i) => dets[i]);
  }

  _cost(t, d, dRel) {
    if (t.arch !== d.arch) return FORBIDDEN;
    const ov = iou(t.rbox, dRel.box);
    const width = Math.max(t.box.w, d.box.w, 1e-3);
    const dist = Math.hypot(dRel.center.u - t.rcenter.u, dRel.center.v - t.rcenter.v) / width;
    const gate = this.maxCentroidDist * (1 + 0.15 * Math.min(t.missing, 6));
    if (ov < this.iouThreshold && dist > gate) return FORBIDDEN;
    return (1 - ov) + 0.5 * Math.min(dist, 3);
  }

  _match(trackIdx, detIdx, dets, rel) {
    if (!trackIdx.length || !detIdx.length) return [];
    const cost = trackIdx.map((ti) => detIdx.map((di) => this._cost(this.tracks[ti], dets[di], rel[di])));
    return assign(cost).map(([r, c]) => [trackIdx[r], detIdx[c]]);
  }

  /**
   * @param {Array} detections mouth-local detections for this frame
   * @param {object} [ctx]
   * @param {{upper:number, lower:number}} [ctx.jawRef] mouth-local v of the
   *   upper / lower inner lip; enables jaw-relative matching
   * @returns {ToothTrack[]} live tracks
   */
  update(detections, { jawRef = null } = {}) {
    const raw = detections ?? [];
    const dets = this.nms(raw);
    const ref = (arch) => (jawRef ? (arch === 'lower' ? jawRef.lower : jawRef.upper) ?? 0 : 0);
    const rel = dets.map((d) => ({
      box: shiftBox(d.box, ref(d.arch)),
      center: { u: d.center.u, v: d.center.v - ref(d.arch) },
    }));

    const high = [], low = [];
    dets.forEach((d, i) => {
      if (d.confidence >= this.highConf) high.push(i);
      else if (d.confidence >= this.lowConf) low.push(i);
    });

    // ---- stage 1: confident detections vs all tracks --------------------
    const allT = this.tracks.map((_, i) => i);
    const m1 = this._match(allT, high, dets, rel);
    const usedT = new Set(m1.map(([t]) => t));
    const usedD = new Set(m1.map(([, d]) => d));

    // ---- stage 2: weak detections vs confirmed tracks still unmatched ---
    const leftT = allT.filter((t) => !usedT.has(t) && this.tracks[t].confirmed);
    const m2 = this._match(leftT, low, dets, rel);
    m2.forEach(([t, d]) => { usedT.add(t); usedD.add(d); });

    for (const [ti, di] of [...m1, ...m2]) {
      const t = this.tracks[ti];
      t._take(dets[di], ref(dets[di].arch));
      t.age += 1;
      t.hits += 1;
      t.missing = 0;
      if (t.hits >= this.minHitsToShow) t.confirmed = true;
    }

    const survivors = [];
    this.tracks.forEach((t, ti) => {
      if (usedT.has(ti)) { survivors.push(t); return; }
      t.age += 1;
      t.missing += 1;
      // SORT: a tentative track that misses was probably noise.
      if (!t.confirmed) return;
      if (t.missing <= this.maxMissing) survivors.push(t);
    });

    let created = 0;
    for (const di of high) {
      if (usedD.has(di)) continue;
      const t = new ToothTrack(dets[di], ref(dets[di].arch));
      if (this.minHitsToShow <= 1) t.confirmed = true;
      survivors.push(t);
      created++;
    }
    this.tracks = survivors;

    const merged = this._mergeDuplicates();
    this._recordVisibility();
    this.lastFrameStats = {
      suppressed: raw.length - dets.length, merged, created, lowRescued: m2.length,
    };
    return this.tracks;
  }

  /** Two confirmed tracks sitting on the same tooth: keep the older one. */
  _mergeDuplicates() {
    let merged = 0;
    const vis = this.tracks.filter((t) => t.confirmed && t.missing === 0);
    const dead = new Set();
    for (let i = 0; i < vis.length; i++) {
      for (let j = i + 1; j < vis.length; j++) {
        const a = vis[i], b = vis[j];
        if (dead.has(a) || dead.has(b) || a.arch !== b.arch) continue;
        if (iou(a.box, b.box) > this.mergeIoU || containment(a.box, b.box) > 0.85) {
          const loser = a.hits >= b.hits ? b : a;
          dead.add(loser);
          merged++;
        }
      }
    }
    if (dead.size) this.tracks = this.tracks.filter((t) => !dead.has(t));
    return merged;
  }

  _recordVisibility() {
    const ids = new Set(this.visibleTracks().map((t) => t.id));
    this._visHistory.push(ids);
    while (this._visHistory.length > 31) this._visHistory.shift();
  }

  /**
   * Mean Jaccard of visible-ID sets between consecutive frames (last 30),
   * over frame pairs where teeth were visible in both. Teeth appearing or
   * disappearing because the mouth opened or closed is occlusion, not an ID
   * failure, and is deliberately not counted against stability.
   */
  stability() {
    const h = this._visHistory;
    let sum = 0, n = 0;
    for (let i = 1; i < h.length; i++) {
      const a = h[i - 1], b = h[i];
      if (!a.size || !b.size) continue;
      let inter = 0;
      for (const id of a) if (b.has(id)) inter++;
      sum += inter / (a.size + b.size - inter);
      n++;
    }
    return n ? sum / n : null;
  }

  /** Tracks worth drawing: confirmed and currently visible. */
  visibleTracks() {
    return this.tracks.filter((t) => t.confirmed && t.missing === 0);
  }

  stats() {
    const vis = this.visibleTracks();
    const conf = vis.length
      ? vis.reduce((s, t) => s + t.confidence, 0) / vis.length : 0;
    const stable = vis.filter((t) => t.isStable).length;
    const stab = this.stability();
    const upper = vis.filter((t) => t.arch === 'upper').length;
    return {
      count: vis.length,
      upper,
      lower: vis.length - upper,
      total: this.tracks.length,
      avgConfidence: conf,
      stable,
      stability: stab,
      stabilityLabel: stab == null ? '—'
        : stab >= 0.85 ? 'Stable' : stab >= 0.6 ? 'Moderate' : 'Unstable',
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
