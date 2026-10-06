/**
 * TrackingSmoother.js — per-tooth temporal smoothing.
 *
 * Reuses the Step-2 one-euro filter, for the same reason it was chosen there:
 * a fixed EMA forces one trade-off between jitter and lag, whereas one-euro
 * adapts its cutoff to the signal's own speed — heavy smoothing while the tooth
 * is still, opening up when the head really moves, so the contour does not drag.
 *
 * Each track gets its own filter bank over (centre u,v) and (box w,h), created
 * lazily and destroyed with the track. Filtering happens in **mouth-local**
 * coordinates, so head motion has already been removed by the anchor and the
 * filters only ever see genuine detection noise — which is precisely what we
 * want smoothed, and why this can be gentle enough not to lag.
 *
 * Contours are smoothed by their centre/scale rather than vertex-by-vertex:
 * the split boundaries move a little frame to frame, so a per-vertex filter
 * would fight a changing vertex count. The shape is taken from the newest
 * detection, then re-centred and re-scaled onto the smoothed box.
 */
import { OneEuroFilter } from './filters/OneEuroFilter.js';

export const TOOTH_SMOOTHING_PRESETS = {
  responsive: { minCutoff: 2.2, beta: 0.05, dCutoff: 1.0 },
  balanced: { minCutoff: 1.0, beta: 0.04, dCutoff: 1.0 },
  smooth: { minCutoff: 0.45, beta: 0.012, dCutoff: 1.0 },
};

export class TrackingSmoother {
  constructor(preset = 'balanced') {
    this.params = TOOTH_SMOOTHING_PRESETS[preset] ?? TOOTH_SMOOTHING_PRESETS.balanced;
    this.banks = new Map();   // trackId -> filters
  }

  _sizeParams() {
    return {
      minCutoff: Math.min(this.params.minCutoff * 0.35, 0.35),
      beta: Math.min(this.params.beta * 0.25, 0.008),
      dCutoff: 1.0,
    };
  }

  setPreset(name) {
    this.params = TOOTH_SMOOTHING_PRESETS[name] ?? TOOTH_SMOOTHING_PRESETS.balanced;
    const sp = this._sizeParams();
    for (const bank of this.banks.values()) {
      bank.u.setParams(this.params);
      bank.v.setParams(this.params);
      bank.w.setParams(sp);
      bank.h.setParams(sp);
      bank.offU.setParams({ minCutoff: 0.6, beta: 0.015, dCutoff: 1.0 });
      bank.offV.setParams({ minCutoff: 0.6, beta: 0.015, dCutoff: 1.0 });
      bank.conf.setParams({ ...this.params, minCutoff: 0.8 });
    }
  }

  reset() { this.banks.clear(); }

  _bank(id) {
    let b = this.banks.get(id);
    if (!b) {
      const sp = this._sizeParams();
      b = {
        u: new OneEuroFilter(this.params),
        v: new OneEuroFilter(this.params),
        w: new OneEuroFilter(sp),
        h: new OneEuroFilter(sp),
        offU: new OneEuroFilter({ minCutoff: 0.6, beta: 0.015, dCutoff: 1.0 }),
        offV: new OneEuroFilter({ minCutoff: 0.6, beta: 0.015, dCutoff: 1.0 }),
        conf: new OneEuroFilter({ ...this.params, minCutoff: 0.8 }),
        lastW: null,
        lastH: null,
      };
      this.banks.set(id, b);
    }
    return b;
  }

  /**
   * @param {Array} tracks live ToothTrack list
   * @param {number} tSec monotonic seconds
   */
  apply(tracks, tSec) {
    const alive = new Set();
    for (const t of tracks) {
      alive.add(t.id);
      const b = this._bank(t.id);

      const cu = b.u.filter(t.center.u, tSec);
      const cv = b.v.filter(t.center.v, tSec);

      // Box dimension stabilization: tooth crowns are physically rigid,
      // so dimension changes below mask discretization thresholds are smoothed.
      let inW = Math.max(1e-4, t.box.w);
      let inH = Math.max(1e-4, t.box.h);
      if (b.lastW !== null && b.lastH !== null) {
        if (Math.abs(inW - b.lastW) < 0.003) inW = b.lastW * 0.75 + inW * 0.25;
        if (Math.abs(inH - b.lastH) < 0.003) inH = b.lastH * 0.75 + inH * 0.25;
      }

      const bw = Math.max(1e-4, b.w.filter(inW, tSec));
      const bh = Math.max(1e-4, b.h.filter(inH, tSec));
      const conf = b.conf.filter(t.confidence, tSec);

      b.lastW = bw;
      b.lastH = bh;

      // Accurate box centering: account for centroid vs bounding box center offset
      const rawBoxCenterU = t.box.u + t.box.w / 2;
      const rawBoxCenterV = t.box.v + t.box.h / 2;
      const offU = b.offU.filter(rawBoxCenterU - t.center.u, tSec);
      const offV = b.offV.filter(rawBoxCenterV - t.center.v, tSec);
      const bcu = cu + offU;
      const bcv = cv + offV;

      // Re-fit the newest contour onto the smoothed centre/size, so the shape
      // stays current while its placement stays steady.
      let contour = t.contour;
      if (contour && contour.length) {
        const rawW = Math.max(t.box.w, 1e-4);
        const rawH = Math.max(t.box.h, 1e-4);
        const sx = bw / rawW, sy = bh / rawH;
        contour = contour.map((p) => ({
          u: cu + (p.u - t.center.u) * sx,
          v: cv + (p.v - t.center.v) * sy,
        }));
      }

      t.smoothed = {
        center: { u: cu, v: cv },
        box: { u: bcu - bw / 2, v: bcv - bh / 2, w: bw, h: bh },
        contour,
        confidence: conf,
      };
    }

    // Drop filter banks for retired tracks so the map cannot grow unbounded
    // over a long session.
    for (const id of [...this.banks.keys()]) {
      if (!alive.has(id)) this.banks.delete(id);
    }
    return tracks;
  }
}
