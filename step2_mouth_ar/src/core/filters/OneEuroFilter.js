/**
 * OneEuroFilter.js — adaptive low-pass filter for noisy real-time signals.
 *
 * Casiez, Roussel & Vogel, "1e Filter: A Simple Speed-based Low-pass Filter for
 * Noisy Input in Interactive Systems" (CHI 2012).
 *
 * Why this rather than a plain EMA: a fixed EMA forces a single trade-off
 * between jitter and lag. Smooth enough to kill landmark jitter while the head
 * is still, and the overlay visibly drags behind fast head motion. The 1-euro
 * filter adapts its cutoff to the signal's own speed -- heavy smoothing when
 * slow (kills jitter), light smoothing when fast (kills lag) -- which is
 * exactly the requirement "reduce jitter ... but do not lag".
 *
 *   minCutoff : cutoff at zero speed. Lower = steadier when still, more lag.
 *   beta      : how fast the cutoff opens up with speed. Higher = less lag.
 *   dCutoff   : cutoff for the derivative estimate itself.
 */

class LowPass {
  constructor() { this.y = null; this.s = null; }
  filter(x, alpha) {
    this.s = this.y === null ? x : alpha * x + (1 - alpha) * this.s;
    this.y = x;
    return this.s;
  }
  get hasLastRawValue() { return this.y !== null; }
  reset() { this.y = null; this.s = null; }
}

const alphaFor = (cutoff, dt) => {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
};

export class OneEuroFilter {
  constructor({ minCutoff = 1.0, beta = 0.0, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPass();
    this.dxFilter = new LowPass();
    this.lastTime = null;
  }

  setParams({ minCutoff, beta, dCutoff }) {
    if (minCutoff !== undefined) this.minCutoff = minCutoff;
    if (beta !== undefined) this.beta = beta;
    if (dCutoff !== undefined) this.dCutoff = dCutoff;
  }

  reset() {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTime = null;
  }

  /** @param {number} x raw sample @param {number} tSec timestamp in seconds */
  filter(x, tSec) {
    if (!Number.isFinite(x)) return this.xFilter.s ?? x;

    let dt = 1 / 60;
    if (this.lastTime !== null && tSec > this.lastTime) {
      dt = tSec - this.lastTime;
    }
    // Guard against pathological dt from tab-switches / dropped frames, which
    // would otherwise make alpha ~1 and let a whole frame of jitter through.
    dt = Math.min(Math.max(dt, 1 / 240), 0.2);
    this.lastTime = tSec;

    const dx = this.xFilter.hasLastRawValue ? (x - this.xFilter.y) / dt : 0;
    const edx = this.dxFilter.filter(dx, alphaFor(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(x, alphaFor(cutoff, dt));
  }
}

/** Independent 1-euro filters over a fixed-length vector. */
export class OneEuroVectorFilter {
  constructor(size, params) {
    this.filters = Array.from({ length: size }, () => new OneEuroFilter(params));
  }
  setParams(p) { for (const f of this.filters) f.setParams(p); }
  reset() { for (const f of this.filters) f.reset(); }
  filter(values, tSec) { return values.map((v, i) => this.filters[i].filter(v, tSec)); }
}
