/**
 * MouthROI.js — dynamic mouth region of interest, rectified into mouth-local space.
 *
 * Reuses the Step-2 mouth landmarks and anchor rather than searching the frame:
 * the ROI is derived from the inner lip ring, so it follows the mouth, updates
 * every frame, scales with face distance and stays aligned under head roll.
 *
 * The ROI is **rectified**: the mouth is resampled into a canonical axis-aligned
 * image via the anchor's basis. That matters for detection, not just tidiness —
 * once head roll is removed, the dental arches are horizontal and interdental
 * gaps are vertical, which is what makes the column-profile split in
 * ToothSegmenter work at all. It is the same trick as Step 1's cup unwrap.
 *
 * Extraction uses a single affine `drawImage` (the mouth-local -> image map is
 * affine under the anchor's weak-perspective model), so the resample is done by
 * the compositor rather than a per-pixel JS loop.
 */
import {
  LIPS_INNER_RING, LOWER_LIP_INNER, UPPER_LIP_INNER,
} from '../landmarks/FaceLandmarkIndices.js';

export const ROI_W = 192;
export const ROI_H = 144;

export class MouthROI {
  constructor({ width = ROI_W, height = ROI_H, padding = 0.16 } = {}) {
    this.width = width;
    this.height = height;
    this.padding = padding;

    // Canvas created lazily on first extract(), so the geometry (bounds,
    // snapshot, aperture) also works headless — e.g. in the Node tests.
    this.canvas = null;
    this.ctx = null;

    this.bounds = null;   // {u0,u1,v0,v1} in mouth-local units
    this.pose = null;
  }

  /**
   * @param {Array} landmarks raw MediaPipe landmarks
   * @param {object} pose smoothed anchor pose (origin/basis/scale)
   * @param {number} frameW @param {number} frameH
   * @returns {{u0,u1,v0,v1}|null}
   */
  computeBounds(landmarks, pose, frameW, frameH) {
    if (!landmarks || !pose) return null;
    const { origin, basis, scale } = pose;
    if (!(scale > 1e-6)) return null;

    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    const local = [];
    let upperV = null, lowerV = null;
    for (const i of LIPS_INNER_RING) {
      const p = landmarks[i];
      const dx = p.x * frameW - origin.x;
      const dy = p.y * frameH - origin.y;
      const dz = (p.z ?? 0) * frameW - origin.z;
      const u = (dx * basis.x.x + dy * basis.x.y + dz * basis.x.z) / scale;
      const v = (dx * basis.y.x + dy * basis.y.y + dz * basis.y.z) / scale;
      local.push({ u, v });
      if (i === UPPER_LIP_INNER) upperV = v;
      if (i === LOWER_LIP_INNER) lowerV = v;
      if (u < u0) u0 = u; if (u > u1) u1 = u;
      if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    if (!Number.isFinite(u0) || u1 - u0 < 1e-4 || v1 - v0 < 1e-4) return null;

    // Pad outward from the ring's own extent, so the window grows with mouth
    // opening instead of clipping the teeth (a fixed window did exactly that).
    const du = (u1 - u0) * this.padding;
    const dv = (v1 - v0) * this.padding;
    this.bounds = { u0: u0 - du, u1: u1 + du, v0: v0 - dv, v1: v1 + dv };
    this.localRing = local;
    this.pose = pose;
    // Inner-lip midpoints: the tracker measures upper teeth from the upper lip
    // and lower teeth from the lower lip, cancelling jaw opening.
    this.jawRef = { upper: upperV ?? v0, lower: lowerV ?? v1 };
    return this.bounds;
  }

  /**
   * Frozen copy of the geometry, for an asynchronous detector: its result must
   * be mapped with the bounds of the frame it was computed from, not the
   * bounds of whatever frame is current when inference finishes.
   */
  snapshot() {
    const b = { ...this.bounds };
    const W = this.width, H = this.height;
    return {
      width: W, height: H, bounds: b,
      localRing: this.localRing?.slice() ?? null,
      jawRef: this.jawRef ? { ...this.jawRef } : null,
      roiToLocal: (px, py) => ({
        u: b.u0 + (px / W) * (b.u1 - b.u0),
        v: b.v0 + (py / H) * (b.v1 - b.v0),
      }),
      localToRoi: (u, v) => ({
        x: ((u - b.u0) / (b.u1 - b.u0)) * W,
        y: ((v - b.v0) / (b.v1 - b.v0)) * H,
      }),
    };
  }

  /** ROI-pixel coords -> mouth-local coords. */
  roiToLocal(px, py) {
    const b = this.bounds;
    if (!b) return null;
    return {
      u: b.u0 + (px / this.width) * (b.u1 - b.u0),
      v: b.v0 + (py / this.height) * (b.v1 - b.v0),
    };
  }

  /** Mouth-local coords -> ROI-pixel coords. */
  localToRoi(u, v) {
    const b = this.bounds;
    if (!b) return null;
    return {
      x: ((u - b.u0) / (b.u1 - b.u0)) * this.width,
      y: ((v - b.v0) / (b.v1 - b.v0)) * this.height,
    };
  }

  /**
   * Resample the mouth region out of `video` into the ROI canvas.
   * Returns ImageData, or null if the geometry is degenerate.
   */
  extract(video) {
    const b = this.bounds;
    const pose = this.pose;
    if (!b || !pose) return null;
    const { origin, basis, scale } = pose;

    // Forward map, ROI pixel -> video pixel (affine):
    //   X = E + A*px + C*py ,  Y = F + B*px + D*py
    const su = (b.u1 - b.u0) / this.width;
    const sv = (b.v1 - b.v0) / this.height;
    const A = scale * su * basis.x.x;
    const B = scale * su * basis.x.y;
    const C = scale * sv * basis.y.x;
    const D = scale * sv * basis.y.y;
    const E = origin.x + scale * (b.u0 * basis.x.x + b.v0 * basis.y.x);
    const F = origin.y + scale * (b.u0 * basis.x.y + b.v0 * basis.y.y);

    // drawImage needs video -> ROI, i.e. the inverse of the above.
    const det = A * D - B * C;
    if (!Number.isFinite(det) || Math.abs(det) < 1e-9) return null;
    const ia = D / det, ib = -B / det, ic = -C / det, id = A / det;
    const ie = (C * F - D * E) / det, iff = (B * E - A * F) / det;

    if (!this.ctx) {
      this.canvas = typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(this.width, this.height)
        : Object.assign(document.createElement('canvas'), { width: this.width, height: this.height });
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.setTransform(ia, ib, ic, id, ie, iff);
    try {
      ctx.drawImage(video, 0, 0);
    } catch {
      return null;                     // video not ready yet
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return ctx.getImageData(0, 0, this.width, this.height);
  }

  /** Binary aperture mask (Uint8Array, 0/255) of the inner lip ring, eroded. */
  apertureMask() {
    if (!this.localRing || !this.bounds) return new Uint8Array(this.width * this.height);
    return buildApertureMask(this.localRing, this.bounds, this.width, this.height);
  }
}

/**
 * Erosion radius that pulls the aperture in off the lips.
 *
 * The inner-ring landmarks sit ON the lip edge, and a lit lower lip is bright
 * enough to be mistaken for enamel, so the polygon is shrunk slightly. This
 * was ported wrongly at first: the Python prototype the segmenter was tuned on
 * used a 7x7 kernel (radius 3 at 144 px), but the JS port treated that number
 * as a *radius* and eroded with 13x13 — shaving ~6 px off both edges, which
 * clips upper incisors that touch the upper lip. Radius ~2.2% of ROI height
 * restores the tuned behaviour; tests/eval measures the difference.
 */
export function apertureErosionRadius(H) {
  return Math.max(1, Math.round(H * 0.022));
}

/**
 * Pure-function aperture mask, usable without a canvas (the Node evaluation
 * harness calls this directly).
 * @param {Array<{u:number,v:number}>} localRing inner lip ring, mouth-local
 * @param {{u0,u1,v0,v1}} bounds ROI bounds, mouth-local
 * @param {number} W @param {number} H ROI size in px
 * @param {number} [radius] erosion radius (defaults to apertureErosionRadius)
 */
export function buildApertureMask(localRing, bounds, W, H, radius = apertureErosionRadius(H)) {
  const mask = new Uint8Array(W * H);
  const toRoi = ({ u, v }) => ({
    x: ((u - bounds.u0) / (bounds.u1 - bounds.u0)) * W,
    y: ((v - bounds.v0) / (bounds.v1 - bounds.v0)) * H,
  });
  const poly = localRing.map(toRoi);

  // even-odd scanline fill
  for (let y = 0; y < H; y++) {
    const yc = y + 0.5;
    const xs = [];
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length];
      if ((p.y <= yc && q.y > yc) || (q.y <= yc && p.y > yc)) {
        xs.push(p.x + ((yc - p.y) / (q.y - p.y)) * (q.x - p.x));
      }
    }
    xs.sort((a, c) => a - c);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const a = Math.max(0, Math.ceil(xs[i] - 0.5));
      const b = Math.min(W - 1, Math.floor(xs[i + 1] - 0.5));
      for (let x = a; x <= b; x++) mask[y * W + x] = 255;
    }
  }
  return radius > 0 ? erode(mask, W, H, radius) : mask;
}

/** Square-kernel erosion via two separable min passes. */
function erode(src, W, H, r) {
  const tmp = new Uint8Array(W * H);
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let m = 255;
      for (let k = -r; k <= r; k++) {
        const xx = x + k;
        if (xx < 0 || xx >= W) { m = 0; break; }
        if (src[y * W + xx] === 0) { m = 0; break; }
      }
      tmp[y * W + x] = m;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let m = 255;
      for (let k = -r; k <= r; k++) {
        const yy = y + k;
        if (yy < 0 || yy >= H) { m = 0; break; }
        if (tmp[yy * W + x] === 0) { m = 0; break; }
      }
      out[y * W + x] = m;
    }
  }
  return out;
}
