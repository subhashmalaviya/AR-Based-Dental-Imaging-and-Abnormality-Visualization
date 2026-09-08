/**
 * ToothSegmenter.js — classical-CV tooth segmentation inside the mouth ROI.
 *
 * This is a real measurement of real pixels from the live camera. It is not a
 * neural network, and the UI says so; see ToothDetector.js for why no learned
 * model is used here.
 *
 * The algorithm was developed and measured against real footage
 * (mouthtestvideo.mp4) with tools/prototype_tooth_seg.py before being ported
 * here. On that clip, over 258 open-mouth frames it finds a mean of 8.2 teeth
 * (median 8), with >=4 teeth on 93.8% of frames and zero detections on 0%.
 *
 * Pipeline
 * --------
 *  1. **whiteness = V * (1 - S/255)**. Inside an open mouth every competing
 *     surface is strongly coloured — lips and gums are pink, the tongue is red,
 *     the cavity is near-black — so enamel is the only bright *neutral* surface.
 *     A plain brightness threshold fails here: a lit lower lip and the tongue
 *     are both bright. Combining value with desaturation separates them.
 *
 *  2. **Adaptive threshold** at a high percentile of the aperture's *own*
 *     whiteness histogram, so it tracks lighting instead of being a constant.
 *     Otsu was tried first and sat far too low (it kept 57% of the aperture and
 *     swallowed the tongue), because the distribution is not cleanly bimodal.
 *     An absolute floor stops a mouth with no visible teeth from having its
 *     brightest pixels promoted into a "detection" by the percentile alone.
 *
 *  3. **Arch extraction.** Per ROI column, keep only the run of candidate
 *     pixels nearest the aperture's top edge and the run nearest its bottom
 *     edge. Teeth line the aperture; the tongue floats in the middle. This
 *     rejects the tongue *structurally* rather than by threshold tuning, which
 *     is what finally fixed it.
 *
 *  4. **Interdental split.** Crowns touch, so connected components merge a
 *     whole arch into one blob. The gaps between crowns are dark vertical lines
 *     — minima in the per-column mean whiteness — and cutting there recovers
 *     individual teeth. This only works because MouthROI rectified the mouth,
 *     making the gaps vertical regardless of head roll.
 */
import { ToothDetector, registerDetector } from './ToothDetector.js';

export class ToothSegmenter extends ToothDetector {
  constructor(opts = {}) {
    super();
    this.whitenessPercentile = opts.whitenessPercentile ?? 62;
    this.whitenessFloor = opts.whitenessFloor ?? 45;
    // Relative floor, as a fraction of the brightest neutral surface present.
    // The percentile alone is not enough: an aperture is mostly dark cavity, so
    // when a big red tongue is visible the percentile can land *inside* the
    // tongue's brightness band and promote it to a "lower arch". Enamel is the
    // brightest neutral surface in a mouth, so anything far darker than the
    // brightest one is not enamel.
    this.whitenessRelative = opts.whitenessRelative ?? 0.52;
    // Interdental sensitivity. Swept on real footage: 0.88 -> 6.3 teeth/frame,
    // 0.94 -> 8.2, 0.97 -> 9.4 but with visible over-segmentation.
    this.splitRatio = opts.splitRatio ?? 0.94;
    this.archEdgeFrac = opts.archEdgeFrac ?? 0.42;
    this.minAreaPx = opts.minAreaPx ?? 18;
  }

  get name() { return 'Classical CV (whiteness + arch split)'; }
  get isLearnedModel() { return false; }

  setParams(p) { Object.assign(this, p); }

  detect(image, aperture, roi) {
    if (!image || !aperture || !roi?.bounds) return [];
    const W = roi.width, H = roi.height;
    const n = W * H;

    let inside = 0;
    for (let i = 0; i < n; i++) if (aperture[i]) inside++;
    if (inside < 80) return [];

    // ---- 1. whiteness ------------------------------------------------
    const white = new Float32Array(n);
    const data = image.data;
    for (let i = 0; i < n; i++) {
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const s = max === 0 ? 0 : (max - min) / max;   // HSV saturation, 0..1
      white[i] = max * (1 - s);
    }

    // ---- 2. adaptive threshold over the aperture ---------------------
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) {
      if (aperture[i]) hist[Math.min(255, white[i] | 0)]++;
    }
    const target = inside * (this.whitenessPercentile / 100);
    let acc = 0, thr = 0;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (acc >= target) { thr = v; break; }
    }
    // p99 rather than the max, so a single specular highlight cannot drag the
    // relative floor up and suppress every real tooth.
    let acc99 = 0, p99 = 0;
    const t99 = inside * 0.99;
    for (let v = 0; v < 256; v++) {
      acc99 += hist[v];
      if (acc99 >= t99) { p99 = v; break; }
    }
    thr = Math.max(thr, this.whitenessFloor, this.whitenessRelative * p99);

    const cand = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (aperture[i] && white[i] >= thr) cand[i] = 1;
    }
    open3(cand, W, H);
    close3(cand, W, H);

    // ---- 3. upper / lower arches -------------------------------------
    const upper = new Uint8Array(n);
    const lower = new Uint8Array(n);
    for (let x = 0; x < W; x++) {
      let apTop = -1, apBot = -1;
      for (let y = 0; y < H; y++) {
        if (aperture[y * W + x]) { if (apTop < 0) apTop = y; apBot = y; }
      }
      if (apTop < 0) continue;
      const apH = Math.max(apBot - apTop, 1);

      let firstStart = -1, firstEnd = -1, lastStart = -1, lastEnd = -1;
      let runStart = -1;
      for (let y = 0; y <= H; y++) {
        const on = y < H && cand[y * W + x] === 1;
        if (on && runStart < 0) runStart = y;
        if (!on && runStart >= 0) {
          if (firstStart < 0) { firstStart = runStart; firstEnd = y - 1; }
          lastStart = runStart; lastEnd = y - 1;
          runStart = -1;
        }
      }
      if (firstStart < 0) continue;

      if ((firstStart - apTop) / apH < this.archEdgeFrac) {
        for (let y = firstStart; y <= firstEnd; y++) upper[y * W + x] = 1;
      }
      if ((apBot - lastEnd) / apH < this.archEdgeFrac) {
        for (let y = lastStart; y <= lastEnd; y++) lower[y * W + x] = 1;
      }
    }

    // ---- 4. split each arch into teeth --------------------------------
    const teeth = [
      ...this._splitArch(upper, white, W, H, roi, 'upper'),
      ...this._splitArch(lower, white, W, H, roi, 'lower'),
    ];

    // Keep the intermediate stages so the debug view can show exactly what
    // the segmenter saw. This is what makes a weak detection diagnosable on
    // a phone: you can see whether the whiteness threshold found any enamel
    // at all, or whether the mouth was simply too dark.
    this.lastDebug = {
      width: W, height: H,
      aperturePx: inside,
      threshold: thr,
      candidate: cand,          // after threshold + morphology
      upper, lower,             // per-arch masks actually split into teeth
      candidatePx: cand.reduce((s, v) => s + v, 0),
      archPx: upper.reduce((s, v) => s + v, 0) + lower.reduce((s, v) => s + v, 0),
      toothCount: teeth.length,
    };

    return teeth;
  }

  _splitArch(arch, white, W, H, roi, archName) {
    const cols = new Float32Array(W);
    const bright = new Float32Array(W);
    let total = 0, xLo = -1, xHi = -1;
    for (let x = 0; x < W; x++) {
      let c = 0, sum = 0;
      for (let y = 0; y < H; y++) {
        if (arch[y * W + x]) { c++; sum += white[y * W + x]; }
      }
      cols[x] = c;
      bright[x] = c ? sum / c : 0;
      total += c;
      if (c > 0) { if (xLo < 0) xLo = x; xHi = x; }
    }
    if (total < 25 || xLo < 0 || xHi - xLo < 10) return [];
    const span = xHi - xLo;

    smooth3(bright, xLo, xHi);
    let maxCols = 0, maxBright = 1e-6;
    for (let x = xLo; x <= xHi; x++) {
      if (cols[x] > maxCols) maxCols = cols[x];
      if (bright[x] > maxBright) maxBright = bright[x];
    }
    // Low score => likely an interdental gap. Both "dark" and "thin" are
    // evidence; brightness dominates because it is the stronger signal.
    const score = new Float32Array(W);
    for (let x = xLo; x <= xHi; x++) {
      score[x] = 0.65 * (bright[x] / maxBright) + 0.35 * (cols[x] / Math.max(maxCols, 1));
    }

    const win = Math.max(2, Math.round(span * 0.05));
    const cuts = [xLo];
    for (let x = xLo + win; x <= xHi - win; x++) {
      let mn = Infinity, mx = -Infinity;
      for (let k = -win; k <= win; k++) {
        const s = score[x + k];
        if (s < mn) mn = s;
        if (s > mx) mx = s;
      }
      if (score[x] <= mn + 1e-6 && score[x] < this.splitRatio * mx
          && (x - cuts[cuts.length - 1]) >= win) {
        cuts.push(x);
      }
    }
    cuts.push(xHi + 1);

    const out = [];
    const minW = Math.max(3, span * 0.05);
    for (let i = 0; i + 1 < cuts.length; i++) {
      const a = cuts[i], b = cuts[i + 1];
      if (b - a < minW) continue;

      let area = 0, sx = 0, sy = 0, y0 = H, y1 = -1, wSum = 0;
      for (let x = a; x < b; x++) {
        for (let y = 0; y < H; y++) {
          if (!arch[y * W + x]) continue;
          area++; sx += x; sy += y; wSum += white[y * W + x];
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
      if (area < this.minAreaPx) continue;

      const cx = sx / area, cy = sy / area;
      const contour = traceSegment(arch, W, H, a, b);
      if (contour.length < 4) continue;

      // Confidence blends how enamel-like the region is with how well formed it
      // is. Deliberately not called a probability -- it is a heuristic quality
      // score from a classical method, not a learned likelihood.
      const meanWhite = wSum / area;
      const fill = area / Math.max((b - a) * (y1 - y0 + 1), 1);
      const aspect = (b - a) / Math.max(y1 - y0 + 1, 1);
      const aspectScore = Math.exp(-Math.pow(Math.log(Math.max(aspect, 0.05) / 0.85), 2) / 0.9);
      const confidence = Math.max(0, Math.min(1,
        0.5 * Math.min(1, meanWhite / 170) + 0.3 * Math.min(1, fill / 0.8) + 0.2 * aspectScore));

      const c = roi.roiToLocal(cx, cy);
      const p0 = roi.roiToLocal(a, y0);
      const p1 = roi.roiToLocal(b, y1 + 1);
      out.push({
        center: { u: c.u, v: c.v },
        box: { u: p0.u, v: p0.v, w: p1.u - p0.u, h: p1.v - p0.v },
        contour: contour.map(([px, py]) => {
          const l = roi.roiToLocal(px, py);
          return { u: l.u, v: l.v };
        }),
        area: Math.abs((p1.u - p0.u) * (p1.v - p0.v)) * (area / Math.max((b - a) * (y1 - y0 + 1), 1)),
        confidence,
        arch: archName,
      });
    }
    return out;
  }
}

// --------------------------------------------------------------- helpers
function smooth3(arr, lo, hi) {
  const copy = arr.slice(lo, hi + 1);
  for (let i = 1; i < copy.length - 1; i++) {
    arr[lo + i] = (copy[i - 1] + copy[i] + copy[i + 1]) / 3;
  }
}

function open3(m, W, H) { erode3(m, W, H); dilate3(m, W, H); }
function close3(m, W, H) { dilate3(m, W, H); erode3(m, W, H); }

function erode3(m, W, H) {
  const src = m.slice();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let keep = 1;
      for (let dy = -1; dy <= 1 && keep; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= H || xx < 0 || xx >= W || !src[yy * W + xx]) { keep = 0; break; }
        }
      }
      m[y * W + x] = keep;
    }
  }
}

function dilate3(m, W, H) {
  const src = m.slice();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && !on; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W && src[yy * W + xx]) { on = 1; break; }
        }
      }
      m[y * W + x] = on;
    }
  }
}

/**
 * Outline of one tooth segment as a closed polygon.
 *
 * For each column in [a,b) take the topmost and bottommost set pixel, then walk
 * the top edge left-to-right and the bottom edge back. The arch runs are
 * vertically contiguous by construction (see arch extraction), so this
 * reproduces the crown outline without a full border-following pass.
 */
function traceSegment(mask, W, H, a, b) {
  const top = [], bottom = [];
  for (let x = a; x < b; x++) {
    let y0 = -1, y1 = -1;
    for (let y = 0; y < H; y++) {
      if (mask[y * W + x]) { if (y0 < 0) y0 = y; y1 = y; }
    }
    if (y0 >= 0) { top.push([x, y0]); bottom.push([x, y1 + 1]); }
  }
  if (!top.length) return [];
  return [...top, ...bottom.reverse()];
}

registerDetector('classical', () => new ToothSegmenter(), {
  label: 'Classical CV (default)',
  learned: false,
  note: 'Real pixel measurement from the live camera. Not a neural network.',
});
