/**
 * toothDecode.js — turns the learned model's dense maps into tooth instances.
 *
 * The network (tools/train/train_tooth_model.py) predicts three maps over the
 * rectified mouth ROI, each a probability in [0,1]:
 *
 *   teeth     P(pixel is enamel of any tooth)             — semantic mask
 *   center    peaked at each tooth's centroid              — one peak per tooth
 *   boundary  high on the thin border between two teeth   — where to cut
 *
 * This is bottom-up instance segmentation (in the spirit of CenterNet seeds +
 * watershed): every centre peak seeds one tooth, and each enamel pixel joins
 * the seed it can reach most cheaply, where crossing a predicted boundary is
 * expensive. Two touching crowns therefore split along the learned
 * interdental line, not along a brightness dip as the classical method does.
 *
 * Enamel that no seed reaches (a tooth whose centre peak was too weak) still
 * becomes an instance if it is large enough, so a missing peak costs a
 * confidence penalty rather than a missed tooth.
 *
 * Pure functions, no DOM: shared by the browser detector, the Node evaluation
 * harness and the unit tests.
 */

const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** Local maxima of `map` above `thr`, restricted to `fg`, in a (2r+1)^2 window. */
export function findPeaks(map, W, H, fg, thr, r) {
  const peaks = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const v = map[i];
      if (v < thr || !fg[i]) continue;
      let isMax = true;
      for (let dy = -r; dy <= r && isMax; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W || (dx === 0 && dy === 0)) continue;
          const u = map[yy * W + xx];
          // strict on one side of the scan order so plateaus yield one peak
          if (u > v || (u === v && yy * W + xx < i)) { isMax = false; break; }
        }
      }
      if (isMax) peaks.push({ i, x, y, score: v });
    }
  }
  return peaks;
}

function dilate(mask, W, H, r) {
  if (r <= 0) return mask;
  const tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let on = 0;
      for (let k = -r; k <= r && !on; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < W && mask[y * W + xx]) on = 1;
      }
      tmp[y * W + x] = on;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let on = 0;
      for (let k = -r; k <= r && !on; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < H && tmp[yy * W + x]) on = 1;
      }
      out[y * W + x] = on;
    }
  }
  return out;
}

/**
 * Outer boundary of one labelled region (Moore-neighbour tracing), as pixel
 * corner-ish points in ROI pixel units (+0.5 = pixel centre).
 */
export function traceContour(labels, W, H, lab, start) {
  const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const at = (x, y) => x >= 0 && y >= 0 && x < W && y < H && labels[y * W + x] === lab;
  const sx = start % W, sy = (start / W) | 0;
  const pts = [[sx + 0.5, sy + 0.5]];
  let x = sx, y = sy, dir = 7;       // came from the left/top, search clockwise
  const maxSteps = 4 * W * H;
  for (let step = 0; step < maxSteps; step++) {
    let found = false;
    for (let k = 0; k < 8; k++) {
      const d = (dir + 6 + k) % 8;     // start just left of the incoming direction
      const nx = x + DIRS[d][0], ny = y + DIRS[d][1];
      if (at(nx, ny)) { x = nx; y = ny; dir = d; found = true; break; }
    }
    if (!found) break;                 // isolated pixel
    if (x === sx && y === sy) break;
    pts.push([x + 0.5, y + 0.5]);
  }
  return pts;
}

/** Keep at most `max` points, evenly spaced along the outline. */
export function decimate(pts, max) {
  if (pts.length <= max) return pts;
  const out = [];
  const step = pts.length / max;
  for (let k = 0; k < max; k++) out.push(pts[Math.floor(k * step)]);
  return out;
}

/**
 * @param {{teeth:Float32Array, center:Float32Array, boundary:Float32Array}} maps
 * @param {number} W @param {number} H
 * @param {Uint8Array|null} aperture  0/255 lip aperture mask at the same size
 * @param {object} [o] thresholds
 * @returns {{instances:Array, labels:Int32Array}}
 */
export function decodeToothMaps(maps, W, H, aperture = null, o = {}) {
  const semThr = o.semThr ?? 0.5;
  const ctrThr = o.ctrThr ?? 0.25;
  const peakR = o.peakRadius ?? 3;
  const bndWeight = o.boundaryWeight ?? 10;
  const minArea = o.minArea ?? Math.max(6, Math.round(W * H * 0.0010));
  const apDilate = o.apertureDilate ?? Math.max(2, Math.round(H * 0.04));
  const { teeth: sem, center: ctr, boundary: bnd } = maps;
  const n = W * H;

  // The aperture is only a sanity bound here: the model has seen lips and
  // knows they are not teeth. Dilate so crowns touching the lip are not cut.
  let inside = null;
  if (aperture) {
    const ap = new Uint8Array(n);
    for (let i = 0; i < n; i++) ap[i] = aperture[i] ? 1 : 0;
    inside = dilate(ap, W, H, apDilate);
  }
  const fg = new Uint8Array(n);
  for (let i = 0; i < n; i++) fg[i] = sem[i] >= semThr && (!inside || inside[i]) ? 1 : 0;

  // ---- seeds ----------------------------------------------------------
  let peaks = findPeaks(ctr, W, H, fg, ctrThr, peakR);
  // Boundary-core seeding (opt-in): split the enamel mask along the predicted
  // interdental borders; every core region that contains no centre peak gets
  // one seed at its own strongest centre response. In dim / low-resolution
  // frames the centre peaks are often weak while the borders are still clear —
  // without this, a whole row of teeth becomes one un-seeded region.
  if (o.coreSeeds === true) peaks = addCoreSeeds(peaks, fg, bnd, ctr, W, H, o);

  // ---- seeded growth, Dial's algorithm (integer bucket queue) ----------
  const labels = new Int32Array(n).fill(-1);
  const dist = new Int32Array(n).fill(0x7fffffff);
  const buckets = [];
  const push = (c, i) => { (buckets[c] ??= []).push(i); };
  peaks.forEach((p, k) => { labels[p.i] = k; dist[p.i] = 0; push(0, p.i); });
  for (let c = 0; c < buckets.length; c++) {
    const b = buckets[c];
    if (!b) continue;
    for (let q = 0; q < b.length; q++) {
      const i = b[q];
      if (dist[i] !== c) continue;
      const x = i % W, y = (i / W) | 0;
      for (const [dx, dy] of N4) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (!fg[j]) continue;
        const nc = c + 1 + Math.round(bndWeight * bnd[j]);
        if (nc < dist[j]) { dist[j] = nc; labels[j] = labels[i]; push(nc, j); }
      }
    }
  }

  // ---- orphan enamel: connected components no seed reached -------------
  let nextLab = peaks.length;
  const orphanOf = new Map();
  for (let s = 0; s < n; s++) {
    if (!fg[s] || labels[s] !== -1) continue;
    const lab = nextLab++;
    const stack = [s];
    labels[s] = lab;
    let best = s;
    while (stack.length) {
      const i = stack.pop();
      if (ctr[i] > ctr[best]) best = i;
      const x = i % W, y = (i / W) | 0;
      for (const [dx, dy] of N4) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (fg[j] && labels[j] === -1) { labels[j] = lab; stack.push(j); }
      }
    }
    orphanOf.set(lab, best);
  }

  // ---- split merged teeth (hierarchical re-seeding) -----------------------
  // A region far wider than the typical tooth in this crop is usually two or
  // more crowns the network could not separate at the global threshold (the
  // thin upper band of a wide-open mouth, dim side teeth). Look again inside
  // it with a lower centre threshold and a smaller peak window; if that finds
  // several peaks, re-grow the region from them. Uses only the model's own
  // evidence — no tooth is invented where no centre response exists.
  // Opt-in: on validation it removed some merges but added more duplicates.
  if (o.splitWide === true) nextLab = splitWide(labels, nextLab, peaks.length, ctr, bnd, fg, W, H, o, bndWeight);

  // ---- per-instance statistics ------------------------------------------
  const L = nextLab;
  const area = new Int32Array(L), sx = new Float64Array(L), sy = new Float64Array(L);
  const semSum = new Float64Array(L);
  const x0 = new Int32Array(L).fill(W), y0 = new Int32Array(L).fill(H);
  const x1 = new Int32Array(L).fill(-1), y1 = new Int32Array(L).fill(-1);
  const first = new Int32Array(L).fill(-1);
  for (let i = 0; i < n; i++) {
    const l = labels[i];
    if (l < 0) continue;
    const x = i % W, y = (i / W) | 0;
    area[l]++; sx[l] += x; sy[l] += y; semSum[l] += sem[i];
    if (x < x0[l]) x0[l] = x; if (x > x1[l]) x1[l] = x;
    if (y < y0[l]) y0[l] = y; if (y > y1[l]) y1[l] = y;
    if (first[l] < 0) first[l] = i;
  }

  const instances = [];
  for (let l = 0; l < L; l++) {
    if (area[l] < minArea) continue;
    const seeded = l < peaks.length;
    const fromCore = seeded && !!peaks[l].core;
    const peak = seeded ? peaks[l].score : ctr[orphanOf.get(l)];
    const meanSem = semSum[l] / area[l];
    // Confidence: how enamel-like the region is and how clearly the network
    // saw a tooth centre in it. An orphan (no peak) is penalised.
    // A core seed is backed by the predicted borders instead of a centre
    // peak: scored on the mask alone, without the orphan penalty.
    const confidence = Math.max(0, Math.min(1, fromCore
      ? 0.85 * meanSem
      : (0.55 * meanSem + 0.45 * Math.min(1, peak / 0.6)) * (seeded ? 1 : 0.7)));
    instances.push({
      label: l,
      area: area[l],
      cx: sx[l] / area[l] + 0.5,
      cy: sy[l] / area[l] + 0.5,
      bbox: { x0: x0[l], y0: y0[l], x1: x1[l] + 1, y1: y1[l] + 1 },
      meanSem,
      peak,
      seeded,
      confidence,
      touchesEdge: x0[l] === 0 || y0[l] === 0 || x1[l] === W - 1 || y1[l] === H - 1,
      contour: decimate(traceContour(labels, W, H, l, first[l]), o.maxContourPoints ?? 28),
    });
  }
  return { instances, labels };
}

function addCoreSeeds(peaks, fg, bnd, ctr, W, H, o) {
  const bThr = o.coreBoundaryThr ?? 0.4;
  const minCore = o.minCoreArea ?? Math.max(8, Math.round(W * H * 0.0015));
  const n = W * H;
  const core = new Int32Array(n).fill(-1);
  const hasPeak = new Set();
  const peakAt = new Set(peaks.map((p) => p.i));
  const out = peaks.slice();
  let c = 0;
  for (let s = 0; s < n; s++) {
    if (!fg[s] || bnd[s] >= bThr || core[s] !== -1) continue;
    const stack = [s];
    core[s] = c;
    let area = 0, best = s, seeded = false;
    while (stack.length) {
      const i = stack.pop();
      area++;
      if (peakAt.has(i)) seeded = true;
      if (ctr[i] > ctr[best]) best = i;
      const x = i % W, y = (i / W) | 0;
      for (const [dx, dy] of N4) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (fg[j] && bnd[j] < bThr && core[j] === -1) { core[j] = c; stack.push(j); }
      }
    }
    if (!seeded && area >= minCore) {
      out.push({ i: best, x: best % W, y: (best / W) | 0, score: ctr[best], core: true });
    }
    c++;
  }
  return out;
}

function splitWide(labels, L, nSeeded, ctr, bnd, fg, W, H, o, bndWeight) {
  const factor = o.splitFactor ?? 1.7;
  const reThr = o.resplitCtrThr ?? 0.08;
  const reR = o.resplitPeakRadius ?? 2;
  const n = W * H;
  const x0 = new Int32Array(L).fill(W), x1 = new Int32Array(L).fill(-1), area = new Int32Array(L);
  for (let i = 0; i < n; i++) {
    const l = labels[i];
    if (l < 0) continue;
    const x = i % W;
    area[l]++;
    if (x < x0[l]) x0[l] = x;
    if (x > x1[l]) x1[l] = x;
  }
  const widths = [];
  for (let l = 0; l < L; l++) if (area[l] > 0) widths.push(x1[l] - x0[l] + 1);
  if (widths.length < 2) return L;
  widths.sort((a, b) => a - b);
  const typical = widths[widths.length >> 1];
  let next = L;
  for (let l = 0; l < L; l++) {
    if (!area[l] || (x1[l] - x0[l] + 1) < factor * typical) continue;
    const inside = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (labels[i] === l) inside[i] = 1;
    const sub = findPeaks(ctr, W, H, inside, reThr, reR)
      // keep peaks at least ~half a typical tooth apart
      .sort((a, b) => b.score - a.score)
      .filter((p, k, arr) => arr.slice(0, k).every((q) => Math.abs(q.x - p.x) >= 0.5 * typical));
    if (sub.length < 2) continue;
    const newLab = sub.map((_, k) => (k === 0 ? l : next++));
    const dist = new Int32Array(n).fill(0x7fffffff);
    const buckets = [];
    const push = (c, i) => { (buckets[c] ??= []).push(i); };
    for (let i = 0; i < n; i++) if (inside[i]) labels[i] = -2;
    sub.forEach((p, k) => { labels[p.i] = newLab[k]; dist[p.i] = 0; push(0, p.i); });
    for (let c = 0; c < buckets.length; c++) {
      const b = buckets[c];
      if (!b) continue;
      for (let q = 0; q < b.length; q++) {
        const i = b[q];
        if (dist[i] !== c) continue;
        const x = i % W, y = (i / W) | 0;
        for (const [dx, dy] of N4) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          const j = yy * W + xx;
          if (!inside[j]) continue;
          const nc = c + 1 + Math.round(bndWeight * bnd[j]);
          if (nc < dist[j]) { dist[j] = nc; labels[j] = labels[i]; push(nc, j); }
        }
      }
    }
    for (let i = 0; i < n; i++) if (labels[i] === -2) labels[i] = l;   // unreachable crumbs
  }
  return next;
}

/**
 * Upper vs lower jaw, from geometry (the model is not trained on jaw labels).
 *
 * Each instance's vertical position is measured relative to the aperture's
 * own midline at its column, which removes the curve of the lips. Teeth then
 * form one or two horizontal bands; the largest gap in that 1-D distribution
 * separates the upper arch from the lower. If there is no clear gap (only
 * one arch visible), the side of the midline decides.
 *
 * @returns {Array<'upper'|'lower'>}
 */
export function assignJaws(instances, W, H, aperture) {
  if (!instances.length) return [];
  const top = new Float32Array(W).fill(-1), bot = new Float32Array(W).fill(-1);
  if (aperture) {
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) {
        if (aperture[y * W + x]) { if (top[x] < 0) top[x] = y; bot[x] = y; }
      }
    }
  }
  const rel = instances.map((t) => {
    const x = Math.min(W - 1, Math.max(0, Math.round(t.cx - 0.5)));
    let a = top[x], b = bot[x];
    if (a < 0) { a = 0; b = H - 1; }
    const mid = (a + b) / 2, half = Math.max((b - a) / 2, 1);
    return (t.cy - mid) / half;       // -1 top edge, 0 midline, +1 bottom edge
  });
  const order = rel.map((r, i) => i).sort((i, j) => rel[i] - rel[j]);
  let bestGap = 0, split = -1;
  for (let k = 0; k + 1 < order.length; k++) {
    const g = rel[order[k + 1]] - rel[order[k]];
    if (g > bestGap) { bestGap = g; split = k; }
  }
  const out = new Array(instances.length);
  if (bestGap >= 0.25 && rel[order[split]] < 0.6 && rel[order[split + 1]] > -0.6) {
    order.forEach((i, k) => { out[i] = k <= split ? 'upper' : 'lower'; });
  } else {
    rel.forEach((r, i) => { out[i] = r <= 0 ? 'upper' : 'lower'; });
  }
  return out;
}
