// Exact Euclidean distance transform via separable lower envelopes of parabolas.
// Felzenszwalb & Huttenlocher: https://www.cs.cornell.edu/~dph/papers/dt.pdf
function squaredDistanceTransform(mask, w, h) {
  const out = new Float64Array(w * h), size = Math.max(w, h);
  const f = new Float64Array(size), d = new Float64Array(size);
  const v = new Int32Array(size), z = new Float64Array(size + 1);
  const far = w * w + h * h + 1;
  const transform = n => {
    let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      let s;
      do {
        const p = v[k]; s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * (q - p));
        if (s > z[k]) break;
        k--;
      } while (k >= 0);
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      d[q] = (q - v[k]) ** 2 + f[v[k]];
    }
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = mask[y * w + x] ? 0 : far;
    transform(w); out.set(d.subarray(0, w), y * w);
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = out[y * w + x];
    transform(h);
    for (let y = 0; y < h; y++) out[y * w + x] = d[y];
  }
  return out;
}

/** Boundary pixel centers using 4-neighbors, retaining interdental boundaries. */
export function labelBoundary(labels, w, h) {
  if (!Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1 || labels.length !== w * h) throw new Error('Boundary dimensions differ');
  const boundary = new Uint8Array(labels.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, id = labels[i];
    if (id > 0 && (x === 0 || y === 0 || x === w - 1 || y === h - 1
      || labels[i - 1] !== id || labels[i + 1] !== id || labels[i - w] !== id || labels[i + w] !== id)) boundary[i] = 1;
  }
  return boundary;
}

export function boundaryDistance(truth, prediction, w, h, pixelsPerMm = null, diagonal = Math.hypot(w, h)) {
  const a = labelBoundary(truth, w, h), b = labelBoundary(prediction, w, h);
  const n = a.reduce((s, x) => s + x, 0), m = b.reduce((s, x) => s + x, 0);
  const calibration = Number.isFinite(pixelsPerMm) && pixelsPerMm > 0 ? pixelsPerMm : null;
  const result = { assdPx: null, assdMm: null, normalizedAssd: null, chamferPx: null, chamferMm: null,
    truthToPredictionPx: null, predictionToTruthPx: null,
    truthBoundaryPixels: n, predictionBoundaryPixels: m, pixelsPerMm: calibration,
    normalizationDiagonalPx: diagonal, status: !n && !m ? 'both empty' : !n ? 'missing ground truth' : !m ? 'missing prediction' : 'ok' };
  if (!n || !m) return result;
  const directedSum = (source, target) => {
    const squared = squaredDistanceTransform(target, w, h);
    let sum = 0;
    for (let i = 0; i < source.length; i++) if (source[i]) sum += Math.sqrt(squared[i]);
    return sum;
  };
  const ab = directedSum(a, b), ba = directedSum(b, a);
  result.truthToPredictionPx = ab / n; result.predictionToTruthPx = ba / m;
  // ASSD pools all surface samples; symmetric Chamfer weights directions equally.
  result.assdPx = (ab + ba) / (n + m);
  result.chamferPx = (ab / n + ba / m) / 2;
  result.normalizedAssd = result.assdPx / diagonal;
  result.assdMm = calibration ? result.assdPx / calibration : null;
  result.chamferMm = calibration ? result.chamferPx / calibration : null;
  return result;
}

/** Score existing one-to-one matches using a compact bounding rectangle per pair. */
export function matchedBoundaryDistances(truth, prediction, w, h, rows, pixelsPerMm) {
  const bounds = labels => {
    const map = new Map();
    for (let i = 0; i < labels.length; i++) if (labels[i] > 0) {
      const x = i % w, y = Math.floor(i / w), b = map.get(labels[i]) ?? [x, y, x, y];
      b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y);
      b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y); map.set(labels[i], b);
    }
    return map;
  };
  const gtBounds = bounds(truth), predBounds = bounds(prediction);
  for (const row of rows) {
    if (row.prediction == null) { row.boundary = null; continue; }
    const g = gtBounds.get(row.groundTruth), p = predBounds.get(row.prediction);
    const x0 = Math.min(g[0], p[0]), y0 = Math.min(g[1], p[1]);
    const cw = Math.max(g[2], p[2]) - x0 + 1, ch = Math.max(g[3], p[3]) - y0 + 1;
    const a = new Uint8Array(cw * ch), b = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const i = (y0 + y) * w + x0 + x, j = y * cw + x;
      a[j] = truth[i] === row.groundTruth ? 1 : 0;
      b[j] = prediction[i] === row.prediction ? 1 : 0;
    }
    row.boundary = boundaryDistance(a, b, cw, ch, pixelsPerMm, Math.hypot(w, h));
  }
}
