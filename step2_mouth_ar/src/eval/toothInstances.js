import { hungarian } from '../core/math/hungarian.js';

export const binaryFromLabels = labels => Uint8Array.from(labels, n => n > 0 ? 1 : 0);

export function instanceStats(labels, width) {
  const stats = new Map();
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i]; if (id <= 0) continue;
    const s = stats.get(id) ?? { id, area: 0, x: 0, y: 0 };
    s.area++; s.x += i % width; s.y += Math.floor(i / width); stats.set(id, s);
  }
  return [...stats.values()].sort((a, b) => a.id - b.id).map(s => ({ ...s, x: s.x / s.area, y: s.y / s.area }));
}

export function connectedLabels(mask, width, height, minArea = 1, excludeBorder = false) {
  const seen = new Uint8Array(mask.length), labels = new Int32Array(mask.length);
  const queue = new Int32Array(mask.length);
  let count = 0;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    let head = 0, tail = 1, border = false; queue[0] = start; seen[start] = 1;
    while (head < tail) {
      const i = queue[head++], x = i % width, y = Math.floor(i / width);
      if (!x || x === width - 1 || !y || y === height - 1) border = true;
      const visit = n => { if (mask[n] && !seen[n]) { seen[n] = 1; queue[tail++] = n; } };
      if (x) visit(i - 1); if (x + 1 < width) visit(i + 1);
      if (y) visit(i - width); if (y + 1 < height) visit(i + width);
    }
    if (tail < minArea || (excludeBorder && border)) continue;
    count++;
    for (let k = 0; k < tail; k++) labels[queue[k]] = count;
  }
  return { labels, count };
}

export function resizeInstanceLabels(labels, W, H, w, h) {
  const out = new Int32Array(w * h), sx = 1 / (w / W), sy = 1 / (h / H);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    out[y * w + x] = Math.max(0, labels[Math.min(H - 1, Math.floor(y * sy)) * W + Math.min(W - 1, Math.floor(x * sx))]);
  }
  return out;
}

/** One prediction can explain at most one GT tooth. A merged prediction is not split using GT. */
export function compareInstances(truth, prediction, width) {
  if (truth.length !== prediction.length) throw new Error('Instance mask dimensions differ');
  const gt = instanceStats(truth, width), pred = instanceStats(prediction, width), intersections = new Map();
  for (let i = 0; i < truth.length; i++) if (truth[i] > 0 && prediction[i] > 0) {
    const k = `${truth[i]}:${prediction[i]}`; intersections.set(k, (intersections.get(k) ?? 0) + 1);
  }
  const inter = (g, p) => intersections.get(`${g.id}:${p.id}`) ?? 0;
  const ious = gt.map(g => pred.map(p => inter(g, p) / (g.area + p.area - inter(g, p))));
  // Dummy columns allow unmatched GT. Maximise total IoU, then reject zero overlap.
  const assignment = hungarian(ious.map(row => [...row.map(iou => -iou), ...gt.map(() => 0)]));
  const used = new Set();
  const rows = gt.map((g, i) => {
    const j = assignment[i], p = j >= 0 && j < pred.length && ious[i][j] > 0 ? pred[j] : null;
    if (p) used.add(p.id);
    const tp = p ? inter(g, p) : 0, fp = p ? p.area - tp : 0, fn = g.area - tp;
    const overlaps = pred.filter(q => inter(g, q) / g.area >= 0.1).map(q => q.id);
    const merged = p ? gt.filter(q => inter(q, p) / q.area >= 0.1).map(q => q.id) : [];
    return { groundTruth: g.id, prediction: p?.id ?? null, gtPixels: g.area, predictionPixels: p?.area ?? 0,
      tp, fp, fn, iou: tp / (tp + fp + fn), dice: 2 * tp / (2 * tp + fp + fn),
      precision: p ? tp / p.area : null, recall: tp / g.area,
      status: !p ? 'Missed' : ious[i][j] >= 0.5 ? 'Matched' : 'Low overlap',
      splitAcross: overlaps.length > 1 ? overlaps : [], mergedWith: merged.length > 1 ? merged : [] };
  });
  return { rows, extraPredictions: pred.filter(p => !used.has(p.id)),
    groundTruthCount: gt.length, predictionCount: pred.length, matchedAt50: rows.filter(r => r.iou >= 0.5).length,
    matching: 'One-to-one maximum total IoU; matched threshold 0.5; merge/split notes require 10% GT coverage.' };
}

export function paintLabelStroke(labels, width, height, from, to, radius, id) {
  const dx = to[0] - from[0], dy = to[1] - from[1], steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, radius / 2)));
  for (let step = 0; step <= steps; step++) {
    const cx = from[0] + dx * step / steps, cy = from[1] + dy * step / steps;
    for (let y = Math.max(0, Math.floor(cy - radius)); y <= Math.min(height - 1, Math.ceil(cy + radius)); y++) {
      for (let x = Math.max(0, Math.floor(cx - radius)); x <= Math.min(width - 1, Math.ceil(cx + radius)); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) labels[y * width + x] = id;
      }
    }
  }
}

export function encodeLabels(labels) {
  const runs = [];
  for (let i = 0; i < labels.length;) {
    const id = labels[i]; let end = i + 1;
    while (end < labels.length && labels[end] === id) end++;
    runs.push([id, end - i]); i = end;
  }
  return runs;
}

export function decodeLabels(runs, length) {
  const labels = new Int32Array(length); let offset = 0;
  for (const [id, count] of runs) {
    if (!Number.isInteger(id) || id < 0 || id > 100000 || !Number.isInteger(count) || count < 1 || offset + count > length) throw new Error('Invalid tooth label data');
    labels.fill(id, offset, offset + count); offset += count;
  }
  if (offset !== length) throw new Error('Incomplete tooth label data');
  return labels;
}
