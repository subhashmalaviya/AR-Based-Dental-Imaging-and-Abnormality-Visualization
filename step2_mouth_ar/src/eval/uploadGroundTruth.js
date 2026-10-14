import { connectedLabels, binaryFromLabels } from './toothInstances.js';
/** Preserve each enclosed annotation as a separate instance. No model input here. */
export function rgbHue(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  if (!d) return 0;
  const h = max === r ? (g - b) / d : max === g ? 2 + (b - r) / d : 4 + (r - g) / d;
  return (h * 60 + 360) % 360;
}

// Separable square binary morphology with zero padding outside the image.
function morph(mask, W, H, radius, dilate) {
  if (!radius) return mask.slice();
  const tmp = new Uint8Array(mask.length), out = new Uint8Array(mask.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = dilate ? 0 : 1;
    for (let k = -radius; k <= radius; k++) {
      const p = x + k >= 0 && x + k < W ? mask[y * W + x + k] : 0;
      if (dilate ? p : !p) { v = dilate ? 1 : 0; break; }
    }
    tmp[y * W + x] = v;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = dilate ? 0 : 1;
    for (let k = -radius; k <= radius; k++) {
      const p = y + k >= 0 && y + k < H ? tmp[(y + k) * W + x] : 0;
      if (dilate ? p : !p) { v = dilate ? 1 : 0; break; }
    }
    out[y * W + x] = v;
  }
  return out;
}

export function extractUploadTruth(clean, marked, W, H, {
  format = 'outlines', color = 'preset', hue = 180, tolerance = 25,
  difference = 20, gap = 2, minArea = 12,
} = {}) {
  if (clean.length !== W * H * 4 || marked.length !== clean.length) throw new Error('Image pixel dimensions differ');
  const mask = new Uint8Array(W * H);
  if (format === 'binary') {
    for (let i = 0; i < mask.length; i++) {
      const j = i * 4;
      mask[i] = marked[j + 3] >= 128 && (marked[j] + marked[j + 1] + marked[j + 2]) / 3 >= 128 ? 1 : 0;
    }
    const { labels, count } = connectedLabels(mask, W, H);
    return { mask, labels, regions: count, strokes: null };
  }
  for (let i = 0; i < mask.length; i++) {
    const j = i * 4, r = marked[j], g = marked[j + 1], b = marked[j + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b), h = rgbHue(r, g, b);
    const sat = max ? (max - min) / max : 0;
    const distance = Math.abs(h - hue);
    const selected = color === 'custom' ? Math.min(distance, 360 - distance) <= tolerance
      : (h > 160 && h < 330) || (h > 44 && h < 80);
    const changed = Math.max(Math.abs(r - clean[j]), Math.abs(g - clean[j + 1]), Math.abs(b - clean[j + 2]));
    mask[i] = marked[j + 3] >= 128 && selected && sat > 0.22 && max > 65 && changed >= difference ? 1 : 0;
  }
  const labels = new Int32Array(W * H);
  let regions = 0, repaired = 0;
  // Preserve small-gap instances first. More aggressive repair may add missed
  // loops, but cannot merge or replace already discovered teeth.
  const radii = gap ? [...new Set([gap, gap + 1, gap + 2, gap + 4])] : [0];
  for (const radius of radii) {
    const closed = morph(morph(mask, W, H, radius, true), W, H, radius, false);
    const candidates = connectedLabels(Uint8Array.from(closed, n => n ? 0 : 1), W, H, minArea, true);
    const overlap = Array.from({ length: candidates.count + 1 }, () => new Set());
    const areas = new Uint32Array(candidates.count + 1), parentAreas = new Map();
    for (let i = 0; i < labels.length; i++) {
      if (labels[i]) parentAreas.set(labels[i], (parentAreas.get(labels[i]) ?? 0) + 1);
      const id = candidates.labels[i];
      if (id) { areas[id]++; if (labels[i]) overlap[id].add(labels[i]); }
    }
    const assigned = new Int32Array(candidates.count + 1), children = new Map(), splitParents = new Set();
    for (let id = 1; id <= candidates.count; id++) {
      if (!overlap[id].size) {
        assigned[id] = ++regions; if (radius !== gap) repaired++;
      } else if (overlap[id].size === 1) {
        const parent = [...overlap[id]][0];
        if (areas[id] >= Math.max(minArea, parentAreas.get(parent) * 0.12)) {
          const list = children.get(parent) ?? []; list.push(id); children.set(parent, list);
        }
      }
    }
    // A repaired dividing stroke can split an initially merged pair of boxes.
    // Only accept multiple substantial children retaining most of the interior.
    for (const [parent, ids] of children) {
      if (ids.length < 2 || ids.reduce((sum, id) => sum + areas[id], 0) < parentAreas.get(parent) * 0.6) continue;
      splitParents.add(parent); ids.sort((a, b) => areas[b] - areas[a]);
      ids.forEach((id, index) => { assigned[id] = index ? ++regions : parent; });
      repaired += ids.length - 1;
    }
    for (let i = 0; i < labels.length; i++) if (splitParents.has(labels[i])) labels[i] = 0;
    for (let i = 0; i < labels.length; i++) if (!labels[i] && assigned[candidates.labels[i]]) labels[i] = assigned[candidates.labels[i]];
  }
  // Extend interiors one pixel toward stroke centers while retaining IDs.
  const expanded = labels.slice();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x; if (labels[i]) continue;
    let owner = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (x + dx < 0 || x + dx >= W || y + dy < 0 || y + dy >= H) continue;
      const id = labels[(y + dy) * W + x + dx];
      if (id && (!owner || id < owner)) owner = id;
    }
    expanded[i] = owner;
  }
  return { mask: binaryFromLabels(expanded), labels: expanded, regions, repaired, strokes: mask };
}

export function resizeBinaryMask(mask, width, height, outWidth, outHeight) {
  const out = new Uint8Array(outWidth * outHeight);
  const sx = 1 / (outWidth / width), sy = 1 / (outHeight / height);
  for (let y = 0; y < outHeight; y++) for (let x = 0; x < outWidth; x++) {
    out[y * outWidth + x] = mask[Math.min(height - 1, Math.floor(y * sy)) * width + Math.min(width - 1, Math.floor(x * sx))] > 0 ? 1 : 0;
  }
  return out;
}
