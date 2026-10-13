/** Binary-mask extraction only. No model predictions enter this module. */
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
    return { mask, regions: null, strokes: null };
  }
  for (let i = 0; i < mask.length; i++) {
    const j = i * 4, r = marked[j], g = marked[j + 1], b = marked[j + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b), h = rgbHue(r, g, b);
    const sat = max ? (max - min) / max : 0;
    const distance = Math.abs(h - hue);
    const selected = color === 'custom' ? Math.min(distance, 360 - distance) <= tolerance
      : (h > 160 && h < 220) || (h > 260 && h < 330) || (h > 44 && h < 80);
    const changed = Math.max(Math.abs(r - clean[j]), Math.abs(g - clean[j + 1]), Math.abs(b - clean[j + 2]));
    mask[i] = marked[j + 3] >= 128 && selected && sat > 0.22 && max > 65 && changed >= difference ? 1 : 0;
  }
  const closed = morph(morph(mask, W, H, gap, true), W, H, gap, false);
  const visited = new Uint8Array(mask.length), fill = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  let regions = 0;
  for (let start = 0; start < mask.length; start++) {
    if (closed[start] || visited[start]) continue;
    let head = 0, tail = 1, border = false;
    queue[0] = start; visited[start] = 1;
    while (head < tail) {
      const i = queue[head++], x = i % W, y = Math.floor(i / W);
      if (!x || x === W - 1 || !y || y === H - 1) border = true;
      const push = n => { if (!closed[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } };
      if (x) push(i - 1);
      if (x + 1 < W) push(i + 1);
      if (y) push(i - W);
      if (y + 1 < H) push(i + W);
    }
    if (!border && tail >= minArea) {
      regions++;
      for (let k = 0; k < tail; k++) fill[queue[k]] = 1;
    }
  }
  // Extend one pixel into thick strokes; this is an approximate boundary.
  return { mask: morph(fill, W, H, 1, true), regions, strokes: mask };
}

export function resizeBinaryMask(mask, width, height, outWidth, outHeight) {
  const out = new Uint8Array(outWidth * outHeight);
  const sx = 1 / (outWidth / width), sy = 1 / (outHeight / height);
  for (let y = 0; y < outHeight; y++) for (let x = 0; x < outWidth; x++) {
    out[y * outWidth + x] = mask[Math.min(height - 1, Math.floor(y * sy)) * width + Math.min(width - 1, Math.floor(x * sx))] > 0 ? 1 : 0;
  }
  return out;
}
