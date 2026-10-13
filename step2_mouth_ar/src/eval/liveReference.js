// A photo template transferred to a new mouth is a reference, not frame-specific GT.
export function maskBounds(mask, width, height) {
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (mask[y * width + x] > 0) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y);
      x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
  }
  return x1 < 0 ? null : [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

export function alignReferenceMask(source, sourceWidth, sourceHeight, aperture, width, height,
  { scaleX = 1, scaleY = 1, offsetX = 0, offsetY = 0, rotation = 0, flip = false } = {}) {
  if (source.length !== sourceWidth * sourceHeight || aperture.length !== width * height) {
    throw new Error('Reference/aperture dimensions do not match');
  }
  if (!(scaleX > 0 && scaleY > 0)) throw new Error('Alignment scale must be positive');
  const from = maskBounds(source, sourceWidth, sourceHeight);
  const to = maskBounds(aperture, width, height);
  const output = new Uint8Array(width * height);
  if (!from || !to) return output;
  const [sx, sy, sw, sh] = from, [tx, ty, tw, th] = to;
  const cx = tx + tw / 2 + offsetX * width, cy = ty + th / 2 + offsetY * height;
  const angle = rotation * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
  // Inverse mapping at pixel centers; nearest-neighbor keeps binary labels.
  // Alignment depends on lip aperture and manual controls, never predicted teeth.
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
    let u = (c * dx + s * dy) / (tw * scaleX) + 0.5;
    const v = (-s * dx + c * dy) / (th * scaleY) + 0.5;
    if (flip) u = 1 - u;
    const px = Math.floor(sx + u * sw), py = Math.floor(sy + v * sh);
    if (px >= sx && px < sx + sw && py >= sy && py < sy + sh) output[y * width + x] = source[py * sourceWidth + px] > 0 ? 1 : 0;
  }
  return output;
}

export function scoreReferenceOverlap(truth, prediction) {
  if (truth.length !== prediction.length) throw new Error('Mask sizes differ');
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < truth.length; i++) {
    if (truth[i] > 0) { if (prediction[i] > 0) tp++; else fn++; }
    else if (prediction[i] > 0) fp++; else tn++;
  }
  const div = (a, b) => b ? a / b : null;
  return { tp, fp, fn, tn, iou: div(tp, tp + fp + fn), dice: div(2 * tp, 2 * tp + fp + fn),
    precision: div(tp, tp + fp), recall: div(tp, tp + fn) };
}
