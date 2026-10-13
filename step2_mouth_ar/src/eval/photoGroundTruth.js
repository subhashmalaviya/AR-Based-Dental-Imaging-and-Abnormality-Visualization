// Fixed coordinates for the user's supplied 880x450 photo pair.
// Ground truth is used only for scoring/display, never as detector input.
export const PHOTO_GT = {
  width: 880, height: 450,
  crops: [[105, 52, 280, 341], [495, 55, 312, 340]],
  cards: [[11, 13, 421, 396], [447, 13, 421, 396]],
};

export function expandCropMask(target, small, width, height, rectangle) {
  const [x0, y0, w, h] = rectangle;
  if (small.length !== width * height || target.length !== PHOTO_GT.width * PHOTO_GT.height) {
    throw new Error('Unexpected mask dimensions');
  }
  // Match OpenCV INTER_NEAREST's reciprocal scale and floor convention,
  // including floating-point rounding at exact source-row boundaries.
  const scaleX = 1 / (w / width), scaleY = 1 / (h / height);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      target[(y0 + y) * PHOTO_GT.width + x0 + x] =
        small[Math.floor(y * scaleY) * width + Math.floor(x * scaleX)] > 0 ? 1 : 0;
    }
  }
}

export function scorePhotoMasks(truth, prediction, cards = PHOTO_GT.cards) {
  if (truth.length !== PHOTO_GT.width * PHOTO_GT.height || prediction.length !== truth.length) {
    throw new Error('Photo masks must both be 880x450');
  }
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const [x0, y0, w, h] of cards) {
    for (let y = y0; y < y0 + h; y++) {
      for (let x = x0; x < x0 + w; x++) {
        const i = y * PHOTO_GT.width + x;
        if (truth[i]) { if (prediction[i]) tp++; else fn++; }
        else if (prediction[i]) fp++;
        else tn++;
      }
    }
  }
  const divide = (a, b) => b ? a / b : null;
  return { tp, fp, fn, tn, iou: divide(tp, tp + fp + fn),
    dice: divide(2 * tp, 2 * tp + fp + fn), precision: divide(tp, tp + fp),
    recall: divide(tp, tp + fn) };
}
