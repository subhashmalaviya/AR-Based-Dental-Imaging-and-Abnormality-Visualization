/** Enhance a display copy; never modify the original inference/annotation pixels. */
export function enhancePixels(data, grayscale, contrast = 1) {
  const out = data.slice(), c = Math.max(0.5, Math.min(3, Number(contrast) || 1));
  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    for (let k = 0; k < 3; k++) out[i + k] = Math.max(0, Math.min(255, ((grayscale ? gray : data[i + k]) - 127.5) * c + 127.5));
  }
  return out;
}
