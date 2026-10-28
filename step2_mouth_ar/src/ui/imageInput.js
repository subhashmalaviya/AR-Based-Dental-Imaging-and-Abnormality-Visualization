// Decode by file contents: phone exports sometimes keep .DNG on JPEG data.
export const IMAGE_ACCEPT = 'image/*,.dng,.DNG,.heic,.heif,.avif,.tif,.tiff';
export const MAX_IMAGE_PIXELS = 32_000_000;
// An accepted original must also work with "Use whole image". Cropping is optional.
export const MAX_EDIT_PIXELS = MAX_IMAGE_PIXELS;

export function imageMime(bytes, fallback = '') {
  const tag = (a, b) => String.fromCharCode(...bytes.slice(a, b));
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'image/jpeg';
  if (tag(1, 4) === 'PNG' && bytes[0] === 137) return 'image/png';
  if (tag(0, 4) === 'RIFF' && tag(8, 12) === 'WEBP') return 'image/webp';
  if (tag(0, 3) === 'GIF') return 'image/gif';
  if (tag(0, 2) === 'BM') return 'image/bmp';
  if (tag(0, 4) === 'II*\0' || tag(0, 4) === 'MM\0*') return 'image/tiff';
  return fallback;
}

export function checkImageDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
    || width * height > MAX_IMAGE_PIXELS || Math.max(width, height) > 16384)
    throw new Error('Open an image up to 32 megapixels and 16384 pixels per side. Export a smaller copy for larger originals.');
}

export function checkEditCrop(crop) {
  if (!Array.isArray(crop) || crop.length !== 4 || !crop.every(Number.isInteger) || crop[0] < 0 || crop[1] < 0)
    throw new Error('Invalid comparison region.');
  checkImageDimensions(crop[2], crop[3]);
}

export async function readImageFile(file) {
  if (file.size > 80 * 1024 * 1024) throw new Error('Use an image smaller than 80 MB.');
  const type = imageMime(new Uint8Array(await file.slice(0, 32).arrayBuffer()), file.type);
  if (type === 'image/tiff') throw new Error('This is a true RAW DNG/TIFF. Export it as PNG or JPEG first; this browser does not develop RAW sensor data. JPEG files named .DNG are supported directly.');
  const url = URL.createObjectURL(new Blob([file], { type }));
  try {
    const image = new Image(); image.src = url;
    try { await image.decode(); }
    catch { throw new Error(`Cannot decode ${file.name}. For HEIC/HEIF or RAW formats unsupported by this browser, export PNG or JPEG (on iPhone, use Most Compatible).`); }
    checkImageDimensions(image.naturalWidth, image.naturalHeight);
    const c = document.createElement('canvas'); c.width = image.naturalWidth; c.height = image.naturalHeight;
    const ctx = c.getContext('2d');
    if (!ctx) throw new Error('Not enough graphics memory to open this image.');
    ctx.drawImage(image, 0, 0);
    return c;
  } finally { URL.revokeObjectURL(url); }
}

export function drawImagePreview(target, source) {
  const scale = Math.min(1, 1600 / Math.max(source.width, source.height));
  target.width = Math.round(source.width * scale); target.height = Math.round(source.height * scale);
  target.getContext('2d').drawImage(source, 0, 0, target.width, target.height);
  return scale;
}
