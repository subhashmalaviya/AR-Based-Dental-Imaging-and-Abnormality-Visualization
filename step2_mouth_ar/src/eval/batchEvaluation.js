import { decodeLabels } from './toothInstances.js';
import { traceLabelContours, validateVectorContours, vectorBoundaryDistance } from './vectorBoundary.js';

export const IMAGE_EXTENSION = /\.(png|jpe?g|webp|bmp|gif|dng|heic|heif|avif|tiff?)$/i;

/** Pair by full relative directory + stem, never across patients. */
export function pairPatientFiles(files) {
  const groups = new Map();
  for (const file of files) {
    const path = file.path.replaceAll('\\', '/');
    if (!IMAGE_EXTENSION.test(path) && !/\.json$/i.test(path)) continue;
    const key = path.replace(/\.[^.\/]+$/, '').toLowerCase();
    if (!groups.has(key)) groups.set(key, { images: [], annotations: [] });
    groups.get(key)[/\.json$/i.test(path) ? 'annotations' : 'images'].push({ ...file, path });
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, group]) => {
    const image = group.images[0], annotation = group.annotations[0], path = image?.path ?? annotation.path;
    const issue = group.images.length > 1 || group.annotations.length > 1 ? 'Ambiguous matching files'
      : !image ? 'Missing clean image' : !annotation ? 'Missing ground-truth JSON' : null;
    return { image, annotation, patient: path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '.', issue };
  });
}

/** Preserve the supplied contour coordinates; never infer a crop from GT. */
export function parsePatientTruth(doc, width, height, imageName) {
  if (doc?.schema === 'tooth-labels/1') {
    const c = doc.crop;
    if (doc.cleanImage !== imageName || !Array.isArray(c) || c.length !== 4 || !c.every(Number.isInteger)
      || c[0] < 0 || c[1] < 0 || c[2] < 1 || c[3] < 1 || c[0] + c[2] > width || c[1] + c[3] > height
      || doc.width !== c[2] || doc.height !== c[3]) throw new Error('Saved labels do not match this image and crop');
    const labels = decodeLabels(doc.labels, c[2] * c[3]);
    const vectors = validateVectorContours(doc.vectorContours ?? [], c[2], c[3], labels);
    const vectorIds = new Set(vectors.map(v => v.id));
    return { crop: c, contours: [...traceLabelContours(labels, c[2], c[3]).filter(v => !vectorIds.has(v.id)), ...vectors], patientId: '' };
  }
  if (!Array.isArray(doc?.teeth) || doc.teeth.length > 256) throw new Error('Expected patient JSON with a teeth array, or tooth-labels/1');
  if (doc.annotation_count != null && doc.annotation_count !== doc.teeth.length) throw new Error('annotation_count differs from teeth array length');
  for (const [key, actual] of [['imageWidth', width], ['imageHeight', height], ['image_width', width], ['image_height', height]])
    if (doc[key] != null && doc[key] !== actual) throw new Error(`${key} does not match the clean image`);
  const contours = []; let totalPoints = 0;
  for (const [index, tooth] of doc.teeth.entries()) {
    const rings = Array.isArray(tooth.contours) && tooth.contours.length ? tooth.contours : [tooth.contour];
    if (rings.length > 512) throw new Error('Too many contours for one tooth');
    for (const points of rings) {
      if (!Array.isArray(points) || points.length < 3 || points.length > 20000) throw new Error(`Tooth ${index + 1} has an invalid contour`);
      totalPoints += points.length;
      if (totalPoints > 100000) throw new Error('Too many contour points');
      for (const p of points) if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)
        || p[0] < 0 || p[1] < 0 || p[0] > width || p[1] > height) throw new Error(`Tooth ${index + 1} has points outside the image`);
      const area2 = points.reduce((s, p, i) => { const q = points[(i + 1) % points.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0);
      if (Math.abs(area2) < 1e-8) throw new Error(`Tooth ${index + 1} has a zero-area contour`);
      contours.push({ id: index + 1, smooth: false, points });
    }
  }
  return { crop: [0, 0, width, height], contours, patientId: String(doc.patient_id ?? ''), caseName: String(doc.case_name ?? '') };
}

export function scoreBatchPrediction(truth, prediction, width, height, options = {}) {
  const contours = traceLabelContours(prediction, width, height);
  const distance = vectorBoundaryDistance(truth, contours, width, height, options);
  return { boundary: distance, error: distance.normalizedAssd, status: distance.status, gtTeeth: new Set(truth.map(c => c.id)).size,
    predictedTeeth: new Set(contours.map(c => c.id)).size };
}

export function summarizeBatch(rows) {
  const valid = rows.filter(r => r.status === 'ok' && Number.isFinite(r.error));
  return { total: rows.length, scored: valid.length, unavailable: rows.length - valid.length,
    mean: valid.length ? valid.reduce((sum, r) => sum + r.error, 0) / valid.length : null };
}
