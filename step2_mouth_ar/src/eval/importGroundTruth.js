import { parsePatientTruth } from './batchEvaluation.js';
import { decodeLabels, instanceStats } from './toothInstances.js';
import { rasterizeVector } from './vectorBoundary.js';

/** Convert patient contour JSON or saved editor JSON into editable labels.
 * Multi-ring teeth keep a single tooth ID. Later teeth own overlapping pixels;
 * affected earlier curves stay as pixel masks rather than disagreeing vectors.
 */
export function importGroundTruth(doc, width, height, imageName) {
  const truth = parsePatientTruth(doc, width, height, imageName);
  const [, , w, h] = truth.crop;
  if (doc.schema === 'tooth-labels/1') {
    const labels = decodeLabels(doc.labels, w * h);
    const count = instanceStats(labels, w).length;
    if (count > 256) throw new Error('Too many tooth labels');
    const vectors = structuredClone(doc.vectorContours ?? []);
    return { ...truth, labels, vectors, pixelOnlyTeeth: count - vectors.length };
  }
  const labels = new Int32Array(w * h), ringToTooth = [0], editableRings = new Set(), ringCounts = new Map();
  for (const c of truth.contours) ringCounts.set(c.id, (ringCounts.get(c.id) ?? 0) + 1);
  for (const [index, c] of truth.contours.entries()) {
    const ringId = index + 1;
    ringToTooth[ringId] = c.id;
    const touched = rasterizeVector(labels, w, h, { ...c, id: ringId });
    for (const id of touched) editableRings.delete(id);
    if (ringCounts.get(c.id) === 1) editableRings.add(ringId);
  }
  for (let i = 0; i < labels.length; i++) labels[i] = ringToTooth[labels[i]];
  const present = new Set(instanceStats(labels, w).map(s => s.id));
  const expected = new Set(truth.contours.map(c => c.id));
  if (present.size !== expected.size) throw new Error('A tooth has no editable pixels (too small or completely covered by another tooth). Check the JSON contours.');
  const vectors = truth.contours.filter((c, index) => editableRings.has(index + 1) && present.has(c.id));
  return { ...truth, labels, vectors: structuredClone(vectors), pixelOnlyTeeth: present.size - vectors.length };
}
