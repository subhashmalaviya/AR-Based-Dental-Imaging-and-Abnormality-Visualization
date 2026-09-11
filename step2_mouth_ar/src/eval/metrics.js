/**
 * metrics.js — tooth-detection evaluation against manual ground truth.
 *
 * Nothing in this module invents a number: every metric is computed from
 * annotations a person made on real frames. With no annotations there are no
 * metrics, and the UI says "not evaluated" rather than showing a default.
 *
 * Ground truth may be annotated in either of two ways, per tooth:
 *   - a BOX (and optionally a POLYGON) — enables IoU-based matching and
 *     segmentation IoU;
 *   - a POINT at the crown centre — much faster to annotate, and exactly what
 *     "is every visible tooth detected as its own instance?" needs. A detection
 *     matches a point GT tooth when the point lies inside the detection.
 *
 * What is counted (per frame, then aggregated):
 *   TP / FP / FN            one-to-one optimal (Hungarian) matching
 *   duplicates              extra detections on an already-detected tooth
 *                           (over-segmentation: one tooth split in two)
 *   merges                  one detection covering two or more GT teeth
 *                           (under-segmentation: two teeth in one box)
 *   count error             |#detections - #GT teeth|
 *   jaw accuracy            upper/lower label agreement on matched teeth
 *   mean IoU                box IoU, or mask IoU where both have polygons
 *
 * Coordinates are whatever pixel space the annotations were made in (frame
 * pixels or rectified-ROI pixels); detections must be in the same space.
 */
import { FORBIDDEN, assign } from '../core/math/hungarian.js';

// ------------------------------------------------------------ geometry
export function boxIoU(a, b) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const iw = x1 - x0, ih = y1 - y0;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export function pointInPolygon(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > pt.y) !== (b.y > pt.y)
        && pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

export function pointInBox(pt, b) {
  return pt.x >= b.x && pt.x <= b.x + b.w && pt.y >= b.y && pt.y <= b.y + b.h;
}

/** Does a detection's region contain a point? Polygon if present, else box. */
export function detContains(det, pt) {
  if (det.polygon && det.polygon.length >= 3) return pointInPolygon(pt, det.polygon);
  return pointInBox(pt, det.box);
}

export function polygonBox(poly) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of poly) {
    if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y;
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/**
 * Mask IoU of two polygons, by rasterising both on a shared grid covering
 * their union bounding box. `res` cells along the longer side is plenty for
 * crown-sized shapes (error well under 1% at 96).
 */
export function polygonIoU(p, q, res = 96) {
  const bp = polygonBox(p), bq = polygonBox(q);
  const x0 = Math.min(bp.x, bq.x), y0 = Math.min(bp.y, bq.y);
  const x1 = Math.max(bp.x + bp.w, bq.x + bq.w), y1 = Math.max(bp.y + bp.h, bq.y + bq.h);
  const span = Math.max(x1 - x0, y1 - y0);
  if (!(span > 0)) return 0;
  const step = span / res;
  let inter = 0, uni = 0;
  for (let y = y0 + step / 2; y < y1; y += step) {
    for (let x = x0 + step / 2; x < x1; x += step) {
      const pt = { x, y };
      const a = pointInPolygon(pt, p), b = pointInPolygon(pt, q);
      if (a || b) uni++;
      if (a && b) inter++;
    }
  }
  return uni ? inter / uni : 0;
}

/** Representative point of a GT tooth (explicit point, else region centre). */
export function gtCenter(gt) {
  if (gt.point) return gt.point;
  if (gt.polygon && gt.polygon.length >= 3) {
    let sx = 0, sy = 0;
    for (const p of gt.polygon) { sx += p.x; sy += p.y; }
    return { x: sx / gt.polygon.length, y: sy / gt.polygon.length };
  }
  return { x: gt.box.x + gt.box.w / 2, y: gt.box.y + gt.box.h / 2 };
}

function detCenter(det) {
  return det.center ?? { x: det.box.x + det.box.w / 2, y: det.box.y + det.box.h / 2 };
}

/** Overlap score for a (GT, detection) pair; 0 means "cannot match". */
function pairScore(gt, det, iouThreshold) {
  if (gt.box) {
    // Boxes that do not intersect have zero mask IoU too: skip the (costly)
    // rasterisation for them. Exact, not an approximation.
    const bIoU = boxIoU(gt.box, det.box);
    if (bIoU === 0) return null;
    const iou = (gt.polygon && det.polygon) ? polygonIoU(gt.polygon, det.polygon) : bIoU;
    return iou >= iouThreshold ? { score: iou, iou } : null;
  }
  // Point GT: the detection must contain the point. Prefer the detection whose
  // centre is nearest, normalised by the detection's own size.
  const pt = gtCenter(gt);
  if (!detContains(det, pt)) return null;
  const c = detCenter(det);
  const size = Math.max(det.box.w, det.box.h, 1e-6);
  const d = Math.hypot(c.x - pt.x, c.y - pt.y) / size;
  return { score: 1 / (1 + d), iou: null };
}

// ------------------------------------------------------------ per frame
/**
 * @param {Array} dets  [{box:{x,y,w,h}, polygon?, center?, jaw?, conf?}]
 * @param {Array} gts   [{box?, polygon?, point?, jaw?}]
 * @param {object} [opts]
 * @param {number} [opts.iouThreshold=0.5]  for box / polygon GT
 */
export function evaluateFrame(dets, gts, { iouThreshold = 0.5, ignore = [] } = {}) {
  dets = dets ?? [];
  gts = gts ?? [];
  // Ignore regions (COCO "crowd"-style): areas where the annotator could see
  // teeth but could not tell them apart. A detection centred in one, covering
  // no annotated tooth, is neither a true nor a false positive.
  let ignored = 0;
  if (ignore.length) {
    const keep = dets.filter((d) => {
      const c = detCenter(d);
      const inIgnore = ignore.some((b) => pointInBox(c, b));
      const coversGT = gts.some((g) => detContains(d, gtCenter(g)));
      return !(inIgnore && !coversGT);
    });
    ignored = dets.length - keep.length;
    dets = keep;
  }
  const scores = gts.map((g) => dets.map((d) => pairScore(g, d, iouThreshold)));
  const cost = scores.map((row) => row.map((s) => (s ? 1 - s.score : FORBIDDEN)));
  const pairs = gts.length && dets.length ? assign(cost) : [];

  const gtMatched = new Array(gts.length).fill(-1);
  const detMatched = new Array(dets.length).fill(-1);
  const ious = [];
  let jawCorrect = 0, jawTotal = 0;
  for (const [gi, di] of pairs) {
    gtMatched[gi] = di;
    detMatched[di] = gi;
    const s = scores[gi][di];
    if (s.iou != null) ious.push(s.iou);
    if (gts[gi].jaw && dets[di].jaw) {
      jawTotal++;
      if (gts[gi].jaw === dets[di].jaw) jawCorrect++;
    }
  }

  // Under-segmentation: a detection containing the centres of 2+ GT teeth.
  let merges = 0;
  dets.forEach((d) => {
    const inside = gts.filter((g) => detContains(d, gtCenter(g))).length;
    if (inside >= 2) merges += inside - 1;
  });

  // Over-segmentation: an unmatched detection whose centre falls on a tooth
  // that is already matched — i.e. a second piece of the same tooth.
  let duplicates = 0;
  dets.forEach((d, di) => {
    if (detMatched[di] >= 0) return;
    const c = detCenter(d);
    const onMatched = gts.some((g, gi) => {
      if (gtMatched[gi] < 0) return false;
      if (g.box) return pointInBox(c, g.box) || boxIoU(g.box, d.box) >= 0.3;
      // Point GT: a second piece of the SAME tooth sits on that tooth's crown
      // centre — its own box (grown by 25 %) contains the GT point, and that
      // point is the nearest annotated tooth to it. An adjacent tooth's box
      // does not reach the neighbour's centre, so it stays an ordinary false
      // positive (or an unannotated tooth), not a duplicate.
      const p = gtCenter(g);
      const b = d.box, gx = 0.25 * b.w, gy = 0.25 * b.h;
      const inGrown = p.x >= b.x - gx && p.x <= b.x + b.w + gx && p.y >= b.y - gy && p.y <= b.y + b.h + gy;
      if (!inGrown) return false;
      const dg = Math.hypot(c.x - p.x, c.y - p.y);
      return gts.every((o) => {
        const q = gtCenter(o);
        return Math.hypot(c.x - q.x, c.y - q.y) >= dg;
      });
    });
    if (onMatched) duplicates++;
  });

  const tp = pairs.length;
  return {
    tp,
    fp: dets.length - tp,
    fn: gts.length - tp,
    duplicates,
    merges,
    ious,
    jawCorrect,
    jawTotal,
    countGT: gts.length,
    countDet: dets.length,
    ignored,
    missed: gtMatched.map((d, gi) => (d < 0 ? gi : -1)).filter((gi) => gi >= 0),
    pairs: pairs.map(([gi, di]) => ({ gt: gi, det: di, iou: scores[gi][di].iou })),
  };
}

// ------------------------------------------------------------ aggregate
export function aggregate(frames) {
  const n = frames.length;
  if (!n) return null;
  const sum = (k) => frames.reduce((s, f) => s + f[k], 0);
  const tp = sum('tp'), fp = sum('fp'), fn = sum('fn');
  const ious = frames.flatMap((f) => f.ious);
  const jawT = sum('jawTotal');
  const precision = tp + fp ? tp / (tp + fp) : null;
  const recall = tp + fn ? tp / (tp + fn) : null;
  const f1 = precision != null && recall != null && precision + recall > 0
    ? (2 * precision * recall) / (precision + recall) : null;
  return {
    frames: n,
    gtTeeth: sum('countGT'),
    detections: sum('countDet'),
    tp, fp, fn,
    precision, recall, f1,
    duplicates: sum('duplicates'),
    merges: sum('merges'),
    meanIoU: ious.length ? ious.reduce((s, v) => s + v, 0) / ious.length : null,
    countMAE: frames.reduce((s, f) => s + Math.abs(f.countDet - f.countGT), 0) / n,
    exactCountRate: frames.filter((f) => f.countDet === f.countGT).length / n,
    jawAccuracy: jawT ? sum('jawCorrect') / jawT : null,
  };
}

/** Plain-text table for console/CLI output. */
export function formatSummary(name, s) {
  if (!s) return `${name}: not evaluated (no annotated frames)`;
  const pct = (v) => (v == null ? '  n/a' : `${(v * 100).toFixed(1)}%`);
  return [
    `${name}`,
    `  frames ${s.frames}   GT teeth ${s.gtTeeth}   detections ${s.detections}`,
    `  precision ${pct(s.precision)}   recall ${pct(s.recall)}   F1 ${pct(s.f1)}`,
    `  TP ${s.tp}  FP ${s.fp}  FN(missed) ${s.fn}  duplicates ${s.duplicates}  merges ${s.merges}`,
    `  count MAE ${s.countMAE.toFixed(2)}   exact-count frames ${pct(s.exactCountRate)}`
      + `   mean IoU ${s.meanIoU == null ? 'n/a' : s.meanIoU.toFixed(3)}`
      + `   jaw acc ${pct(s.jawAccuracy)}`,
  ].join('\n');
}
