import { boundaryOptions } from './boundaryOptions.js';
import kdTreePkg from 'kd-tree-javascript';
const kdTree = kdTreePkg.kdTree || kdTreePkg.default?.kdTree || kdTreePkg;
// Coordinates are native crop pixels. Raster contours follow pixel CELL EDGES.
// Curves are periodic cubic interpolating splines through editable knots.
export function traceLabelContours(labels, w, h) {
  if (labels.length !== w * h) throw new Error('Boundary dimensions differ');
  const byId = new Map();
  const edge = (id, a, b) => {
    let map = byId.get(id); if (!map) byId.set(id, map = new Map());
    const key = a.join(','); if (!map.has(key)) map.set(key, []);
    map.get(key).push([a, b]);
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, id = labels[i]; if (!id) continue;
    if (!y || labels[i-w] !== id) edge(id, [x,y], [x+1,y]);
    if (x === w-1 || labels[i+1] !== id) edge(id, [x+1,y], [x+1,y+1]);
    if (y === h-1 || labels[i+w] !== id) edge(id, [x+1,y+1], [x,y+1]);
    if (!x || labels[i-1] !== id) edge(id, [x,y+1], [x,y]);
  }
  const contours = [];
  for (const [id, map] of byId) while (map.size) {
    const first = map.values().next().value[0][0], points = [];
    let p = first, previous = null;
    do {
      points.push(p);
      const key = p.join(','), edges = map.get(key);
      if (!edges?.length) throw new Error('Open mask contour');
      // At diagonal contacts turn right, keeping separate components separate.
      let index = 0;
      if (edges.length > 1 && previous) {
        const dx = p[0]-previous[0], dy = p[1]-previous[1];
        index = edges.findIndex(([,b]) => dx*(b[1]-p[1])-dy*(b[0]-p[0]) > 0);
        if (index < 0) index = 0;
      }
      const [,next] = edges.splice(index,1)[0]; if (!edges.length) map.delete(key);
      previous = p; p = next;
    } while (p[0] !== first[0] || p[1] !== first[1]);
    // Remove collinear cell corners without shifting any border.
    const reduced = points.filter((p,i) => {
      const a = points[(i+points.length-1)%points.length], b = points[(i+1)%points.length];
      return (p[0]-a[0])*(b[1]-p[1]) !== (p[1]-a[1])*(b[0]-p[0]);
    });
    contours.push({ id, smooth: false, points: reduced });
  }
  return contours;
}

export function cubicSegments(points) {
  const n = points.length;
  return points.map((p,i) => {
    const a=points[(i+n-1)%n], q=points[(i+1)%n], b=points[(i+2)%n];
    // Clamp control points to each adjacent knot's bounding box: no overshoot.
    const clamp=(v,k)=>Math.max(Math.min(p[k],q[k]), Math.min(Math.max(p[k],q[k]),v));
    return [p, [clamp(p[0]+(q[0]-a[0])/6,0),clamp(p[1]+(q[1]-a[1])/6,1)],
      [clamp(q[0]-(b[0]-p[0])/6,0),clamp(q[1]-(b[1]-p[1])/6,1)], q];
  });
}
export function contourPath(c) {
  if (c.points.length < 3) return '';
  const start=`M${c.points[0].join(' ')}`;
  return start + (c.smooth ? cubicSegments(c.points).map(s=>`C${s.slice(1).flat().join(' ')}`).join('')
    : c.points.slice(1).map(p=>`L${p.join(' ')}`).join('')) + 'Z';
}
const dist2 = (p,a,b) => {
  const dx=b[0]-a[0],dy=b[1]-a[1], l=dx*dx+dy*dy;
  const t=l ? Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/l)) : 0;
  return (p[0]-a[0]-t*dx)**2+(p[1]-a[1]-t*dy)**2;
};
export function flattenContour(c, tolerance=0.1) {
  if (!c.smooth) return c.points;
  const points = [c.points[0]], mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
  const split=(a,b,d,e,depth=0)=>{
    if (depth>=16 || Math.max(dist2(b,a,e),dist2(d,a,e))<=tolerance*tolerance) { points.push(e); return; }
    const ab=mid(a,b),bd=mid(b,d),de=mid(d,e),x=mid(ab,bd),y=mid(bd,de),m=mid(x,y);
    split(a,ab,x,m,depth+1);split(m,y,de,e,depth+1);
  };
  for (const s of cubicSegments(c.points)) split(...s);
  points.pop(); return points;
}
export function validateVectorContours(value,w,h,labels=null) {
  if (!Array.isArray(value) || value.length>512) throw new Error('Invalid vector contours');
  let total=0; const seen=new Set();
  const ids=labels ? new Set(labels) : null;
  for (const c of value) {
    if (!Number.isInteger(c.id)||c.id<1||c.id>100000||typeof c.smooth!=='boolean'||!Array.isArray(c.points)
      ||c.points.length<3||c.points.length>20000|| (ids&&!ids.has(c.id))) throw new Error('Invalid vector tooth');
    if(seen.has(c.id)) throw new Error("Only one editable curve per tooth ID is supported");
    seen.add(c.id); total+=c.points.length;
    if(total>100000) throw new Error('Too many vector points');
    for(const p of c.points) if(!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite)
      ||p[0]<0||p[1]<0||p[0]>w||p[1]>h) throw new Error('Vector point outside image');
  }
  if (labels) for (const c of value) {
    const copy=labels.slice(); rasterizeVector(copy,w,h,c);
    if(copy.some((v,i)=>v!==labels[i]))throw new Error('Vector border and saved tooth pixels disagree');
  }
  return structuredClone(value);
}

export function sampleBoundaryPoints(contours, step = 0.5) {
  const points = [];
  for (const c of contours) {
    const poly = flattenContour(c, 0.005);
    if (!poly || poly.length < 2) continue;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len === 0) continue;
      const count = Math.max(1, Math.ceil(len / step));
      const prev = poly[(i - 1 + poly.length) % poly.length];
      for (let k = 0; k < count; k++) {
        const t = (k + 0.5) / count;
        points.push({
          x: a[0] + t * (b[0] - a[0]),
          y: a[1] + t * (b[1] - a[1]),
          segA: a, segB: b,
          segPrevA: prev, segPrevB: a,
          id: c.id
        });
      }
    }
  }
  return points;
}

const pointToSegmentDist2 = (p, a, b) => {
  const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy;
  const t = l ? Math.max(0, Math.min(1, ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / l)) : 0;
  return (p.x - a[0] - t * dx) ** 2 + (p.y - a[1] - t * dy) ** 2;
};

/** Distances to vector segments, retaining interdental boundaries.
 * Normalizes errors by the number of points from teeth boundaries, avoiding dividing by contour length.
 * Computes both average error (mean surface distance) and absolute max error (Hausdorff distance)
 * using standard open-source KD-tree spatial index (kd-tree-javascript).
 * Provides unit-normalized metrics (divided by image/crop diagonal). */
export function vectorBoundaryDistance(truth, prediction, w, h, options = {}) {
  if (!(w > 0 && h > 0 && Number.isFinite(w + h))) throw new Error('Invalid image dimensions');
  const { tolerancePx, requiredCoverage } = boundaryOptions(options);
  const diagonal = Math.hypot(w, h);
  const status = !truth.length && !prediction.length ? 'both empty' : !truth.length ? 'missing ground truth' : !prediction.length ? 'missing prediction' : 'ok';
  const result = {
    status,
    assdPx: null, assdMm: null, normalizedAssd: null,
    averageErrorPx: null, averageErrorMm: null, normalizedAverageError: null,
    hausdorffPx: null, hausdorffMm: null, normalizedHausdorff: null,
    maxErrorPx: null, maxErrorMm: null, absoluteMaxErrorPx: null, normalizedMaxError: null,
    hausdorff95Px: null, normalizedHausdorff95: null,
    normalizationDiagonalPx: diagonal,
    truthBoundaryPoints: 0, predictionBoundaryPoints: 0, totalBoundaryPoints: 0,
    sdfCoverage: status === 'missing prediction' ? 0 : null,
    sdfError: status === 'missing prediction' ? 1 : null,
    sdfPass: status === 'missing prediction' ? false : null,
    tolerancePx, requiredCoverage,
    definition: 'Point-normalized bidirectional Euclidean boundary distance (mean surface error) and absolute max error (Hausdorff distance) using standard KD-tree spatial index across teeth boundary points. Errors are normalized by the number of boundary points from teeth boundaries, avoiding division by contour length. Normalized metrics divide by the image region diagonal. Missing boundaries remain explicitly unavailable.'
  };
  if (status !== 'ok') return result;

  const step = options.samplingStepPx ?? (diagonal <= 50 ? 0.05 : diagonal <= 500 ? 0.25 : 1.0);
  const ptsA = sampleBoundaryPoints(truth, step);
  const ptsB = sampleBoundaryPoints(prediction, step);
  if (!ptsA.length || !ptsB.length) {
    const s = !ptsA.length && !ptsB.length ? 'both empty' : !ptsA.length ? 'missing ground truth' : 'missing prediction';
    return { ...result, status: s, sdfCoverage: s === 'missing prediction' ? 0 : null, sdfError: s === 'missing prediction' ? 1 : null, sdfPass: s === 'missing prediction' ? false : null };
  }

  const treeB = new kdTree(ptsB, (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2, ['x', 'y']);
  const treeA = new kdTree(ptsA, (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2, ['x', 'y']);

  const query = (sourcePts, tree) => {
    let sum = 0, max = 0, within = 0;
    const distances = new Float64Array(sourcePts.length);
    const k = Math.min(3, tree.root ? 3 : 1);
    for (let i = 0; i < sourcePts.length; i++) {
      const p = sourcePts[i];
      const candidates = tree.nearest(p, k);
      let minD2 = Infinity;
      for (const [cand] of candidates) {
        minD2 = Math.min(
          minD2,
          pointToSegmentDist2(p, cand.segA, cand.segB),
          pointToSegmentDist2(p, cand.segPrevA, cand.segPrevB)
        );
      }
      const d = Math.sqrt(minD2 < 1e-12 ? 0 : minD2);
      distances[i] = d;
      sum += d;
      if (d > max) max = d;
      if (d <= tolerancePx + 1e-10) within++;
    }
    return { sum, max, within, count: sourcePts.length, distances };
  };

  const qA = query(ptsA, treeB);
  const qB = query(ptsB, treeA);

  const totalPoints = qA.count + qB.count;
  const avgErrorPx = (qA.sum + qB.sum) / totalPoints;
  const rawHd = Math.max(qA.max, qB.max);
  const maxErrorPx = rawHd < 1e-10 ? 0 : rawHd;
  const coverage = Math.min(1, Math.max(0, (qA.within + qB.within) / totalPoints));
  const normalizedAvg = avgErrorPx / diagonal;
  const normalizedMax = maxErrorPx / diagonal;

  const allD = new Float64Array(totalPoints);
  allD.set(qA.distances, 0);
  allD.set(qB.distances, qA.count);
  allD.sort();
  const hd95 = allD[Math.min(totalPoints - 1, Math.floor(totalPoints * 0.95))];

  const calibration = Number.isFinite(options.pixelsPerMm) && options.pixelsPerMm > 0 ? options.pixelsPerMm : null;

  return {
    ...result,
    assdPx: avgErrorPx,
    averageErrorPx: avgErrorPx,
    normalizedAssd: normalizedAvg,
    normalizedAverageError: normalizedAvg,
    assdMm: calibration ? avgErrorPx / calibration : null,
    averageErrorMm: calibration ? avgErrorPx / calibration : null,

    hausdorffPx: maxErrorPx,
    maxErrorPx,
    absoluteMaxErrorPx: maxErrorPx,
    normalizedHausdorff: normalizedMax,
    normalizedMaxError: normalizedMax,
    hausdorffMm: calibration ? maxErrorPx / calibration : null,
    maxErrorMm: calibration ? maxErrorPx / calibration : null,

    hausdorff95Px: hd95,
    normalizedHausdorff95: hd95 / diagonal,

    truthBoundaryPoints: qA.count,
    predictionBoundaryPoints: qB.count,
    totalBoundaryPoints: totalPoints,

    sdfCoverage: coverage,
    sdfError: 1 - coverage,
    sdfPass: coverage + 1e-10 >= requiredCoverage,

    truthWithinTolerance: Math.min(1, qA.within / qA.count),
    predictionWithinTolerance: Math.min(1, qB.within / qB.count),
    truthToPredictionPx: qA.sum / qA.count,
    predictionToTruthPx: qB.sum / qB.count
  };
}

/** Deterministic even-odd fill at native pixel centers, shared by editor/imports.
 * The curve stays authoritative for distances; this mask only matches instances. */
export function rasterizeVector(labels,w,h,curve) {
  const points=flattenContour(curve),touched=new Set();
  for(let i=0;i<labels.length;i++)if(labels[i]===curve.id)labels[i]=0;
  let low=Infinity,high=-Infinity;for(const p of points){low=Math.min(low,p[1]);high=Math.max(high,p[1]);}
  const ymin=Math.max(0,Math.floor(low)),ymax=Math.min(h-1,Math.ceil(high));
  for(let y=ymin;y<=ymax;y++) {
    const scan=y+.5,xs=[];
    for(let j=0;j<points.length;j++) {
      const a=points[j],b=points[(j+1)%points.length];
      if((a[1]>scan)!==(b[1]>scan))xs.push(a[0]+(scan-a[1])*(b[0]-a[0])/(b[1]-a[1]));
    }
    xs.sort((a,b)=>a-b);
    for(let j=0;j+1<xs.length;j+=2)for(let x=Math.max(0,Math.ceil(xs[j]-.5));x<Math.min(w,Math.ceil(xs[j+1]-.5));x++) {
      const i=y*w+x;touched.add(labels[i]);labels[i]=curve.id;
    }
  }
  return touched;
}
