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

function segments(contours) {
  return contours.flatMap(c=>{const p=flattenContour(c);return p.map((a,i)=>[a,p[(i+1)%p.length]])})
    .filter(([a,b])=>a[0]!==b[0]||a[1]!==b[1]);
}
function tree(lines) {
  const bounds=[Infinity,Infinity,-Infinity,-Infinity];
  for(const s of lines) for(const p of s) {
    bounds[0]=Math.min(bounds[0],p[0]);bounds[1]=Math.min(bounds[1],p[1]);
    bounds[2]=Math.max(bounds[2],p[0]);bounds[3]=Math.max(bounds[3],p[1]);
  }
  if(lines.length<=8)return {bounds,lines};
  const axis=bounds[2]-bounds[0]>bounds[3]-bounds[1]?0:1;
  lines.sort((a,b)=>a[0][axis]+a[1][axis]-b[0][axis]-b[1][axis]);
  const m=Math.floor(lines.length/2);return {bounds,left:tree(lines.slice(0,m)),right:tree(lines.slice(m))};
}
const boxDistance=(p,b)=>Math.max(b[0]-p[0],0,p[0]-b[2])**2+Math.max(b[1]-p[1],0,p[1]-b[3])**2;
function nearest(p,node,best=Infinity) {
  if(boxDistance(p,node.bounds)>best)return best;
  if(node.lines) {for(const [a,b] of node.lines)best=Math.min(best,dist2(p,a,b));return best;}
  let a=node.left,b=node.right;
  if(boxDistance(p,a.bounds)>boxDistance(p,b.bounds)) [a,b]=[b,a];
  return nearest(p,b,nearest(p,a,best));
}
/** Arc-length weighted bidirectional Euclidean ASSD. Curves flatten to 0.1px;
 * midpoint quadrature spacing <=0.5px. Distance is to segments, not raster pixels.
 * Missing surfaces yield null, never a misleading zero. Shared tooth edges count
 * once for each tooth, preserving interdental instance boundaries. */
export function vectorBoundaryDistance(truth,prediction,w,h) {
  if(!(w>0&&h>0&&Number.isFinite(w+h)))throw new Error('Invalid image dimensions');
  const a=segments(truth),b=segments(prediction),diagonal=Math.hypot(w,h);
  const status=!a.length&&!b.length?'both empty':!a.length?'missing ground truth':!b.length?'missing prediction':'ok';
  const result={status,assdPx:null,assdMm:null,normalizedAssd:null,normalizationDiagonalPx:diagonal,
    definition:'arc-length weighted bidirectional Euclidean ASSD; curve flattening 0.1px, sampling <=0.5px; pixel-cell edge prediction contours'};
  if(status!=='ok')return result;
  const integrate=(source,target)=>{
    const index=tree(target.slice());let sum=0,length=0;
    for(const [a,b] of source) {
      const l=Math.hypot(b[0]-a[0],b[1]-a[1]), n=Math.ceil(l/0.5);
      for(let k=0;k<n;k++){const t=(k+0.5)/n;sum+=Math.sqrt(nearest([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])],index))*l/n;}
      length+=l;
    } return {sum,length};
  };
  const ab=integrate(a,b),ba=integrate(b,a);
  return {...result,assdPx:(ab.sum+ba.sum)/(ab.length+ba.length),
    normalizedAssd:(ab.sum+ba.sum)/(ab.length+ba.length)/diagonal,
    truthToPredictionPx:ab.sum/ab.length,predictionToTruthPx:ba.sum/ba.length};
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
