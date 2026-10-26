// Fit a closed vector border to a continuous pointer stroke in native pixels.
// Simplification is independent of editor zoom; clicks and tiny drags are ignored.
export function freehandBorder(samples, id, tolerance = 0.6) {
  const points = samples.filter((p, i) => !i || Math.hypot(p[0]-samples[i-1][0], p[1]-samples[i-1][1]) > 0.05);
  if (points.length < 4) return null;
  if (Math.hypot(points[0][0]-points.at(-1)[0], points[0][1]-points.at(-1)[1]) < .5) points.pop();
  // Split the ring away from its start before RDP, avoiding a zero-length baseline.
  let far = 1;
  for (let i=2;i<points.length;i++) if (distance2(points[i],points[0])>distance2(points[far],points[0])) far=i;
  const simplify = input => {
    const keep = new Set([0,input.length-1]), stack = [[0,input.length-1]];
    while (stack.length) {
      const [a,b]=stack.pop(); let best=tolerance*tolerance, at=-1;
      for (let i=a+1;i<b;i++) {
        const d=segmentDistance2(input[i],input[a],input[b]);
        if(d>best){best=d;at=i;}
      }
      if(at>=0){keep.add(at);stack.push([a,at],[at,b]);}
    }
    return [...keep].sort((a,b)=>a-b).map(i=>input[i].slice());
  };
  const knots=[...simplify(points.slice(0,far+1)).slice(0,-1),...simplify([...points.slice(far),points[0]]).slice(0,-1)];
  const area=Math.abs(knots.reduce((sum,p,i)=>{const q=knots[(i+1)%knots.length];return sum+p[0]*q[1]-p[1]*q[0];},0))/2;
  if(knots.length<3 || area<4) return null;
  return {id,smooth:true,points:knots};
}
const distance2=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
function segmentDistance2(p,a,b) {
  const l=distance2(a,b),t=l?Math.max(0,Math.min(1,((p[0]-a[0])*(b[0]-a[0])+(p[1]-a[1])*(b[1]-a[1]))/l)):0;
  return distance2(p,[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);
}
