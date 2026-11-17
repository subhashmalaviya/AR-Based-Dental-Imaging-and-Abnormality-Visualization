export function benchmarkCsv(report) {
  const fields=['model','target','id','group','split','timeMs','status','normalizedBoundaryError','hausdorffPx','sdfCoverage','sdfPass','missingTeeth','extraPredictions','temporalBoundaryChange','inferenceMs','error'];
  const cell=v=>{let s=v==null?'':String(v);if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  const rows=[fields,...report.models.flatMap(m=>[...m.rows,...(m.error?[{status:'model failed',error:m.error}]:[])].map(r=>fields.map(k=>k==='model'?m.id:r[k]??r.boundary?.[k]??'')))];
  return rows.map(r=>r.map(cell).join(',')).join('\r\n');
}
// Separate series prevent lines joining different sources, crops or targets.
export function benchmarkSeries(report,target,key) {
  const series=[];
  for(const m of report.models) {
    const groups=new Map();
    for(const r of m.rows)if(r.target===target&&r.sequence){if(!groups.has(r.sequence))groups.set(r.sequence,[]);groups.get(r.sequence).push(r);}
    for(const [sequence,rows]of groups){const paths=[];let points=[];
      for(const r of rows){if(!Number.isFinite(r[key])){if(points.length)paths.push(points);points=[];}else points.push([r.timeMs,r[key]]);}
      if(points.length)paths.push(points);series.push({model:m.id,sequence,paths});
    }
  }return series;
}
