// Pure aggregation; no auto-deployment, no ranking unlabelled temporal scores.
export function summarizeBenchmark(rows) {
  const gt=rows.filter(r=>r.hasGroundTruth);
  return {frames:rows.length,labelledFrames:gt.length,
    meanNormalizedBoundaryError:gt.length?gt.reduce((s,r)=>s+(r.normalizedBoundaryError??1),0)/gt.length:null,
    failedOrMissingPredictions:gt.filter(r=>r.normalizedBoundaryError==null).length,
    missingTeeth:gt.reduce((s,r)=>s+(r.missingTeeth??0),0),extraPredictions:gt.reduce((s,r)=>s+(r.extraPredictions??0),0),
    meanInferenceMs:rows.length?rows.reduce((s,r)=>s+(r.inferenceMs??0),0)/rows.length:null};
}
export function chooseValidationCandidate(models,baselineId) {
  const identity=rows=>rows.map(r=>JSON.stringify([r.id,r.rgbSha256,r.labelsSha256])).sort();
  const baseline=models.find(m=>m.id===baselineId),expected=identity(baseline?.rows.filter(r=>r.split==='valid'&&r.hasGroundTruth)??[]);
  if(!expected.length)return {selected:null,reason:'No labelled validation samples. Test frames cannot select a winner.'};
  const summaries=models.map(m=>{
    const rows=m.rows.filter(r=>r.split==='valid'&&r.hasGroundTruth),summary=summarizeBenchmark(rows);
    return {id:m.id,summary,eligible:!m.error&&JSON.stringify(identity(rows))===JSON.stringify(expected)&&summary.failedOrMissingPredictions===0};
  });
  const b=summaries.find(s=>s.id===baselineId);
  const candidates=summaries.filter(s=>s.id!==baselineId&&s.eligible&&b.eligible
    &&s.summary.missingTeeth<=b.summary.missingTeeth&&s.summary.extraPredictions<=b.summary.extraPredictions
    &&s.summary.meanNormalizedBoundaryError<b.summary.meanNormalizedBoundaryError).sort((a,b)=>a.summary.meanNormalizedBoundaryError-b.summary.meanNormalizedBoundaryError);
  return {selected:candidates[0]?.id??null,reason:candidates.length?'Lower error on the same validation samples without more missed/extra instances; confirm on independent test data before deployment.':'No candidate demonstrated lower validation error without worse tooth coverage.',summaries};
}
export function resizeRgb(rgb,w,h,W,H) {
  if(rgb.length!==w*h*3)throw new Error('Invalid RGB input');
  const out=new Uint8ClampedArray(W*H*4);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++) {
    const u=(x+.5)*w/W-.5,v=(y+.5)*h/H-.5,x0=Math.floor(u),y0=Math.floor(v),fx=u-x0,fy=v-y0;
    const at=(xx,yy,k)=>rgb[(Math.max(0,Math.min(h-1,yy))*w+Math.max(0,Math.min(w-1,xx)))*3+k];
    for(let k=0;k<3;k++)out[(y*W+x)*4+k]=(1-fy)*((1-fx)*at(x0,y0,k)+fx*at(x0+1,y0,k))+fy*((1-fx)*at(x0,y0+1,k)+fx*at(x0+1,y0+1,k));
    out[(y*W+x)*4+3]=255;
  }return {data:out,width:W,height:H};
}
