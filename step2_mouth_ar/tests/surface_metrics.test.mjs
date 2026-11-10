import test from 'node:test';
import assert from 'node:assert/strict';
import { vectorBoundaryDistance } from '../src/eval/vectorBoundary.js';
import { boundaryDistance } from '../src/eval/boundaryDistance.js';
import { boundaryOptions, summarizeBoundaryMetrics } from '../src/eval/boundaryOptions.js';
import { evaluateVideoFrame } from '../src/eval/frameEvaluation.js';
import { scoreBatchPrediction } from '../src/eval/batchEvaluation.js';
import { createBatchWorkbook } from '../src/eval/batchWorkbook.js';
const square=(x=2,y=2,size=4)=>({id:1,smooth:false,points:[[x,y],[x+size,y],[x+size,y+size],[x,y+size]]});
const close=(a,b,t=1e-8)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);

test('identical borders have zero distances and 100% SDF coverage even at zero tolerance',()=>{
  const b=vectorBoundaryDistance([square()],[square()],20,20,{tolerancePx:0,requiredCoverage:1});
  assert.equal(b.assdPx,0);assert.equal(b.hausdorffPx,0);close(b.sdfCoverage,1);assert.equal(b.sdfPass,true);
});
test('Hausdorff catches a remote extra region that average distance dilutes; symmetric and normalized',()=>{
  const a=[square()],b=[square(),{...square(14,2,1),id:2}];
  const m=vectorBoundaryDistance(a,b,20,20),reverse=vectorBoundaryDistance(b,a,20,20);
  close(m.hausdorffPx,9);assert.ok(m.assdPx<m.hausdorffPx/3);
  close(m.normalizedHausdorff,9/Math.hypot(20,20));close(m.hausdorffPx,reverse.hausdorffPx);close(m.sdfCoverage,reverse.sdfCoverage);
  assert.equal(m.sdfPass,false);
});
test('SDF tolerance is inclusive for inside and outside offsets; configured pass threshold changes verdict',()=>{
  const gt=[square(2,2,4)],pred=[square(2.1,2.1,3.8)];
  const small=vectorBoundaryDistance(gt,pred,20,20,{tolerancePx:.099});
  close(small.sdfCoverage,0); assert.equal(small.sdfPass,false);
  const at=vectorBoundaryDistance(gt,pred,20,20,{tolerancePx:.1,requiredCoverage:.95});
  close(at.sdfCoverage,30.4/31.2,0.001);assert.equal(at.sdfPass,true);
  assert.equal(vectorBoundaryDistance(gt,pred,20,20,{tolerancePx:.1,requiredCoverage:1}).sdfPass,false);
  close(at.sdfError,1-at.sdfCoverage);
});
test('Hausdorff includes segment interiors, not only vertices',()=>{
  const a=[{id:1,smooth:false,points:[[2,2],[10,2],[10,10],[2,10]]}];
  const b=a[0].points.map((p,i)=>({...square(p[0]-.01,p[1]-.01,.02),id:i+1}));
  const r=vectorBoundaryDistance(a,b,20,20);
  assert.ok(r.hausdorffPx>3.9); // all GT vertices themselves lie very close to the prediction
});
test('empty prediction fails SDF; absent GT and both-empty are unavailable, never PASS',()=>{
  const missing=vectorBoundaryDistance([square()],[],20,20);
  assert.equal(missing.hausdorffPx,null);assert.equal(missing.sdfCoverage,0);assert.equal(missing.sdfPass,false);
  for(const [a,b] of [[[],[]],[[],[square()]]]){
    const r=vectorBoundaryDistance(a,b,20,20);assert.equal(r.hausdorffPx,null);assert.equal(r.sdfPass,null);
  }
  assert.throws(()=>boundaryOptions({tolerancePx:-1}));assert.throws(()=>boundaryOptions({requiredCoverage:1.01}));
});
test('raster pixel-center metric computes exact max and tolerance coverage',()=>{
  const a=new Uint8Array(25),b=new Uint8Array(25);a[6]=1;b[8]=1;
  const r=boundaryDistance(a,b,5,5,null,Math.hypot(5,5),{tolerancePx:2});
  assert.equal(r.hausdorffPx,2);assert.equal(r.sdfCoverage,1);assert.equal(r.sdfPass,true);
});
test('batch and video frame scoring expose the same metrics and retain evaluation settings',()=>{
  const labels=new Uint32Array(400);for(let y=2;y<6;y++)for(let x=2;x<6;x++)labels[y*20+x]=1;
  const options={tolerancePx:.2,requiredCoverage:1},truth={contours:[square()]};
  const batch=scoreBatchPrediction(truth.contours,labels,20,20,options);
  const frame=evaluateVideoFrame(labels,20,20,{truth,boundaryOptions:options});
  assert.deepEqual(batch.boundary,frame.gt);assert.equal(frame.gt.sdfPass,true);
});
test('Excel includes every image metric, numeric summaries, settings, and unavailable/failing rows',async()=>{
  const exact=vectorBoundaryDistance([square()],[square()],20,20);
  const missing=vectorBoundaryDistance([square()],[],20,20);
  const rows=[{patient:'p',image:'one.png',error:0,status:'ok',boundary:exact},
    {patient:'p',image:'two.png',error:null,status:'missing prediction',boundary:missing},
    {patient:'p',image:'three.png',error:null,status:'failed'}];
  const metrics=summarizeBoundaryMetrics(rows);assert.equal(metrics.sdfFailed,1);assert.equal(metrics.sdfUnavailable,1);close(metrics.meanSdfCoverage,.5);
  const book=await createBatchWorkbook(rows,{timestamp:'2026-11-10T00:00:00Z',boundaryOptions:{tolerancePx:.1,requiredCoverage:.95}});
  const bytes=await book.xlsx.writeBuffer();const {default:ExcelJS}=await import('exceljs');const round=new ExcelJS.Workbook();await round.xlsx.load(bytes);
  const images=round.getWorksheet('Images'),s=round.getWorksheet('Summary');
  assert.equal(images.getCell('F2').value,0);assert.equal(images.getCell('F3').value,null);
  close(images.getCell('H2').value,1);assert.equal(images.getCell('H3').value,0);assert.equal(images.getCell('H4').value,null);
  assert.equal(images.getCell('J2').value,'PASS');assert.equal(images.getCell('J3').value,'FAIL');assert.equal(images.getCell('J4').value,'Unavailable');
  assert.equal(images.getCell('K2').value,.1);assert.equal(images.getCell('L2').value,.95);
  close(s.getCell('B19').value.result,.5);assert.equal(s.getCell('B21').value.result,1);assert.equal(s.getCell('B22').value.result,1);
});
