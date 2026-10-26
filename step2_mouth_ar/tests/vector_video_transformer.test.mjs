import test from 'node:test';
import assert from 'node:assert/strict';
import { traceLabelContours, contourPath, flattenContour, vectorBoundaryDistance, validateVectorContours } from '../src/eval/vectorBoundary.js';
import { sampleTimes, validateCrop, frameTruth, evaluateVideoFrame } from '../src/eval/frameEvaluation.js';
import { encodeLabels } from '../src/eval/toothInstances.js';
import { plotSeries } from '../src/ui/VideoEvaluation.js';
import { decodeSegformer, validateTransformerCard } from '../src/core/TransformerToothDetector.js';
import { summarizeBenchmark, chooseValidationCandidate } from '../src/eval/benchmark.js';
const rectangle=(x,y,w,h,id=1)=>({id,smooth:false,points:[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]});
test('vectorization keeps holes, diagonal components, touching teeth and image edges',()=>{
  const c=traceLabelContours([1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,1],4,4);
  assert.equal(c.length,3);assert.equal(c.filter(c=>c.points.length===4).length,3);
  const area=c=>Math.abs(c.points.reduce((s,a,i)=>{const b=c.points[(i+1)%c.points.length];return s+a[0]*b[1]-a[1]*b[0]},0))/2;
  assert.deepEqual(c.map(area).sort((a,b)=>a-b),[1,1,9]);
  const touching=traceLabelContours([1,2],2,1);assert.equal(touching.length,2);
  assert.equal(vectorBoundaryDistance(touching,touching,2,1).assdPx,0);
  assert.ok(vectorBoundaryDistance(touching,traceLabelContours([1,1],2,1),2,1).assdPx>0);
});
test('distance is symmetric, subdivision invariant, handles missing boundaries and scales',()=>{
  const a=[rectangle(2,2,4,6)],b=[rectangle(5,2,4,6)],score=vectorBoundaryDistance(a,b,20,20);
  assert.equal(score.assdPx,vectorBoundaryDistance(b,a,20,20).assdPx);assert.ok(score.assdPx>0);
  const subdivided=structuredClone(a);subdivided[0].points.splice(1,0,[4,2]);
  assert.equal(vectorBoundaryDistance(a,subdivided,20,20).normalizedAssd,0);
  assert.equal(vectorBoundaryDistance([],a,20,20).normalizedAssd,null);
  assert.equal(vectorBoundaryDistance(a,[],20,20).status,'missing prediction');
  const scaled=c=>c.map(q=>({...q,points:q.points.map(p=>p.map(v=>v*2))}));
  assert.ok(Math.abs(vectorBoundaryDistance(scaled(a),scaled(b),40,40).normalizedAssd-score.normalizedAssd)<.0001);
});
test('curve flattening converges; malformed imports fail',()=>{
  const c={...rectangle(2,2,8,8),smooth:true};assert.match(contourPath(c),/C/);
  assert.ok(flattenContour(c,.01).length>=flattenContour(c,.1).length);
  assert.ok(vectorBoundaryDistance([c],[c],20,20).assdPx<1e-12);
  assert.deepEqual(validateVectorContours([c],20,20),[c]);
  assert.throws(()=>validateVectorContours([{...c,points:[[0,0],[NaN,1],[1,1]]}],20,20),/outside/);
});
test('video sampling, crop validation, GT hash/time binding and missing GT gaps',()=>{
  assert.deepEqual(sampleTimes(1.1,2),[0,500,1000]);assert.throws(()=>sampleTimes(601,1),/600/);
  assert.throws(()=>validateCrop([0,0,11,8],10,10),/crop/);
  const labels=new Int32Array(64).fill(1),ctx={videoSha256:'abc',timeMs:500,sourceCrop:[2,3,8,8]};
  const doc={schema:'tooth-labels/1',width:8,height:8,crop:[0,0,8,8],frame:ctx,labels:encodeLabels(labels)};
  const truth=frameTruth(doc,ctx,8,8);assert.equal(evaluateVideoFrame(labels,8,8,{truth}).normalizedBoundaryError,0);
  assert.throws(()=>frameTruth(doc,{...ctx,timeMs:0},8,8),/match/);
  assert.throws(()=>frameTruth({...doc,annotationStatus:'unreviewed_model_draft'},ctx,8,8),/reviewed/);
  const u=evaluateVideoFrame(labels,8,8,{previous:truth.contours});assert.equal(u.normalizedBoundaryError,null);assert.equal(u.temporalBoundaryChange,0);
  assert.equal(plotSeries([{timeMs:0,v:0},{timeMs:500,v:null},{timeMs:1000,v:.1}],'v').paths.length,2);
});
const card={schema:'dental-segformer-onnx/1',architecture:'segformer',modality:'rgb-intraoral',classes:3,toothClasses:[1,2],minAreaPixels:1,input:{width:32,height:32,mean:[.5,.5,.5],std:[.25,.25,.25]}};
test('SegFormer decoding separates adjacent classes, applies aperture and rejects wrong models',()=>{
  const data=new Float32Array(3*32*32);for(let y=0;y<32;y++)for(let x=0;x<32;x++)data[(x<16?1:2)*1024+y*32+x]=10;
  const aperture=new Uint8Array(1024).fill(255);aperture[0]=0;
  const r=decodeSegformer({dims:[1,3,32,32],data},card,aperture);
  assert.equal(r.labels[0],0);assert.notEqual(r.labels[10],r.labels[20]);assert.equal(Object.keys(r.classById).length,2);assert.ok(r.prob[100]>.99);
  assert.throws(()=>decodeSegformer({dims:[1,2,32,32],data},card,aperture),/logits/);
  assert.throws(()=>validateTransformerCard({...card,modality:'x-ray'}),/RGB/);
});
test('selection rejects test-only data, missing predictions, and lower error with worse coverage',()=>{
  const row={id:'a',split:'valid',hasGroundTruth:true,normalizedBoundaryError:.1,missingTeeth:0,extraPredictions:0,inferenceMs:10};
  const base={id:'base',rows:[row]},better={id:'candidate',rows:[{...row,normalizedBoundaryError:.05}]};
  assert.equal(chooseValidationCandidate([base,better],'base').selected,'candidate');
  better.rows[0].missingTeeth=1;assert.equal(chooseValidationCandidate([base,better],'base').selected,null);
  better.rows[0].normalizedBoundaryError=null;assert.equal(summarizeBenchmark(better.rows).meanNormalizedBoundaryError,1);
  assert.equal(chooseValidationCandidate([{...base,rows:[{...row,split:'test'}]}],'base').selected,null);
  assert.equal(summarizeBenchmark([{hasGroundTruth:false,temporalBoundaryChange:0}]).meanNormalizedBoundaryError,null);
});

test('curve masks round-trip and contradictory labels are rejected',async()=>{
  const {rasterizeVector}=await import('../src/eval/vectorBoundary.js');
  const c={...rectangle(2,2,8,8),smooth:true},labels=new Int32Array(400);
  rasterizeVector(labels,20,20,c);assert.ok(labels.includes(1));
  assert.deepEqual(validateVectorContours([c],20,20,labels),[c]);
  const moved=structuredClone(c);moved.points[0][0]+=3;
  assert.throws(()=>validateVectorContours([moved],20,20,labels),/disagree/);
});

test('SVG borders keep a one-screen-pixel stroke and native path under zoom',async()=>{
  const {ToothMaskEditor}=await import('../src/ui/ToothMaskEditor.js');
  const saved=globalThis.document;
  globalThis.document={createElementNS(ns,name){return {name,attrs:{},setAttribute(k,v){this.attrs[k]=v;}}}};
  try {
    const editor=Object.create(ToothMaskEditor.prototype),elements={toothEditPrediction:{checked:false},toothEditTool:{value:'vector-edit'}};
    const c={...rectangle(2,2,8,8),smooth:true};let scale=1;
    Object.assign(editor,{width:20,height:20,points:[],vectors:[c],selected:1,el:id=>elements[id],getContours:()=>[c],
      canvas:{getBoundingClientRect:()=>({width:20*scale})},svg:{children:[],setAttribute(){},replaceChildren(){this.children=[];},append(e){this.children.push(e);}}});
    editor.drawVectors();const first=editor.svg.children[0].attrs;
    scale=8;editor.drawVectors();const zoomed=editor.svg.children[0].attrs;
    assert.equal(first.d,zoomed.d);assert.equal(zoomed['stroke-width'],'1');assert.equal(zoomed['vector-effect'],'non-scaling-stroke');
    assert.equal(Number(editor.svg.children[1].attrs.r),3/8);
  }finally{globalThis.document=saved;}
});
