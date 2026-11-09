import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { importGroundTruth } from '../src/eval/importGroundTruth.js';
import { validateVectorContours, rasterizeVector } from '../src/eval/vectorBoundary.js';
import { encodeLabels } from '../src/eval/toothInstances.js';
import { UploadedPhotoComparison } from '../src/ui/UploadedPhotoComparison.js';

const box = (x,y,w=3,h=3) => [[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
test('JSON import creates individual editable masks and movable borders without coordinate shifts', () => {
  const doc={teeth:[{contour:box(1,1)},{contour:box(6,1)}]};
  const result=importGroundTruth(doc,12,8,'a.png');
  assert.deepEqual(result.crop,[0,0,12,8]);
  assert.equal(result.labels[2*12+2],1);assert.equal(result.labels[2*12+7],2);
  assert.equal(result.labels[0],0);assert.equal(result.vectors.length,2);
  assert.deepEqual(result.vectors[0].points,doc.teeth[0].contour);
  assert.deepEqual(validateVectorContours(result.vectors,12,8,result.labels),result.vectors);
  const edited=structuredClone(result.vectors[0]);edited.points[0]=[0,0];
  rasterizeVector(result.labels,12,8,edited);assert.equal(result.labels[0],1);
});
test('multipart teeth retain every contour under one label and overlaps never create inconsistent vectors', () => {
  const multi=importGroundTruth({teeth:[{contours:[box(1,1),box(6,1)]}]},12,8,'a.png');
  assert.equal(multi.labels[2*12+2],1);assert.equal(multi.labels[2*12+7],1);
  assert.equal(multi.pixelOnlyTeeth,1);assert.equal(multi.vectors.length,0);
  const overlap=importGroundTruth({teeth:[{contour:box(1,1,5,5)},{contour:box(4,1)}]},12,8,'a.png');
  assert.equal(overlap.labels[2*12+2],1);assert.equal(overlap.labels[2*12+5],2);
  assert.equal(overlap.pixelOnlyTeeth,1);
  assert.doesNotThrow(()=>validateVectorContours(overlap.vectors,12,8,overlap.labels));
  assert.throws(()=>importGroundTruth({teeth:[{contour:box(1,1)},{contour:box(1,1)}]},12,8,'a.png'),/no editable pixels/);
});
test('saved editor JSON restores exact pixel corrections and crop, not a newly approximated mask', () => {
  const labels=Int32Array.from([0,7,0,0]);
  const doc={schema:'tooth-labels/1',cleanImage:'a.png',crop:[2,3,2,2],width:2,height:2,labels:encodeLabels(labels),vectorContours:[]};
  const result=importGroundTruth(doc,10,10,'a.png');
  assert.deepEqual(result.labels,labels);assert.deepEqual(result.crop,doc.crop);
  assert.throws(()=>importGroundTruth(doc,10,10,'b.png'),/match/);
});
test('patient fixtures import all tooth labels with consistent editable vectors', async t => {
  let count=0;
  for(const patient of [1,2]) for(const view of ['Center','Down','Left','Right','Upper']) {
    const name=`IOP_${view}_${patient}`,base=new URL(`../../testvideophoto/Patient_${patient}/${name}`,import.meta.url);
    let doc,bytes;
    try {doc=JSON.parse(await readFile(base.pathname+'.json','utf8'));bytes=await readFile(base.pathname+'.png');}
    catch(e){if(e.code==='ENOENT'){t.skip('Optional patient fixtures unavailable');return;}throw e;}
    const w=bytes.readUInt32BE(16),h=bytes.readUInt32BE(20),result=importGroundTruth(doc,w,h,name+'.png');
    const ids=new Set(result.labels);ids.delete(0);assert.equal(ids.size,doc.annotation_count);
    assert.doesNotThrow(()=>validateVectorContours(result.vectors,w,h,result.labels));count++;
  }
  assert.equal(count,10);
});
test('invalid JSON import leaves current mask and prediction intact and unlocks controls', async () => {
  const app=Object.create(UploadedPhotoComparison.prototype),els=new Map();
  app.el=id=>{if(!els.has(id))els.set(id,{disabled:false});return els.get(id);};
  const labels=new Int32Array(100),pred=new Int32Array(100),locks=[];
  Object.assign(app,{clean:{width:10,height:10},cleanName:'a.png',truthLabels:labels,predictionLabels:pred,
    root:{querySelectorAll:()=>[app.el('uploadRun')]},editor:{refresh(){}},setBusy:v=>locks.push(v)});
  await app.loadTruthJson({size:5,text:async()=>'{oops'});
  assert.strictEqual(app.truthLabels,labels);assert.strictEqual(app.predictionLabels,pred);
  assert.equal(app.busy,false);assert.deepEqual(locks,[true,false]);
  assert.match(app.el('uploadStatus').textContent,/Could not load/);assert.equal(app.el('uploadRun').disabled,false);
});
