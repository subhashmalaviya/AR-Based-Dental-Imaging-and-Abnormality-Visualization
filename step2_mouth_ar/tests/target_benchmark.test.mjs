import test from 'node:test';
import assert from 'node:assert/strict';
import {validateBenchmarkMedia,readTargetTruth,scoreTarget,summarizeTargets,runTargetBenchmark,sequenceKey} from '../src/eval/targetBenchmark.js';
import {acceptedUnetLabels} from '../src/eval/benchmarkModel.js';
import {benchmarkCsv,benchmarkSeries} from '../src/eval/benchmarkReport.js';
import {encodeLabels} from '../src/eval/toothInstances.js';
import {decodeSegformer,validateTransformerCard} from '../src/core/TransformerToothDetector.js';
const sample={id:'a',group:'patient1',split:'test',cleanImage:'a.png',width:8,height:8,crop:[0,0,8,8]};
const filled=new Int32Array(64).fill(1),empty=new Int32Array(64);
const doc=(s=sample,labels=filled)=>({schema:'tooth-labels/1',cleanImage:s.cleanImage,width:s.width,height:s.height,crop:s.crop,labels:encodeLabels(labels)});
const frame=(id,timeMs,extra={})=>({...sample,id,timeMs,videoSha256:'video-a',captureProtocol:'browser-seek/1',sourceCrop:[0,0,8,8],...extra});
const media=samples=>({schema:'evaluation-media/1',samples});

test('U-Net background -1 and first tooth 0 map correctly; rejected instances stay background',()=>{
 assert.deepEqual([...acceptedUnetLabels(Int32Array.from([-1,0,1,2]),[{mask:{label:0}},{mask:{label:2}}])],[0,1,0,3]);
});
test('gum masks union adjacent labels; teeth retain individual borders; absent target is unlabelled',()=>{
 const labels=Int32Array.from({length:64},(_,i)=>i%8<4?1:2),d=doc(sample,labels);
 const teeth=readTargetTruth(d,sample).teeth,gums=readTargetTruth({...d,target:'gums'},sample).gums;
 assert.equal(scoreTarget(filled,gums,sample,'gums').normalizedBoundaryError,0);
 assert.ok(scoreTarget(filled,teeth,sample,'teeth').normalizedBoundaryError>0);
 assert.equal(scoreTarget(null,gums,sample,'gums').status,'unsupported target');
 assert.equal(scoreTarget(filled,null,sample,'gums').hasGroundTruth,false);
 assert.equal(readTargetTruth(d,sample).gums,undefined);
 assert.throws(()=>readTargetTruth({...d,cleanImage:'other.png'},sample),/differ/);
 assert.throws(()=>readTargetTruth({...d,schema:'dental-targets/1',targets:{teeth:d}},sample),/reviewed/);
 assert.throws(()=>readTargetTruth({...d,annotationStatus:'unreviewed_model_draft'},sample),/reviewed/);
});
test('missing boundary penalty differs from unsupported, unlabelled, and two empty masks',()=>{
 const truth=readTargetTruth(doc(),sample).teeth;
 const missing={...scoreTarget(empty,truth,sample,'teeth'),target:'teeth'};
 const unsupported={...scoreTarget(null,truth,sample,'teeth'),target:'teeth'};
 const unlabelled={...scoreTarget(filled,null,sample,'teeth'),target:'teeth'};
 const summary=summarizeTargets([missing,unsupported,unlabelled]).teeth;
 assert.equal(summary.meanNormalizedBoundaryError,1);assert.equal(summary.labelled,1);assert.equal(summary.unsupported,1);
 assert.equal(scoreTarget(empty,readTargetTruth(doc(sample,empty),sample).teeth,sample,'teeth').normalizedBoundaryError,0);
 assert.throws(()=>scoreTarget(new Int32Array(2),truth,sample,'teeth'),/Invalid/);
});
test('sequence validation prevents split leakage and mixes neither crop nor video',()=>{
 const a=frame('a',0),b=frame('b',1000);
 validateBenchmarkMedia(media([a,b]));
 assert.throws(()=>validateBenchmarkMedia(media([a,{...b,split:'valid'}])),/crosses/);
 assert.throws(()=>validateBenchmarkMedia(media([a,{...b,timeMs:0}])),/increase/);
 assert.throws(()=>validateBenchmarkMedia(media([{...a,sourceCrop:[0,0,9,8]}])),/matching crop/);
 assert.notEqual(sequenceKey(a),sequenceKey({...a,sourceCrop:[2,0,8,8]}));
 assert.notEqual(sequenceKey(a),sequenceKey({...a,videoSha256:'video-b'}));
 assert.throws(()=>readTargetTruth({...doc(a),frame:{...a,timeMs:50}},a),/timestamp/);
});
test('benchmark processes two checkpoints, preserves no-GT gaps, resets failed sequences, and releases models',async()=>{
 const frames=[frame('a',0),frame('b',1000),frame('c',2000),frame('d',3000),frame('e',4000),frame('f',0,{videoSha256:'video-b'})];
 let disposed=0;
 const report=await runTargetBenchmark({media:media(frames),models:[{id:'one'},{id:'two'}],
  readSample:async s=>{if(s.id==='c')throw new Error('Unreadable frame');return {rgb:new Uint8Array(192)};},
  loadModel:async()=>({targets:['teeth'],infer:async(rgb,s)=>({teeth:s.id==='b'?empty:filled}),dispose:async()=>{disposed++;}})});
 assert.equal(disposed,2);
 for(const m of report.models){const rows=m.rows.filter(r=>r.target==='teeth');
  assert.equal(rows[0].temporalBoundaryChange,null);assert.equal(rows[1].temporalBoundaryChange,1);
  assert.equal(rows[2].status,'failed');assert.equal(rows[3].temporalBoundaryChange,null);assert.equal(rows[4].temporalBoundaryChange,0);
  assert.equal(rows[5].temporalBoundaryChange,null);assert.equal(m.summary.teeth.meanNormalizedBoundaryError,null);
  assert.equal(m.summary.teeth.failed,1);assert.equal(m.summary.gums.unsupported,5);
 }
});
test('initialization failure and cancellation produce explicit partial results',async()=>{
 const common={media:media([sample]),models:[{id:'bad'},{id:'good'}],readSample:async()=>({rgb:new Uint8Array(192),doc:doc()})};
 const r=await runTargetBenchmark({...common,loadModel:async s=>{if(s.id==='bad')throw new Error('Missing weights');return {targets:['teeth'],infer:async()=>({teeth:filled})};}});
 assert.equal(r.models[0].error,'Missing weights');assert.equal(r.models[1].summary.teeth.meanNormalizedBoundaryError,0);
 const cancelled=await runTargetBenchmark({...common,cancelled:()=>true});assert.equal(cancelled.cancelled,true);assert.equal(cancelled.models.length,0);
});
test('SegFormer explicit gum classes use multiclass argmax without labelling background or teeth as gums',()=>{
 const card={schema:'dental-segformer-onnx/1',architecture:'segformer',modality:'rgb-intraoral',classes:4,toothClasses:[1],gumClasses:[2,3],minAreaPixels:1,input:{width:32,height:32,mean:[0,0,0],std:[1,1,1]}};
 const data=new Float32Array(4096);for(let i=0;i<1024;i++)data[Math.floor(i%32/8)*1024+i]=10;
 const tensor={dims:[1,4,32,32],data},aperture=new Uint8Array(1024).fill(255);
 const teeth=decodeSegformer(tensor,card,aperture).labels,gums=decodeSegformer(tensor,card,aperture,'gums').labels;
 assert.equal(teeth[0],0);assert.ok(teeth[8]>0);assert.equal(teeth[16],0);
 assert.equal(gums[0],0);assert.equal(gums[8],0);assert.ok(gums[16]>0);assert.equal(gums[16],gums[24]);
 assert.throws(()=>validateTransformerCard({...card,gumClasses:[1]}),/disjoint/);
 validateTransformerCard({...card,toothClasses:[]});
});
test('reports keep per-frame errors and plot gaps with safe CSV cells',()=>{
 const report={models:[{id:'=bad,"name',rows:[{id:'a',target:'teeth',sequence:'A',timeMs:0,normalizedBoundaryError:.1,boundary:{hausdorffPx:2}},{id:'b',target:'teeth',sequence:'A',timeMs:1000,normalizedBoundaryError:null},{id:'c',target:'teeth',sequence:'A',timeMs:2000,normalizedBoundaryError:.2}]}]};
 const csv=benchmarkCsv(report);assert.ok(csv.includes('"\'=bad,""name"'));assert.ok(csv.includes('"0.1"'));assert.ok(csv.includes('"2"'));
 assert.equal(benchmarkSeries(report,'teeth','normalizedBoundaryError')[0].paths.length,2);
 assert.deepEqual(benchmarkSeries(report,'gums','normalizedBoundaryError'),[]);
});
