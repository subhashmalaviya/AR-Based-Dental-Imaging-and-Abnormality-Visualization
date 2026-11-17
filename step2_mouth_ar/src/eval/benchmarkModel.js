import { LearnedToothDetector, readModel } from '../core/LearnedToothDetector.js';
import { TransformerToothDetector } from '../core/TransformerToothDetector.js';
import { resizeRgb } from './benchmark.js';
import { resizeInstanceLabels } from './toothInstances.js';
import { loadIopBenchmarkModel } from './iopBenchmarkModel.js';
export async function sha256(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
}
export function acceptedUnetLabels(labels,detections) {
  const accepted=new Set(detections.map(d=>d.mask.label));
  return Int32Array.from(labels,n=>n>=0&&accepted.has(n)?n+1:0);
}
export async function loadBenchmarkModel(spec) {
  if(spec.kind==='iop')return loadIopBenchmarkModel(spec);
  const detector=spec.kind==='learned'?new LearnedToothDetector({modelUrl:spec.model,infoUrl:spec.info}):
    spec.kind==='segformer'?new TransformerToothDetector({modelUrl:spec.model,card:spec.card}):null;
  if(!detector)throw new Error('Supported checkpoint kinds: learned, segformer');
  try {
    const hash=await sha256(await readModel(spec.model));
    if(spec.sha256&&hash!==spec.sha256)throw new Error('Checkpoint hash mismatch');
    await detector.init();
    const W=detector.inputWidth,H=detector.inputHeight;
    return {sha256:hash,card:detector.info,parameters:detector.params,
      targets:spec.kind==='learned'?['teeth']:[...(spec.card.toothClasses.length?['teeth']:[]),...(spec.card.gumClasses?.length?['gums']:[])],
      async infer(rgb,s) {
        const image=resizeRgb(rgb,s.width,s.height,W,H),roi={bounds:{u0:0,u1:1,v0:0,v1:1},roiToLocal:(x,y)=>({u:x/W,v:y/H})};
        const detections=await detector.detectAsync(image,new Uint8Array(W*H).fill(255),roi);
        const maps=spec.kind==='learned'?{teeth:acceptedUnetLabels(detector.lastDebug.labels,detections)}:detector.lastDebug.targets;
        return Object.fromEntries(Object.entries(maps).map(([target,labels])=>[target,resizeInstanceLabels(labels,W,H,s.width,s.height)]));
      },async dispose(){await detector.dispose();}};
  }catch(e){await detector.dispose();throw e;}
}
