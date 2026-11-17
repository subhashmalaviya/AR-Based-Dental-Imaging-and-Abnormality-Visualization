import { decodeLabels, compareInstances, instanceStats } from './toothInstances.js';
import { traceLabelContours, validateVectorContours, vectorBoundaryDistance } from './vectorBoundary.js';
import { summarizeBoundaryMetrics } from './boundaryOptions.js';
export const TARGETS=['teeth','gums'];
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
export function validateBenchmarkMedia(media) {
  if(media?.schema!=='evaluation-media/1'||!Array.isArray(media.samples)||!media.samples.length)throw new Error('Expected a nonempty evaluation-media/1 manifest');
  const ids=new Set(),groups=new Map(),sources=new Map(),times=new Map();
  for(const s of media.samples) {
    if(typeof s.id!=='string'||!s.id||ids.has(s.id))throw new Error('Sample IDs must be unique');ids.add(s.id);
    if(!s.group||!['train','valid','test','unassigned'].includes(s.split))throw new Error('Each sample needs a session group and split');
    for(const [map,key] of [[groups,s.group],[sources,s.videoSha256??s.sourceSha256]])if(key){if(map.has(key)&&map.get(key)!==s.split)throw new Error('Session/source crosses splits');map.set(key,s.split);}
    if(![s.width,s.height].every(v=>Number.isInteger(v)&&v>=1)||s.width*s.height>32_000_000)throw new Error('Invalid native sample dimensions (maximum 32 MP)');
    if(s.timeMs!=null){
      if(!s.videoSha256||!s.captureProtocol||!Array.isArray(s.sourceCrop)||s.sourceCrop.length!==4||!s.sourceCrop.every(Number.isInteger)||s.sourceCrop[0]<0||s.sourceCrop[1]<0||s.sourceCrop[2]!==s.width||s.sourceCrop[3]!==s.height||!Number.isFinite(s.timeMs)||s.timeMs<0)throw new Error('Video samples require source hash, protocol, matching crop and timestamp');
      const key=sequenceKey(s),prev=times.get(key);if(prev!=null&&s.timeMs<=prev)throw new Error('Sequence timestamps must increase');times.set(key,s.timeMs);
    }
  }
  return media;
}
export function sequenceKey(s) {
  if(s.timeMs==null||!s.videoSha256)return null;
  return JSON.stringify([s.group,s.split,s.videoSha256,s.captureProtocol,s.sourceCrop,s.width,s.height]);
}
export function readTargetTruth(doc,s) {
  if(!['tooth-labels/1','dental-targets/1'].includes(doc?.schema)||doc.annotationStatus?.startsWith('unreviewed'))throw new Error('Use reviewed dental target or tooth labels');
  if(doc.width!==s.width||doc.height!==s.height||doc.cleanImage!==s.cleanImage||!same(doc.crop,s.crop))throw new Error('GT image dimensions, filename or crop differ');
  if(s.videoSha256 && doc.frame) {
    for(const k of ['captureProtocol','videoSha256','timeMs','sourceCrop'])if(!same(doc.frame[k],s[k]))throw new Error('GT video identity, timestamp or crop differ');
  }
  if(doc.schema==='dental-targets/1' && doc.annotationStatus!=='reviewed')throw new Error('Multi-target GT must explicitly be reviewed');
  const targetDocs=doc.schema==='tooth-labels/1'?{[doc.target??'teeth']:doc}:doc.targets;
  if(!targetDocs||Object.keys(targetDocs).some(k=>!TARGETS.includes(k)))throw new Error('Unknown ground-truth target');
  const targets={};
  for(const target of TARGETS)if(targetDocs[target]) {
    const entry=targetDocs[target],raw=decodeLabels(entry.labels,s.width*s.height);
    const vectors=validateVectorContours(entry.vectorContours??[],s.width,s.height,raw);
    // Gums are a tissue region, not individual teeth: union component labels.
    const labels=target==='gums'?Int32Array.from(raw,n=>n>0?1:0):raw;
    const ids=new Set(vectors.map(c=>c.id));
    const contours=target==='gums'?traceLabelContours(labels,s.width,s.height):
      [...traceLabelContours(labels,s.width,s.height).filter(c=>!ids.has(c.id)),...vectors];
    targets[target]={labels,contours};
  }
  return targets;
}
export function scoreTarget(prediction,truth,s,target,options={}) {
  if(prediction==null)return {status:'unsupported target',hasGroundTruth:!!truth,supported:false,normalizedBoundaryError:null};
  if(prediction.length!==s.width*s.height||prediction.some(n=>!Number.isInteger(n)||n<0))throw new Error('Invalid prediction label map');
  const labels=target==='gums'?Int32Array.from(prediction,n=>n>0?1:0):prediction;
  const contours=traceLabelContours(labels,s.width,s.height),base={supported:true,hasGroundTruth:!!truth,predictedRegions:instanceStats(labels,s.width).length};
  if(!truth)return {...base,status:'unlabelled',normalizedBoundaryError:null,contours};
  const bothEmpty=!contours.length&&!truth.contours.length;
  const boundary=vectorBoundaryDistance(truth.contours,contours,s.width,s.height,options);
  const error=bothEmpty?0:boundary.normalizedAssd;
  const instances=target==='teeth'?compareInstances(truth.labels,labels,s.width):null;
  return {...base,status:bothEmpty?'both empty':!truth.contours.length?'unexpected prediction':boundary.status,
    normalizedBoundaryError:error,boundary,instances,contours,
    missingTeeth:instances?instances.groundTruthCount-instances.matchedAt50:null,
    extraPredictions:instances?instances.predictionCount-instances.matchedAt50:null};
}
export function summarizeTargets(rows) {
  return Object.fromEntries(TARGETS.map(target=>{
    const all=rows.filter(r=>r.target===target),supported=all.filter(r=>r.supported),gt=supported.filter(r=>r.hasGroundTruth&&!r.error),
      errors=gt.map(r=>r.normalizedBoundaryError??1),temporal=supported.map(r=>r.temporalBoundaryChange).filter(Number.isFinite);
    const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
    return [target,{...summarizeBoundaryMetrics(gt),samples:all.length,supported:supported.length,unsupported:all.filter(r=>r.status==='unsupported target').length,
      failed:all.filter(r=>r.error).length,labelled:gt.length,missingBoundaries:gt.filter(r=>r.normalizedBoundaryError==null).length,
      meanNormalizedBoundaryError:mean(errors),missingTeeth:target==='teeth'&&gt.length?gt.reduce((s,r)=>s+(r.missingTeeth??0),0):null,
      extraPredictions:target==='teeth'&&gt.length?gt.reduce((s,r)=>s+(r.extraPredictions??0),0):null,
      temporalPairs:temporal.length,meanTemporalChange:mean(temporal),meanInferenceMs:mean(supported.map(r=>r.inferenceMs).filter(Number.isFinite))}];
  }));
}
export async function runTargetBenchmark({media,models,readSample,loadModel,boundaryOptions={},onProgress=()=>{},cancelled=()=>false}) {
  validateBenchmarkMedia(media);
  if(!models.length||new Set(models.map(m=>m.id)).size!==models.length)throw new Error('Choose unique model IDs');
  const results=[];
  for(const spec of models) {
    if(cancelled())break;
    const result={id:spec.id,provenance:spec.provenance??null,rows:[]};results.push(result);
    let model;const previous=new Map();
    try {
      onProgress(`Loading ${spec.id}`);model=await loadModel(spec);result.modelSha256=model.sha256;result.card=model.card;result.parameters=model.parameters;
      for(const s of media.samples) {
        if(cancelled())break;
        onProgress(`${spec.id}: ${s.id}`);
        const base={id:s.id,group:s.group,split:s.split,timeMs:s.timeMs??null,sequence:sequenceKey(s)};
        try {
          const {rgb,doc,rgbSha256,labelsSha256}=await readSample(s);
          const truth=doc?readTargetTruth(doc,s):{};
          const start=performance.now(),prediction=await model.infer(rgb,s),inferenceMs=performance.now()-start;
          for(const target of TARGETS) {
            const scored=scoreTarget(prediction[target],truth[target],s,target,boundaryOptions),key=base.sequence?`${target}:${base.sequence}`:null;
            const prev=key?previous.get(key):null;
            const temporal=prev&&scored.supported?vectorBoundaryDistance(prev,scored.contours,s.width,s.height,boundaryOptions):null;
            const temporalBoundaryChange=temporal?(prev.length===0&&scored.contours.length===0?0:temporal.normalizedAssd??1):null;
            if(key && scored.supported)previous.set(key,scored.contours);
            const {contours,...metrics}=scored;
            result.rows.push({...base,target,...metrics,inferenceMs,rgbSha256,labelsSha256,
              temporalBoundaryChange,temporalStatus:temporal?.status??null});
          }
        }catch(e){
          for(const target of TARGETS){previous.delete(`${target}:${base.sequence}`);result.rows.push({...base,target,status:'failed',error:e.message,supported:model.targets.includes(target),hasGroundTruth:false});}
        }
        await new Promise(r=>setTimeout(r,0));
      }
    }catch(e){result.error=e.message;}
    finally{try{await model?.dispose?.();}catch(e){result.error=`Cleanup failed: ${e.message}`;}result.summary=summarizeTargets(result.rows);}
  }
  return {schema:'dental-target-benchmark/1',createdAt:new Date().toISOString(),cancelled:cancelled(),boundaryOptions,models:results,
    scope:'Same native RGB samples for every model. Teeth: instance borders and one-to-one IoU>=0.5 coverage. Gums: union tissue boundary. Missing one-sided boundaries receive 1 in aggregate; both empty receives 0. Unsupported targets and failed entries are reported separately. Temporal change is unregistered, motion-sensitive, and is not accuracy. No automatic deployment.'};
}
