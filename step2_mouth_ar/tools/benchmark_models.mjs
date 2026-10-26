#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { TransformerToothDetector } from '../src/core/TransformerToothDetector.js';
import { decodeLabels, encodeLabels, resizeInstanceLabels, compareInstances, instanceStats } from '../src/eval/toothInstances.js';
import { traceLabelContours, validateVectorContours, vectorBoundaryDistance } from '../src/eval/vectorBoundary.js';
import { frameTruth } from '../src/eval/frameEvaluation.js';
import { summarizeBenchmark, chooseValidationCandidate, resizeRgb } from '../src/eval/benchmark.js';

const [mediaFile,modelsFile,outFile]=process.argv.slice(2);
if(!outFile)throw new Error('Usage: benchmark_models.mjs media/manifest.json models.json report.json');
const media=JSON.parse(fs.readFileSync(mediaFile)),config=JSON.parse(fs.readFileSync(modelsFile));
if(media.schema!=='evaluation-media/1')throw new Error('Prepare native crops with prepare_evaluation_media.py');
if (!Array.isArray(config.models)||!config.models.length||new Set(config.models.map(m=>m.id)).size!==config.models.length) throw new Error('Model IDs must be unique');
const groups=new Map(),sources=new Map(),sampleIds=new Set();
for (const s of media.samples) {
  if(sampleIds.has(s.id))throw new Error('Duplicate sample ID');sampleIds.add(s.id);
  if(!s.group||!['train','valid','test','unassigned'].includes(s.split))throw new Error('Every sample needs a group and explicit split');
  if(groups.has(s.group)&&groups.get(s.group)!==s.split)throw new Error('Session group crosses splits');groups.set(s.group,s.split);
  const sha=s.videoSha256??s.sourceSha256;
  if(sha&&sources.has(sha)&&sources.get(sha)!==s.split)throw new Error('Same source crosses splits');if(sha)sources.set(sha,s.split);
}

const mediaRoot=path.dirname(mediaFile),modelRoot=path.dirname(modelsFile),models=[];
const hash=f=>createHash('sha256').update(fs.readFileSync(f)).digest('hex');
for(const m of config.models) {
  let detector;const result={id:m.id,rows:[]};models.push(result);
  try {
    const model=path.resolve(modelRoot,m.model),card=m.card?JSON.parse(fs.readFileSync(path.resolve(modelRoot,m.card))):null;
    result.modelSha256=hash(model);result.card=card;
    if(m.kind==='learned')detector=new LearnedToothDetector({modelUrl:model});
    else if(m.kind==='segformer')detector=new TransformerToothDetector({modelUrl:model,card});
    else throw new Error('Unknown model kind');
    await detector.init();result.parameters=detector.params;const W=detector.inputWidth,H=detector.inputHeight;
    let previous=null,previousGroup=null,previousTime=null;
    for(const s of media.samples) {
      if(!Number.isInteger(s.width)||!Number.isInteger(s.height)||s.width<1||s.height<1||s.width*s.height>4_000_000)throw new Error('Invalid native crop');
      const rgbFile=path.resolve(mediaRoot,s.rgb),rgb=fs.readFileSync(rgbFile),image=resizeRgb(rgb,s.width,s.height,W,H);
      const roi={bounds:{u0:0,u1:1,v0:0,v1:1},roiToLocal:(x,y)=>({u:x/W,v:y/H})};
      const start=performance.now();await detector.detectAsync(image,new Uint8Array(W*H).fill(255),roi);const inferenceMs=performance.now()-start;
      const prediction=resizeInstanceLabels(detector.lastDebug.labels,W,H,s.width,s.height),pred=traceLabelContours(prediction,s.width,s.height);
      let boundary=null,instances=null,labelsSha256=null;
      if(s.labels) {
        const file=path.resolve(mediaRoot,s.labels),doc=JSON.parse(fs.readFileSync(file));labelsSha256=hash(file);
        let labels,gt;
        if(s.videoSha256 && doc.frame) {const truth=frameTruth(doc,s,s.width,s.height);labels=truth.labels;gt=truth.contours;}
        else {
          if(doc.schema!=='tooth-labels/1'||doc.annotationStatus?.startsWith('unreviewed')||doc.width!==s.width||doc.height!==s.height
            ||doc.cleanImage!==s.cleanImage||JSON.stringify(doc.crop)!==JSON.stringify(s.crop))throw new Error('GT does not match the native photo crop');
          labels=decodeLabels(doc.labels,s.width*s.height);const vectors=validateVectorContours(doc.vectorContours??[],s.width,s.height,labels),ids=new Set(vectors.map(c=>c.id));
          gt=[...traceLabelContours(labels,s.width,s.height).filter(c=>!ids.has(c.id)),...vectors];
        }
        if(!gt.length)throw new Error('Ground truth has no labelled teeth');
        boundary=vectorBoundaryDistance(gt,pred,s.width,s.height);instances=compareInstances(labels,prediction,s.width);
      }
      const temporal=s.timeMs!=null&&s.group===previousGroup&&previousTime!=null&&s.timeMs>previousTime
        ?vectorBoundaryDistance(previous,pred,s.width,s.height):null;
      result.rows.push({id:s.id,group:s.group,split:s.split,timeMs:s.timeMs??null,sourceCrop:s.sourceCrop,videoSha256:s.videoSha256,captureProtocol:s.captureProtocol,decodedFrameIndex:s.decodedFrameIndex,decodedTimeMs:s.decodedTimeMs,
        hasGroundTruth:!!s.labels,labelsSha256,rgbSha256:hash(rgbFile),normalizedBoundaryError:boundary?.normalizedAssd??null,
        temporalBoundaryChange:temporal?.normalizedAssd??null,status:boundary?.status??'unlabelled',
        predictedRegions:instanceStats(prediction,s.width).length,
        missingTeeth:instances ? instances.groundTruthCount-instances.matchedAt50 : null,extraPredictions:instances ? instances.predictionCount-instances.matchedAt50 : null,
        boundary,instances,inferenceMs});
      previous=pred;previousGroup=s.group;previousTime=s.timeMs??null;
      if(config.exportPredictions) {
        const dir=path.join(path.dirname(outFile),'predictions',encodeURIComponent(m.id));fs.mkdirSync(dir,{recursive:true});
        fs.writeFileSync(path.join(dir,`${encodeURIComponent(s.id)}.json`),JSON.stringify({schema:'tooth-labels/1',annotationStatus:'unreviewed_model_draft',cleanImage:s.image,
          width:s.width,height:s.height,crop:[0,0,s.width,s.height],labels:encodeLabels(prediction)}));
      }
    }
    result.summary=summarizeBenchmark(result.rows);
  }catch(e){result.error=e.message;result.summary=summarizeBenchmark(result.rows);}
  finally{detector?.dispose();}
  console.log(JSON.stringify({id:result.id,error:result.error,summary:result.summary}));
}
const report={schema:'dental-model-benchmark/1',createdAt:new Date().toISOString(),models,
  selection:chooseValidationCandidate(models,config.baselineId),
  scope:'Native crop vector ASSD divided by crop diagonal. Bilinear RGB resize before each model. Missing predictions penalized as 1 in summary. Coverage counts use one-to-one IoU >= 0.5 matches. Temporal change is unregistered, includes motion and is not accuracy. No model deployed automatically.'};
fs.writeFileSync(outFile,JSON.stringify(report,null,2));
// Standalone SVG plot, with gaps wherever GT is unavailable.
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500"><rect width="900" height="500" fill="#151b24"/>'];
for(const [panel,key,title]of [[0,'normalizedBoundaryError','GT normalized boundary error'],[1,'temporalBoundaryChange','Unregistered temporal change (not accuracy)']]) {
  const top=30+panel*240,values=models.flatMap(m=>m.rows.map(r=>r[key])).filter(Number.isFinite),max=Math.max(.001,...values);
  parts.push(`<text x="40" y="${top}" fill="white">${title} — max ${max.toPrecision(3)}</text>`);
  models.forEach((m,index)=>{const color=['#38bdf8','#fbbf24','#e879f9','#4ade80'][index%4],n=Math.max(1,m.rows.length-1);let points=[];
    const flush=()=>{if(points.length)parts.push(`<polyline points="${points.join(' ')}" stroke="${color}" fill="none"/>`);points=[];};
    m.rows.forEach((r,i)=>{if(!Number.isFinite(r[key])){flush();return;}const x=40+i/n*820,y=top+180-r[key]/max*150;points.push(`${x},${y}`);parts.push(`<circle cx="${x}" cy="${y}" r="2" fill="${color}"/>`);});flush();
    parts.push(`<text x="${40+index*220}" y="${top+205}" fill="${color}">${escape(m.id)} (sample order)</text>`);
  });
  if(!values.length)parts.push(`<text x="80" y="${top+90}" fill="#94a3b8">No matching boundaries available</text>`);
}
parts.push('</svg>');fs.writeFileSync(outFile.replace(/\.json$/,'')+'.svg',parts.join('\n'));
