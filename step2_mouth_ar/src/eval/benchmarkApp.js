import {readImageFile,IMAGE_ACCEPT} from '../ui/imageInput.js';
import {importGroundTruth} from './importGroundTruth.js';
import {encodeLabels} from './toothInstances.js';
import {runTargetBenchmark,validateBenchmarkMedia} from './targetBenchmark.js';
import {loadBenchmarkModel,sha256} from './benchmarkModel.js';
import {benchmarkCsv,benchmarkSeries} from './benchmarkReport.js';
import {sampleTimes,validateCrop} from './frameEvaluation.js';
import {boundaryOptions} from './boundaryOptions.js';
const $=id=>document.getElementById(id),specs=[
 {id:'unet-v1.1-current',kind:'learned',model:'models/tooth_seg.onnx',sha256:'a7fb02a9a95eeaf9d4fe3de86b7450ac89f8d5f750f3cfcd3eb7d5c0775fd198',provenance:'Shipped ToothNet-lite 1.1'},
 {id:'iop-mask-rcnn-local',kind:'iop',provenance:'Local IOP-Compass Mask R-CNN, view classifier and frozen postprocessing'},
 {id:'unet-v1.0-historical',selected:false,kind:'learned',model:'models/benchmarks/unet-v1.onnx',sha256:'c50072d07f1b44cfa53129e81b866b85e410922d52e4a50f653d9a80eee628d0',provenance:'Git dc01add ToothNet-lite 1.0'}];
let media,readSample,report,cancel=false,busy=false,videoUrl,videoHash,frameLabels=new Map();
const status=s=>$('status').textContent=s;
function listModels(){ $('models').replaceChildren();for(const [i,m]of specs.entries()){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=m.selected!==false;input.value=i;label.append(input,document.createTextNode(' '+m.id));$('models').append(label);} }
listModels();
const guarded=fn=>async()=>{try{await fn();}catch(e){status(e.message);}};
$('addModel').onclick=guarded(async()=>{const model=$('modelFile').files[0],file=$('cardFile').files[0];if(!model||!file)throw new Error('Choose both ONNX and its model card');const card=JSON.parse(await file.text());
 const kind=card.schema==='dental-segformer-onnx/1'?'segformer':card.name==='ToothNet-lite'?'learned':null;if(!kind)throw new Error('Unsupported model card');
 const id=model.name;if(specs.some(m=>m.id===id))throw new Error('Checkpoint name already added');
 specs.push({id,kind,model:URL.createObjectURL(model),info:URL.createObjectURL(file),card,provenance:'User supplied local checkpoint'});listModels();status('Checkpoint added.');});
$('checkIop').onclick=guarded(async()=>{status('Loading local IOP weights…');const model=await loadBenchmarkModel({kind:'iop'});$('iopPath').textContent=`Ready: ${model.card.modelPath} (${model.parameters.device}). Classifier: ${model.card.classifierPath}`;status('Local IOP is ready. Select learned U-Net and IOP, choose your images or video, then compare.');});
const hashBytes=async file=>{const bytes=new Uint8Array(await file.arrayBuffer());return {bytes,hash:await sha256(bytes)};};
function invalidate(){media=null;readSample=null;$('run').disabled=true;$('dataset').textContent='No evaluation samples selected.';}
function ready(){validateBenchmarkMedia(media);$('dataset').textContent=`${media.samples.length} samples. ${media.samples.filter(s=>s.timeMs!=null).length} video frames. Same inputs for all models.`;$('run').disabled=false;}
$('photo').accept=IMAGE_ACCEPT;
$('photo').onchange=$('photoGt').onchange=invalidate;
$('preparePhoto').onclick=guarded(async()=>{
 invalidate();const file=$('photo').files[0],gtFile=$('photoGt').files[0];if(!file)throw new Error('Choose a clean photo');
 const source=await readImageFile(file);let doc=gtFile?JSON.parse(await gtFile.text()):null,crop=[0,0,source.width,source.height];
 if(doc){
   if(doc.annotationStatus?.startsWith('unreviewed'))throw new Error('Review model-draft labels in the editor before using them as GT');
   if(doc.schema==='dental-targets/1'){crop=validateCrop(doc.crop,source.width,source.height,32_000_000);}
   else {const imported=importGroundTruth(doc,source.width,source.height,file.name);crop=imported.crop;
     if(doc.schema!=='tooth-labels/1')doc={schema:'tooth-labels/1',cleanImage:file.name,width:crop[2],height:crop[3],crop,labels:encodeLabels(imported.labels),vectorContours:imported.vectors};}
 }
 const s={id:file.name,cleanImage:file.name,group:file.name,split:'unassigned',width:crop[2],height:crop[3],crop};
 validateBenchmarkMedia({schema:'evaluation-media/1',samples:[s]});
 const rgba=source.getContext('2d').getImageData(...crop).data,rgb=new Uint8Array(s.width*s.height*3);
 for(let i=0;i<s.width*s.height;i++)rgb.set(rgba.subarray(i*4,i*4+3),i*3);
 source.width=source.height=1;
 const rgbSha256=await sha256(rgb),labelsSha256=gtFile?(await hashBytes(gtFile)).hash:null;
 media={schema:'evaluation-media/1',samples:[s]};readSample=async()=>({rgb,doc,rgbSha256,labelsSha256});ready();status('Photo ready for both models. Ground-truth JSON is required for accuracy scores.');
});
$('pack').onchange=guarded(async()=>{
 invalidate();
 const files=[...$('pack').files],manifests=files.filter(f=>f.name==='manifest.json');if(manifests.length!==1)throw new Error('Choose a folder containing exactly one manifest.json');
 const manifest=manifests[0],prefix=manifest.webkitRelativePath.slice(0,-manifest.name.length),map=new Map(files.map(f=>[f.webkitRelativePath,f]));
 const resolve=p=>{let f=map.get(prefix+p);if(!f){const hits=files.filter(f=>f.name===p.split('/').at(-1));if(hits.length===1)f=hits[0];}if(!f)throw new Error(`Missing or ambiguous pack file: ${p}`);return f;};
 const candidate=JSON.parse(await manifest.text());validateBenchmarkMedia(candidate);media=candidate;
 readSample=async s=>{const rgb=await hashBytes(resolve(s.rgb)),labels=s.labels?await hashBytes(resolve(s.labels)):null;
 return {rgb:rgb.bytes,doc:labels?JSON.parse(new TextDecoder().decode(labels.bytes)):null,rgbSha256:rgb.hash,labelsSha256:labels?.hash??null};};
 ready();status('Evaluation folder loaded.');});
const waitVideo=(event,action)=>new Promise((resolve,reject)=>{const v=$('source'),cleanup=()=>{clearTimeout(timer);v.removeEventListener(event,done);v.removeEventListener('error',fail);},done=()=>{cleanup();resolve();},fail=()=>{cleanup();reject(new Error('Video decoding failed'));},timer=setTimeout(()=>{cleanup();reject(new Error('Video decoding timed out'));},20000);v.addEventListener(event,done,{once:true});v.addEventListener('error',fail,{once:true});try{action();}catch(e){cleanup();reject(e);}});
$('videoFile').onchange=guarded(async()=>{const file=$('videoFile').files[0];if(!file)return;invalidate();videoHash=null;if(file.size>200e6)throw new Error('Use a video under 200 MB');const bytes=await hashBytes(file);if(videoUrl)URL.revokeObjectURL(videoUrl);videoUrl=URL.createObjectURL(file);await waitVideo('loadeddata',()=>{$('source').src=videoUrl;$('source').load();});videoHash=bytes.hash;status('Video loaded. Set a fixed mouth crop, then Use video sequence.');});
$('labels').onchange=guarded(async()=>{const nextLabels=new Map();for(const file of $('labels').files){const {bytes,hash}=await hashBytes(file),doc=JSON.parse(new TextDecoder().decode(bytes));if(!doc.frame)throw new Error(`${file.name} has no video identity metadata`);const key=JSON.stringify([doc.frame.videoSha256,doc.frame.timeMs,doc.frame.sourceCrop,doc.frame.captureProtocol]);if(nextLabels.has(key))throw new Error('Duplicate GT for a video frame');nextLabels.set(key,{doc,labelsSha256:hash});}frameLabels=nextLabels;status(`${frameLabels.size} frame annotations loaded.`);});
$('prepareVideo').onclick=guarded(async()=>{
 invalidate();
 if(!videoHash)throw new Error('Load a video first');const v=$('source'),crop=validateCrop([Number($('x').value),Number($('y').value),Number($('w').value)||v.videoWidth,Number($('h').value)||v.videoHeight],v.videoWidth,v.videoHeight);
 const times=sampleTimes(v.duration,Number($('fps').value),600);v.pause();
 media={schema:'evaluation-media/1',samples:times.map(timeMs=>({id:`video_${videoHash.slice(0,12)}_${timeMs}ms`,cleanImage:`video_${videoHash.slice(0,12)}_${timeMs}ms.png`,width:crop[2],height:crop[3],crop:[0,0,crop[2],crop[3]],sourceCrop:crop,group:videoHash,split:'unassigned',captureProtocol:'browser-seek/1',videoSha256:videoHash,timeMs}))};
 readSample=async s=>{v.pause();if(v.seeking)await waitVideo('seeked',()=>{});if(Math.abs(v.currentTime-s.timeMs/1000)>1e-7)await waitVideo('seeked',()=>{v.currentTime=s.timeMs/1000;});const c=document.createElement('canvas');c.width=s.width;c.height=s.height;const ctx=c.getContext('2d');ctx.drawImage(v,...crop,0,0,c.width,c.height);const rgba=ctx.getImageData(0,0,c.width,c.height).data,rgb=new Uint8Array(c.width*c.height*3);for(let i=0;i<c.width*c.height;i++)rgb.set(rgba.subarray(i*4,i*4+3),i*3);c.width=c.height=1;const key=JSON.stringify([s.videoSha256,s.timeMs,s.sourceCrop,s.captureProtocol]);return {rgb,rgbSha256:await sha256(rgb),...frameLabels.get(key)};};ready();status('Video sequence ready. No annotations means no accuracy scores.');});
function lock(value){busy=value;for(const e of document.querySelectorAll('input,button'))e.disabled=value;$('stop').disabled=!value;$('source').controls=!value;if(!value){$('run').disabled=!media;$('json').disabled=$('csv').disabled=!report;}}
$('run').onclick=guarded(async()=>{if(busy)return;const models=[...$('models').querySelectorAll('input:checked')].map(e=>specs[Number(e.value)]);if(!models.length)throw new Error('Select at least one checkpoint');const options=boundaryOptions({tolerancePx:Number($('tolerance').value)});lock(true);cancel=false;
 try{report=await runTargetBenchmark({media,models,readSample,loadModel:loadBenchmarkModel,boundaryOptions:options,onProgress:status,cancelled:()=>cancel});render();status(cancel?'Stopped. Partial report available.':'Comparison completed. Inspect target coverage and failed rows.');}finally{lock(false);}});
$('stop').onclick=()=>{cancel=true;status('Stopping after current sample…');};
const number=v=>Number.isFinite(v)?v.toPrecision(5):'—';
function row(parent,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value??'—';tr.append(td);}parent.append(tr);}
function render(){if(!report)return;const target=$('target').value;$('scope').textContent=report.scope;$('summary').replaceChildren();$('rows').replaceChildren();
 for(const m of report.models){const s=m.summary?.[target];row($('summary'),[m.id,s?`${s.labelled} / ${s.samples}`:'—',m.error??`${s?.unsupported??0} / ${s?.failed??0}`,number(s?.meanNormalizedBoundaryError),number(s?.meanHausdorffPx),number(s?.meanSdfCoverage),target==='teeth'?`${s?.missingTeeth??'—'} / ${s?.extraPredictions??'—'}`:'Not tooth instances',number(s?.meanInferenceMs)]);
 for(const r of m.rows.filter(r=>r.target===target))row($('rows'),[m.id,`${r.id}${r.timeMs!=null?` / ${r.timeMs/1000}s`:''}`,r.split,r.error??r.status,number(r.normalizedBoundaryError),number(r.boundary?.hausdorffPx),number(r.boundary?.sdfCoverage),number(r.temporalBoundaryChange)]);}
 drawPlots();$('json').disabled=$('csv').disabled=false;}
function drawPlots(){const series=benchmarkSeries(report,$('target').value,$('plotMetric').value),keys=[...new Set(series.map(s=>s.sequence))];$('plots').replaceChildren();
 if(!keys.length){$('plots').textContent='No video sequences in this report.';return;}
 const ns='http://www.w3.org/2000/svg',make=(name,attrs,text)=>{const e=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e;};
 for(const [idx,key]of keys.entries()){const group=series.filter(s=>s.sequence===key),points=group.flatMap(s=>s.paths.flat()),maxX=Math.max(1,...points.map(p=>p[0])),maxY=Math.max(.001,...points.map(p=>p[1])),svg=make('svg',{viewBox:'0 0 900 250',role:'img','aria-label':`Sequence ${idx+1} boundary metric`});svg.append(make('text',{x:20,y:20,fill:'white'},`Sequence ${idx+1} · 0–${(maxX/1000).toFixed(2)} s · max ${number(maxY)}`));
 group.forEach((s,i)=>{const color=['#38bdf8','#fbbf24','#e879f9','#4ade80'][i%4];for(const path of s.paths){svg.append(make('polyline',{points:path.map(([x,y])=>`${40+x/maxX*820},${200-y/maxY*155}`).join(' '),fill:'none',stroke:color}));for(const[x,y]of path)svg.append(make('circle',{cx:40+x/maxX*820,cy:200-y/maxY*155,r:2,fill:color}));}svg.append(make('text',{x:20+(i%3)*290,y:225+Math.floor(i/3)*18,fill:color},s.model));});if(!points.length)svg.append(make('text',{x:40,y:100,fill:'#b7c7d7'},'No matching labelled boundary scores'));$('plots').append(svg);}}
$('target').onchange=render;$('plotMetric').onchange=render;
$('reportFile').onchange=guarded(async()=>{const file=$('reportFile').files[0];if(!file)return;const doc=JSON.parse(await file.text());if(doc.schema!=='dental-target-benchmark/1'||!Array.isArray(doc.models))throw new Error('Choose a dental-target-benchmark/1 report');report=doc;render();});
function save(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('json').onclick=()=>save('dental-target-comparison.json',JSON.stringify(report,null,2),'application/json');$('csv').onclick=()=>save('dental-target-comparison.csv',benchmarkCsv(report),'text/csv');
