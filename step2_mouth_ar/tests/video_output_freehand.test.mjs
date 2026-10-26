import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {freehandBorder} from '../src/eval/freehandBorder.js';
import {contourPath, rasterizeVector, validateVectorContours} from '../src/eval/vectorBoundary.js';
import {ToothMaskEditor} from '../src/ui/ToothMaskEditor.js';
import {VideoEvaluation} from '../src/ui/VideoEvaluation.js';
import {videoSchedule, TeethWebM, vp8FromWebP, drawTeethFrame, createVideoWriter} from '../src/eval/videoOutput.js';

const loop=Array.from({length:241},(_,i)=>[40+20*Math.cos(i*Math.PI/120),40+15*Math.sin(i*Math.PI/120)]);
test('continuous stroke becomes a compact closed curve and survives label serialization',()=>{
  const curve=freehandBorder(loop,7);assert.equal(curve.id,7);assert.equal(curve.smooth,true);
  assert.ok(curve.points.length<loop.length/2);assert.match(contourPath(curve),/^M.*C.*Z$/);
  const labels=new Int32Array(80*80);rasterizeVector(labels,80,80,curve);
  assert.equal(labels[40*80+40],7);assert.equal(labels[0],0);
  assert.deepEqual(validateVectorContours(JSON.parse(JSON.stringify([curve])),80,80,labels),[curve]);
  assert.equal(freehandBorder([[1,1],[1,1],[1,1],[1,1]],1),null);
  assert.equal(freehandBorder([[1,1],[2,2],[3,3],[4,4]],1),null);
});
function fakeElement() {
  return {events:{},value:'',style:{},children:[],addEventListener(n,f){this.events[n]=f;},
    replaceChildren(){this.children=[];},append(e){this.children.push(e);},setAttribute(n,v){this[n]=v;},
    setPointerCapture(){},hasPointerCapture(){return false;},getBoundingClientRect(){return {left:10,top:20,width:320,height:320};}};
}
test('freehand drag applies on release at zoomed coordinates; cancel and undo preserve prior labels',t=>{
  const old=globalThis.document;t.after(()=>{globalThis.document=old;});
  globalThis.document={createElement:()=>fakeElement(),createElementNS:()=>fakeElement()};
  const els=new Map(),el=id=>{if(!els.has(id))els.set(id,fakeElement());return els.get(id);};
  let changes=0;const editor=new ToothMaskEditor({querySelector:el},()=>changes++);
  editor.draw=()=>{};editor.refresh=()=>{};
  editor.setData(new Int32Array(80*80),{width:80,height:80},{});el('#toothEditTool').value='curve';
  const event=p=>({pointerId:0,button:0,clientX:10+p[0]*4,clientY:20+p[1]*4,preventDefault(){}});
  const canvas=el('#toothEditCanvas');
  const drag=()=>{canvas.events.pointerdown(event(loop[0]));for(const p of loop.slice(1))canvas.events.pointermove(event(p));};
  drag();assert.ok(editor.labels.every(v=>v===0));
  const svg=el('#toothEditVectors');assert.equal(svg.children.filter(e=>e.cx!=null).length,0,'freehand does not show control point dots');
  canvas.events.pointerup(event(loop.at(-1)));
  assert.equal(changes,1);const expected=freehandBorder(loop,1);assert.equal(editor.vectors[0].points.length,expected.points.length);
  editor.vectors[0].points.forEach((p,i)=>assert.ok(Math.hypot(p[0]-expected.points[i][0],p[1]-expected.points[i][1])<1e-10));
  const saved=editor.labels.slice();drag();canvas.events.pointercancel(event(loop.at(-1)));
  assert.deepEqual(editor.labels,saved);assert.equal(changes,1);
  drag();canvas.events.keydown({key:'Escape'});assert.deepEqual(editor.labels,saved);
  el('#toothEditUndo').events.click();assert.ok(editor.labels.every(v=>v===0));assert.equal(editor.vectors.length,0);
  el('#toothEditRedo').events.click();assert.deepEqual(editor.labels,saved);
  const original=structuredClone(editor.vectors[0]);
  el('#toothEditTool').value='vector-edit';
  const p=original.points[0],q=[p[0]-5,p[1]+3];
  canvas.events.pointerdown(event(p));canvas.events.pointermove(event(q));
  assert.deepEqual(editor.vectors[0],original,'point preview must not modify saved geometry');
  canvas.events.pointerup(event(q));
  assert.deepEqual(editor.vectors[0].points[0],q);const edited=editor.labels.slice();
  assert.notDeepEqual(edited,saved);
  assert.deepEqual(validateVectorContours(editor.vectors,80,80,editor.labels),editor.vectors);
  editor.undo();assert.deepEqual(editor.labels,saved);assert.deepEqual(editor.vectors[0],original);
  editor.redo();assert.deepEqual(editor.labels,edited);assert.deepEqual(editor.vectors[0].points[0],q);
  canvas.events.pointerdown(event(q));canvas.events.pointermove(event(p));canvas.events.keydown({key:'Escape'});
  assert.deepEqual(editor.labels,edited);assert.deepEqual(editor.vectors[0].points[0],q);
  editor.undo();el('#toothEditTool').value='curve';drag();canvas.events.pointerup(event(loop.at(-1)));
  assert.equal(editor.redoHistory.length,0,'a new edit clears the redo branch');
});

test('output schedule preserves source duration at 30 FPS independent of inference delay',()=>{
  const schedule=videoSchedule(19.3299,30);
  assert.equal(schedule.length,580);assert.equal(schedule[0].timestamp,0);
  assert.equal(schedule.at(-1).timestamp+schedule.at(-1).duration,19329900);
  assert.ok(schedule.every((f,i)=>f.duration>0&&(!i||f.timestamp===schedule[i-1].timestamp+schedule[i-1].duration)));
  assert.throws(()=>videoSchedule(Infinity,30));assert.throws(()=>videoSchedule(2,1));
});
test('output draws only source pixels and teeth contours at the selected crop offset',()=>{
  const operations=[],ctx=new Proxy({}, {get:(o,k)=>o[k]??((...args)=>operations.push([k,...args])),set:(o,k,v)=>(o[k]=v,true)});
  const source={width:100,height:80},canvas={...source,getContext:()=>ctx};
  drawTeethFrame(canvas,source,Int32Array.from([0,1,0,1]),[10,20,2,2]);
  assert.strictEqual(operations[0][1],source);assert.deepEqual(operations.find(o=>o[0]==='translate'),['translate',10,20]);
  assert.equal(operations.filter(o=>o[0]==='stroke').length,1);
  assert.ok(!operations.some(o=>['fillText','strokeText','fill','fillRect'].includes(o[0])));
});
test('any paused timestamp opens a clean crop without needing a video inference run',async()=>{
  const app=Object.create(VideoEvaluation.prototype),clean={},context={};
  Object.assign(app,{el:()=>({checked:true}),hash:'abcd1234567890',crop:[12,34,80,60],video:{currentTime:1.234567,seeking:false,videoWidth:200,videoHeight:100,pause(){}},
    lock(v){this.busy=v;},async seek(ms){context.seek=ms;},capture(c){context.crop=c;return Object.assign(clean,{width:80,height:60});},
    onAnnotate(frame,name,meta){context.frame=frame;context.name=name;context.meta=meta;},status(){},drawCrop(){}});
  await app.extract(false);
  assert.equal(context.seek,1234.567);assert.strictEqual(context.frame,clean);assert.deepEqual(context.crop,[12,34,80,60]);
  assert.equal(context.meta.timeMs,1234.567);assert.equal(context.meta.videoSha256,app.hash);assert.match(context.name,/1234.567ms/);
  assert.equal(app.busy,false);
});

test('WebM rejects missing keyframe, nonincreasing timestamps and oversized output',()=>{
  const mux=new TeethWebM(32,24,30,3);
  assert.throws(()=>mux.add(new Uint8Array([1]),0,false),/keyframe/);
  mux.add(new Uint8Array([1]),0,true);
  assert.throws(()=>mux.add(new Uint8Array([1]),0,true),/increase/);
  assert.throws(()=>mux.add(new Uint8Array(4),100,true),/250 MB/);
  assert.throws(()=>vp8FromWebP(new Uint8Array([1,2])),/cannot encode/);
});

const hasFFmpeg=spawnSync('ffmpeg',['-version']).status===0&&spawnSync('ffprobe',['-version']).status===0;
test('real VP8 frames mux to a decodable, seekable WebM with correct duration; WebP fallback decodes too',{skip:!hasFFmpeg},async t=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'dental-webm-'));t.after(()=>fs.rmSync(dir,{recursive:true,force:true}));
  const exec=(cmd,args)=>{const r=spawnSync(cmd,args,{maxBuffer:10_000_000});assert.equal(r.status,0,r.stderr?.toString());return r.stdout;};
  const ivf=exec('ffmpeg',['-v','error','-f','lavfi','-i','testsrc2=size=32x24:rate=10:duration=1','-c:v','libvpx','-g','5','-f','ivf','pipe:1']);
  const mux=new TeethWebM(32,24,10);let index=0;
  for(let i=32;i<ivf.length;){const size=ivf.readUInt32LE(i),data=ivf.subarray(i+12,i+12+size);mux.add(data,index++*100000,(data[0]&1)===0);i+=12+size;}
  assert.equal(index,10);
  const file=path.join(dir,'teeth.webm');fs.writeFileSync(file,new Uint8Array(await mux.finish(1e6).arrayBuffer()));
  const probe=JSON.parse(exec('ffprobe',['-v','error','-count_frames','-show_streams','-show_format','-of','json',file]));
  assert.equal(probe.streams[0].codec_name,'vp8');assert.equal(probe.streams[0].nb_read_frames,'10');assert.equal(Number(probe.format.duration),1);
  assert.equal(exec('ffmpeg',['-v','error','-i',file,'-f','rawvideo','-pix_fmt','rgb24','pipe:1']).length,32*24*3*10);
  assert.equal(exec('ffmpeg',['-v','error','-ss','0.6','-i',file,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']).length,32*24*3);
  const webp=exec('ffmpeg',['-v','error','-f','lavfi','-i','color=c=cyan:size=32x24','-frames:v','1','-c:v','libwebp','-lossless','0','-f','image2pipe','pipe:1']);
  const fallback=new TeethWebM(32,24,10);fallback.add(vp8FromWebP(webp),0);
  const single=path.join(dir,'fallback.webm');fs.writeFileSync(single,new Uint8Array(await fallback.finish(100000).arrayBuffer()));
  assert.equal(exec('ffmpeg',['-v','error','-i',single,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']).length,32*24*3);
});

test('writer sends timestamps to WebCodecs and always closes frames',async t=>{
  const oldEncoder=globalThis.VideoEncoder,oldFrame=globalThis.VideoFrame;
  t.after(()=>{globalThis.VideoEncoder=oldEncoder;globalThis.VideoFrame=oldFrame;});
  const frames=[];
  globalThis.VideoFrame=class {constructor(canvas,options){Object.assign(this,options);frames.push(this);}close(){this.closed=true;}};
  globalThis.VideoEncoder=class {
    static async isConfigSupported(){return {supported:true};}
    constructor(callbacks){this.callbacks=callbacks;this.state='configured';this.encodeQueueSize=0;}
    configure(){} encode(frame,{keyFrame}){this.callbacks.output({timestamp:frame.timestamp,type:keyFrame?'key':'delta',byteLength:1,copyTo(a){a[0]=0;}});}
    async flush(){}close(){this.state='closed';}
  };
  const writer=await createVideoWriter({width:20,height:10},30);
  await writer.add(0,33333);await writer.add(33333,33334);
  assert.deepEqual(frames.map(f=>f.timestamp),[0,33333]);assert.ok(frames.every(f=>f.closed));
  assert.ok((await writer.finish(66667)).size>0);writer.close();
});

test('extracted frame accepts a matching hand-marked upload without re-uploading the clean image',async t=>{
  const {UploadedPhotoComparison}=await import('../src/ui/UploadedPhotoComparison.js');
  const oldDoc=globalThis.document,oldImage=globalThis.Image;
  t.after(()=>{globalThis.document=oldDoc;globalThis.Image=oldImage;});
  const makeCanvas=()=>({width:80,height:60,getContext:()=>({drawImage(){}})});
  globalThis.document={createElement:()=>makeCanvas()};
  globalThis.Image=class {naturalWidth=80;naturalHeight=60;async decode(){}};
  const els=new Map(),el=id=>{if(!els.has(id))els.set(id,{...fakeElement(),...makeCanvas(),files:[]});return els.get(id);};
  const app=Object.create(UploadedPhotoComparison.prototype),clean=makeCanvas();
  Object.assign(app,{el,revision:0,editor:{setData(){}},drawSelection(){},updateComparison(){}});
  const frameContext={videoSha256:'abc',timeMs:1234,sourceCrop:[5,6,80,60]};
  app.loadAnnotationFrame(clean,'extracted.png',frameContext);
  assert.strictEqual(app.input,clean);assert.equal(el('uploadRun').disabled,false);
  el('uploadMarked').files=[Object.assign(new Blob(['image'],{type:'image/png'}),{name:'marked.png'})];
  await app.loadPair('uploadMarked');
  assert.strictEqual(app.clean,clean);assert.equal(app.cleanName,'extracted.png');assert.equal(app.markedName,'marked.png');
  assert.deepEqual(app.frameContext,frameContext);assert.equal(el('uploadPreview').disabled,false);
});

test('optional region selection defaults to whole frame and can be toggled without losing a selected crop',()=>{
  const app=Object.create(VideoEvaluation.prototype),checkbox={checked:false};
  Object.assign(app,{video:{videoWidth:200,videoHeight:120},crop:[10,20,50,60],el:()=>checkbox});
  assert.deepEqual(app.selectedCrop(),[0,0,200,120]);checkbox.checked=true;
  const crop=app.selectedCrop();assert.deepEqual(crop,[10,20,50,60]);crop[0]=99;
  assert.equal(app.crop[0],10,'processing receives a copy');checkbox.checked=false;
  assert.deepEqual(app.selectedCrop(),[0,0,200,120]);
});

test('detection PNG freezes displayed frame and timestamp while processing continues',async t=>{
  const old=globalThis.document;t.after(()=>{globalThis.document=old;});
  const elements={videoProcessingPreview:{width:20,height:10,pixels:'frame A'},videoDetectionFrame:{},videoDetectionDownload:{},videoSnapshotStatus:{}};
  let encode,download,copy;
  globalThis.document={createElement:tag=>{
    if(tag==='a')return {click(){download=this.download;}};
    copy={getContext:()=>({drawImage(source){copy.pixels=source.pixels;}}),toBlob(callback){encode=callback;}};return copy;
  }};
  const app=Object.create(VideoEvaluation.prototype);
  Object.assign(app,{busy:true,running:true,processedFrame:{timeMs:1234,hash:'abcdef0123456789',detector:'learned'},el:id=>elements[id]});
  const saving=app.extractDetection(true);
  assert.equal(copy.pixels,'frame A');assert.equal(elements.videoDetectionFrame.disabled,true);
  elements.videoProcessingPreview.pixels='frame B';app.processedFrame.timeMs=4567;
  encode(new Blob(['png']));await saving;
  assert.equal(copy.pixels,'frame A');assert.match(download,/1234ms_teeth-detection\.png$/);
  assert.match(elements.videoSnapshotStatus.textContent,/1\.234 s/);
  assert.equal(elements.videoDetectionFrame.disabled,false,'snapshot stays usable during inference');
  assert.equal(app.busy,true,'saving a prediction never unlocks processing controls');
});

test('extract detection opens the completed inference crop and exact labels without seeking or re-inference',async t=>{
  const old=globalThis.document;t.after(()=>{globalThis.document=old;});
  let captured,handed;
  globalThis.document={createElement:()=>({getContext:()=>({drawImage(source){captured=source;}})})};
  const elements={videoDetectionFrame:{},videoDetectionDownload:{},videoSnapshotStatus:{}};
  const app=Object.create(VideoEvaluation.prototype),clean={width:80,height:60,pixels:'completed frame'},labels=new Int32Array(80*60);labels[10]=4;
  const metadata={clean,labels,crop:[20,30,80,60],hash:'abc123',timeMs:2500,detector:'learned',runInfo:{detector:'learned',modelInput:[160,120]}};
  Object.assign(app,{busy:true,running:true,processedFrame:metadata,el:id=>elements[id],onAnnotate(...args){handed=args;},seek(){throw new Error('Must not seek');}});
  await app.extractDetection();assert.strictEqual(captured,clean);
  assert.equal(handed[1],'video_abc123_2500ms.png');assert.deepEqual(handed[2].sourceCrop,[20,30,80,60]);
  assert.deepEqual(handed[3].labels,labels);assert.notStrictEqual(handed[3].labels,labels);
  assert.deepEqual(handed[3].runInfo,metadata.runInfo);
  labels[10]=99;metadata.runInfo.modelInput[0]=10;
  assert.equal(handed[3].labels[10],4);assert.equal(handed[3].runInfo.modelInput[0],160);
  assert.equal(app.busy,true);assert.equal(app.snapshotBusy,false);
});
