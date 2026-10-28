import { checkEditCrop } from './imageInput.js';
import { validateCrop } from '../eval/frameEvaluation.js';
import { evaluationDetector, inferCanvas } from '../eval/inferCanvas.js';
import { videoSchedule, drawTeethFrame, createVideoWriter } from '../eval/videoOutput.js';

export function plotSeries(rows,key,width=780,height=180) {
  const values=rows.map(r=>r[key]).filter(Number.isFinite),max=Math.max(.001,...values),duration=Math.max(1,...rows.map(r=>r.timeMs));
  const paths=[];let current=[];
  for(const r of rows) {
    if(!Number.isFinite(r[key])){if(current.length)paths.push(current);current=[];continue;}
    current.push([45+r.timeMs/duration*(width-60),height-25-r[key]/max*(height-45)]);
  }
  if(current.length)paths.push(current);
  return {paths,max,duration};
}
function waitEvent(target,event,action) {
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);target.removeEventListener(event,done);target.removeEventListener('error',fail);};
    const done=()=>{cleanup();resolve();},fail=()=>{cleanup();reject(new Error('Video decoding failed'));};
    const timer=setTimeout(()=>{cleanup();reject(new Error(`Timed out waiting for ${event}`));},15000);
    target.addEventListener(event,done,{once:true});target.addEventListener('error',fail,{once:true});
    try{action();}catch(e){cleanup();reject(e);}
  });
}
export class VideoEvaluation {
  constructor({root,modelUrl,getDetector,onAnnotate}) {
    Object.assign(this,{root,modelUrl,getDetector,onAnnotate});
    this.el=id=>root.querySelector(`#${id}`);this.video=this.el('evaluationVideo');
    this.el('videoUpload').addEventListener('change',()=>this.load());
    this.el('videoDetectionFrame').addEventListener('click',()=>this.extractDetection());
    this.el('videoDetectionDownload').addEventListener('click',()=>this.extractDetection(true));
    this.el('videoUseRegion').addEventListener('change',()=>{
      this.el('videoRegionPanel').hidden=!this.el('videoUseRegion').checked;this.video.pause();this.drawCrop();
    });
    this.el('videoRun').addEventListener('click',()=>this.run());
    this.el('videoStop').addEventListener('click',()=>{this.cancelled=true;this.status('Stopping after the current frame…');});
    this.el('videoFrameUse').addEventListener('click',()=>this.extract(false));
    this.el('videoFrameDownload').addEventListener('click',()=>this.extract(true));
    this.el('videoTime').addEventListener('change',()=>this.goTo(Number(this.el('videoTime').value)));
    this.el('videoPrevious').addEventListener('click',()=>this.goTo(this.video.currentTime-1/Number(this.el('videoFps').value)));
    this.el('videoNext').addEventListener('click',()=>this.goTo(this.video.currentTime+1/Number(this.el('videoFps').value)));
    this.el('videoWhole').addEventListener('click',()=>{this.el('videoUseRegion').checked=false;this.el('videoRegionPanel').hidden=true;this.crop=[0,0,this.video.videoWidth,this.video.videoHeight];this.drawCrop();});
    this.video.addEventListener('timeupdate',()=>{if(!this.busy)this.el('videoTime').value=this.video.currentTime.toFixed(3);});
    this.video.addEventListener('seeked',()=>{if(!this.busy)this.drawCrop();});
    this.video.addEventListener('play',()=>{if(this.busy)this.video.pause();});
    this.video.addEventListener('pause',()=>{if(!this.busy&&this.hash)this.drawCrop();});
    const c=this.el('videoCrop');
    const point=e=>{const r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,Math.floor((e.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.floor((e.clientY-r.top)*c.height/r.height)))];};
    c.addEventListener('pointerdown',e=>{if(this.busy||!this.hash||!this.el('videoUseRegion').checked)return;this.video.pause();this.drag=point(e);c.setPointerCapture(e.pointerId);});
    c.addEventListener('pointermove',e=>{if(!this.drag)return;const p=point(e),a=this.drag;
      const crop=[Math.min(a[0],p[0]),Math.min(a[1],p[1]),Math.abs(p[0]-a[0]),Math.abs(p[1]-a[1])];
      if(crop[2]>=8&&crop[3]>=8){this.crop=crop;this.drawCrop();}});
    const end=()=>{this.drag=null;};c.addEventListener('pointerup',end);c.addEventListener('pointercancel',end);
    this.lock(false);
  }
  status(s){this.el('videoStatus').textContent=s;}
  lock(value){
    this.busy=value;
    this.root.querySelectorAll('input,select,button').forEach(e=>e.disabled=value||(!this.hash&&e.id!=='videoUpload'));
    this.el('videoStop').disabled=!value||!this.running;
    this.video.controls=!value;
    this.syncSnapshotButton();
  }
  syncSnapshotButton() {
    for(const id of ['videoDetectionFrame','videoDetectionDownload'])this.el(id).disabled=!this.processedFrame||!!this.snapshotBusy;
  }
  selectedCrop() {
    return this.el('videoUseRegion').checked ? this.crop.slice() : [0,0,this.video.videoWidth,this.video.videoHeight];
  }
  resetProcessedFrame() {
    this.processedFrame=null;this.el('videoProcessingPreview').hidden=true;
    this.el('videoProcessedTime').textContent='';
    this.el('videoSnapshotStatus').textContent='Extract the displayed detection frame directly into ground truth comparison, then draw vector GT borders against its saved prediction.';
    this.syncSnapshotButton();
  }
  async extractDetection(download=false) {
    if(!this.processedFrame||this.snapshotBusy)return;
    // Keep the last completed inference frame, not the video element (which may
    // already be seeking ahead). Never infer again or turn predictions into GT.
    const metadata={...this.processedFrame};
    this.snapshotBusy=true;this.syncSnapshotButton();
    try {
      if(!download) {
        checkEditCrop([0, 0, metadata.clean.width, metadata.clean.height]);
        const clean=document.createElement('canvas');clean.width=metadata.clean.width;clean.height=metadata.clean.height;
        clean.getContext('2d').drawImage(metadata.clean,0,0);
        const context={captureProtocol:'browser-seek/1',videoSha256:metadata.hash,timeMs:metadata.timeMs,sourceCrop:metadata.crop.slice()};
        this.onAnnotate(clean,`video_${metadata.hash.slice(0,12)}_${metadata.timeMs}ms.png`,context,
          {labels:metadata.labels.slice(),runInfo:structuredClone(metadata.runInfo)});
        this.el('videoSnapshotStatus').textContent=`Opened detection frame at ${(metadata.timeMs/1000).toFixed(3)} s in Your ground truth comparison. Orange vector borders show this exact video prediction; draw your GT borders to compare.`;
        return;
      }
      // PNG export is optional. Copy pixels before asynchronous encoding.
      const preview=this.el('videoProcessingPreview'),copy=document.createElement('canvas');
      copy.width=preview.width;copy.height=preview.height;copy.getContext('2d').drawImage(preview,0,0);
      const blob=await new Promise(resolve=>copy.toBlob(resolve,'image/png'));
      if(!blob)throw new Error('Could not export detection frame');
      const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;
      a.download=`video_${metadata.hash.slice(0,12)}_${metadata.timeMs}ms_teeth-detection.png`;
      a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      this.el('videoSnapshotStatus').textContent=`Saved model detection image at ${(metadata.timeMs/1000).toFixed(3)} s (${metadata.detector}).`;
    }catch(e){this.el('videoSnapshotStatus').textContent=e.message;}
    finally{this.snapshotBusy=false;this.syncSnapshotButton();}
  }
  async seek(ms) {
    const time=Math.max(0,Math.min(ms/1000,Math.max(0,this.video.duration-.000001)));
    if(this.video.seeking)await waitEvent(this.video,'seeked',()=>{});
    if(Math.abs(this.video.currentTime-time)<.0000001&&this.video.readyState>=2)return;
    await waitEvent(this.video,'seeked',()=>{this.video.currentTime=time;});
    if(this.video.readyState<2)await waitEvent(this.video,'loadeddata',()=>{});
  }
  async goTo(seconds) {
    if(this.busy||!this.hash)return;this.video.pause();this.lock(true);
    try{if(!Number.isFinite(seconds))throw new Error('Enter a time in seconds');await this.seek(seconds*1000);this.drawCrop();}
    catch(e){this.status(e.message);}finally{this.lock(false);}
  }
  clearOutput(){
    this.el('videoOutputPanel').hidden=true;const output=this.el('videoOutput');output.pause();output.removeAttribute('src');output.load();
    if(this.outputUrl)URL.revokeObjectURL(this.outputUrl);this.outputUrl=null;
    this.el('videoDownload').removeAttribute('href');
  }
  async load(file = this.el('videoUpload').files[0]) {
    if (!file) return;
    if (this.busy) throw new Error('Wait for the current video operation to finish');
    this.video.pause();this.lock(true);this.hash=null;this.clearOutput();this.resetProcessedFrame();this.el('videoSourcePanel').hidden=true;
    try {
      if(file.size>200_000_000)throw new Error('Use a video smaller than 200 MB');
      if(this.url)URL.revokeObjectURL(this.url);this.url=URL.createObjectURL(file);
      this.status('Opening video…');
      this.el('videoUpload').value = '';
      await waitEvent(this.video,'loadeddata',()=>{this.video.src=this.url;this.video.load();});
      // Some MediaRecorder WebM files omit duration metadata until the end is sought.
      if (!Number.isFinite(this.video.duration)) {
        await waitEvent(this.video, 'durationchange', () => { this.video.currentTime = 1e10; });
        await this.seek(0);
      }
      videoSchedule(this.video.duration,30);
      if(this.video.videoWidth*this.video.videoHeight>8_300_000)throw new Error('Use video up to 4K');
      this.hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
      this.el('videoUseRegion').checked=false;this.el('videoRegionPanel').hidden=true;
      this.filename=file.name;this.crop=[0,0,this.video.videoWidth,this.video.videoHeight];
      this.el('videoTime').max=this.video.duration;this.drawCrop();this.el('videoSourcePanel').hidden=false;
      this.status('Ready. Process the video, or pause at any time and use that clean frame for comparison.');
    }catch(e){this.hash=null;this.status(e.message);return false;}finally{this.lock(false);}
    return true;
  }
  drawCrop() {
    if(!this.crop)return;
    const c=this.el('videoCrop');c.width=this.video.videoWidth;c.height=this.video.videoHeight;
    const ctx=c.getContext('2d');ctx.drawImage(this.video,0,0);ctx.strokeStyle='#38bdf8';ctx.lineWidth=Math.max(1,c.width/500);if(this.el('videoUseRegion').checked)ctx.strokeRect(...this.crop);
    this.el('videoTime').value=this.video.currentTime.toFixed(3);
    this.el('videoCropNote').textContent=`Region: ${this.selectedCrop().join(', ')} (x, y, width, height in original pixels). Used for detection and extracted frames.`;
  }
  capture(crop) {
    const c=document.createElement('canvas');c.width=crop[2];c.height=crop[3];c.getContext('2d').drawImage(this.video,...crop,0,0,c.width,c.height);return c;
  }
  async run() {
    if(!this.hash||this.busy)return;
    const originalTime=this.video.currentTime;this.video.pause();
    let detector,writer;this.running=true;this.lock(true);this.cancelled=false;this.clearOutput();this.resetProcessedFrame();
    const preview=this.el('videoProcessingPreview');preview.hidden=false;
    try {
      const crop=validateCrop(this.selectedCrop(),this.video.videoWidth,this.video.videoHeight,8_300_000),fps=Number(this.el('videoFps').value);
      const schedule=videoSchedule(this.video.duration,fps),selection=this.getDetector();
      preview.width=this.video.videoWidth;preview.height=this.video.videoHeight;
      preview.getContext('2d').drawImage(this.video,0,0);
      writer=await createVideoWriter(preview,fps);
      detector=evaluationDetector(selection,this.modelUrl);this.status('Loading the selected teeth model…');await detector.init?.();detector.setParams?.(selection.params);
      for(let i=0;i<schedule.length;i++) {
        if(this.cancelled)break;
        const {timestamp,duration}=schedule[i];
        await this.seek(timestamp/1000);const clean=this.capture(crop);
        const {labels,inputSize}=await inferCanvas(detector,clean);
        if(this.cancelled)break;
        drawTeethFrame(preview,this.video,labels,crop);
        this.processedFrame={timeMs:timestamp/1000,hash:this.hash,detector:selection.key,clean,labels,crop:crop.slice(),
          runInfo:{detector:selection.key,parameters:structuredClone(detector.params??selection.params??{}),modelInput:inputSize,
            source:'video prediction snapshot',videoTimeMs:timestamp/1000}};
        this.el('videoProcessedTime').textContent=`Displayed model frame: ${(timestamp/1e6).toFixed(3)} s`;
        this.syncSnapshotButton();
        await writer.add(timestamp,duration);
        this.status(`Processing ${i+1}/${schedule.length} frames (${Math.round((i+1)/schedule.length*100)}%). Model: ${selection.key}.`);
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      if(this.cancelled){this.status('Processing stopped. No incomplete video was exported.');return;}
      this.status('Finishing the output video…');
      const blob=await writer.finish(Math.round(this.video.duration*1e6));
      if(this.cancelled){this.status('Processing stopped.');return;}
      this.outputUrl=URL.createObjectURL(blob);this.el('videoOutput').src=this.outputUrl;
      const link=this.el('videoDownload');link.href=this.outputUrl;link.download=`${this.filename.replace(/\.[^.]+$/,'')}-teeth.webm`;
      this.el('videoOutputPanel').hidden=false;
      this.status(`Finished: ${schedule.length} frames at ${fps} FPS. Play or download your teeth-detection video below. Output is silent.`);
    }catch(e){this.status(`Video processing failed: ${e.message}`);}
    finally{
      writer?.close();detector?.dispose?.();preview.hidden=!this.processedFrame;
      try{await this.seek(originalTime*1000);this.drawCrop();}catch{/* Keep the processing error if source decoding failed. */}
      this.running=false;this.lock(false);
    }
  }
  async extract(download) {
    if(this.busy||!this.hash)return;
    this.video.pause();this.lock(true);
    try {
      if(this.video.seeking)await waitEvent(this.video,'seeked',()=>{});
      // Canonical microsecond time makes JSON saved here reproducible on extraction.
      const timeMs=Math.round(this.video.currentTime*1e6)/1000;
      await this.seek(timeMs);
      const crop=validateCrop(this.selectedCrop(),this.video.videoWidth,this.video.videoHeight,8_300_000);
      const frame=this.capture(crop),name=`video_${this.hash.slice(0,12)}_${timeMs}ms.png`;
      if(download){
        const blob=await new Promise(resolve=>frame.toBlob(resolve,'image/png'));
        if(!blob)throw new Error('Could not export frame');
        const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        this.status('Clean frame downloaded. Mark this exact image to upload it as matching ground truth.');
      }else{
        checkEditCrop([0, 0, frame.width, frame.height]);
        this.onAnnotate(frame,name,{captureProtocol:'browser-seek/1',videoSha256:this.hash,timeMs,sourceCrop:crop.slice()});
        this.status(`Opened the clean frame at ${(timeMs/1000).toFixed(3)} s in image comparison. Draw GT or upload a matching hand-marked image, then Detect & compare.`);
      }
    }catch(e){this.status(e.message);}finally{this.drawCrop();this.lock(false);}
  }
}
