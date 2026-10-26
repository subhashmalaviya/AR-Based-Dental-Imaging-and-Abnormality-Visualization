import { traceLabelContours } from './vectorBoundary.js';

// Offline processing: timestamps follow the source timeline, never inference time.
export function videoSchedule(duration, fps) {
  if(!Number.isFinite(duration)||duration<=0||duration>600) throw new Error('Use a video up to 10 minutes long');
  if(![10,15,24,30].includes(fps)) throw new Error('Choose a supported output frame rate');
  return Array.from({length:Math.ceil(duration*fps)},(_,i)=>{
    const timestamp=Math.round(i*1e6/fps);
    return {timestamp,duration:Math.min(Math.round((i+1)*1e6/fps),Math.round(duration*1e6))-timestamp};
  });
}

export function drawTeethFrame(canvas, source, labels, crop) {
  const ctx=canvas.getContext('2d');
  ctx.drawImage(source,0,0,canvas.width,canvas.height);
  ctx.save();ctx.translate(crop[0],crop[1]);
  ctx.strokeStyle='#00e5ff';ctx.lineWidth=Math.max(1,canvas.width/640);ctx.lineJoin='round';
  for(const c of traceLabelContours(labels,crop[2],crop[3])) {
    ctx.beginPath();c.points.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](...p));ctx.closePath();ctx.stroke();
  }
  ctx.restore();
}

const concat=parts=>{
  const out=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let i=0;
  for(const p of parts){out.set(p,i);i+=p.length;}return out;
};
function uint(value, size=0) {
  if(!size)size=Math.max(1,Math.ceil(Math.log2(value+1)/8));
  const out=new Uint8Array(size);
  for(let i=size-1;i>=0;i--){out[i]=value%256;value=Math.floor(value/256);}return out;
}
function vint(value) {
  let size=1;while(value>=2**(7*size)-1)size++;
  const out=uint(value,size);out[0]|=1<<(8-size);return out;
}
const element=(id,data)=>concat([uint(id),vint(data.length),data]);
const num=(id,value)=>element(id,uint(value));
const str=(id,value)=>element(id,new TextEncoder().encode(value));
const master=(id,children)=>element(id,concat(children));
function float(value) {const a=new Uint8Array(8);new DataView(a.buffer).setFloat64(0,value);return a;}

// Minimal single-track VP8 WebM. Millisecond timecodes, duration and keyframe cues
// make the offline result playable and seekable. No audio or live AR layers.
export class TeethWebM {
  constructor(width,height,fps,maxBytes=250_000_000) {Object.assign(this,{width,height,fps,maxBytes,frames:[],bytes:0});}
  add(data,timestamp,key=true) {
    if(!Number.isFinite(timestamp)||timestamp<0 || (this.frames.length&&timestamp<=this.frames.at(-1).timestamp)) throw new Error('Video timestamps must increase');
    if(!this.frames.length&&!key)throw new Error('Video must start with a keyframe');
    this.bytes+=data.length;if(this.bytes>this.maxBytes)throw new Error('Output exceeds 250 MB; use a shorter video or lower output FPS');
    this.frames.push({data,timestamp,key});
  }
  finish(durationUs) {
    if(!this.frames.length)throw new Error('No processed frames');
    const header=master(0x1a45dfa3,[num(0x4286,1),num(0x42f7,1),num(0x42f2,4),num(0x42f3,8),str(0x4282,'webm'),num(0x4287,2),num(0x4285,2)]);
    const info=master(0x1549a966,[num(0x2ad7b1,1000000),str(0x4d80,'Dental AR'),str(0x5741,'Dental AR'),element(0x4489,float(durationUs/1000))]);
    const tracks=master(0x1654ae6b,[master(0xae,[num(0xd7,1),num(0x73c5,1),num(0x83,1),num(0x9c,0),str(0x86,'V_VP8'),num(0x23e383,Math.round(1e9/this.fps)),master(0xe0,[num(0xb0,this.width),num(0xba,this.height)])])]);
    const clusters=[],cues=[];let position=info.length+tracks.length;
    for(const f of this.frames) {
      const ms=Math.round(f.timestamp/1000);
      const cluster=master(0x1f43b675,[num(0xe7,ms),element(0xa3,concat([new Uint8Array([0x81,0,0,f.key?0x80:0]),f.data]))]);
      if(f.key)cues.push(master(0xbb,[num(0xb3,ms),master(0xb7,[num(0xf7,1),num(0xf1,position)])]));
      clusters.push(cluster);position+=cluster.length;
    }
    const cueElement=master(0x1c53bb6b,cues),size=position+cueElement.length;
    return new Blob([header,uint(0x18538067),vint(size),info,tracks,...clusters,cueElement],{type:'video/webm'});
  }
}

// Canvas WebP provides an offline VP8 keyframe fallback when WebCodecs is absent.
export function vp8FromWebP(bytes) {
  const tag=(offset)=>String.fromCharCode(...bytes.subarray(offset,offset+4));
  if(tag(0)!=='RIFF'||tag(8)!=='WEBP')throw new Error('This browser cannot encode video. Open this page in Chrome or Edge.');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  for(let i=12;i+8<=bytes.length;) {
    const size=view.getUint32(i+4,true);
    if(i+8+size>bytes.length)break;
    if(tag(i)==='VP8 ')return bytes.slice(i+8,i+8+size);
    i+=8+size+(size%2);
  }
  throw new Error('Browser did not provide a lossy WebP frame; use Chrome or Edge for video export.');
}
const webpFrame=async canvas=>{
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/webp',.9));
  if(!blob)throw new Error('Could not encode video frame');
  return vp8FromWebP(new Uint8Array(await blob.arrayBuffer()));
};
export async function createVideoWriter(canvas,fps) {
  const mux=new TeethWebM(canvas.width,canvas.height,fps);
  const config={codec:'vp8',width:canvas.width,height:canvas.height,framerate:fps,bitrate:4_000_000,latencyMode:'realtime'};
  let encoder,error;
  if(globalThis.VideoEncoder && globalThis.VideoFrame) {
    let supported=false;try{supported=(await VideoEncoder.isConfigSupported(config)).supported;}catch{/* Use WebP fallback. */}
    if(supported) {
      encoder=new VideoEncoder({output:chunk=>{
        try{const data=new Uint8Array(chunk.byteLength);chunk.copyTo(data);mux.add(data,chunk.timestamp,chunk.type==='key');}catch(e){error=e;}
      },error:e=>{error=e;}});
      encoder.configure(config);
    }
  }
  // Check fallback capability before running the model on the entire video.
  if(!encoder)await webpFrame(canvas);
  let index=0;
  return {
    async add(timestamp,duration) {
      if(error)throw error;
      if(encoder) {
        const frame=new VideoFrame(canvas,{timestamp,duration});
        try{encoder.encode(frame,{keyFrame:index++%(fps*2)===0});}finally{frame.close();}
        if(encoder.encodeQueueSize>=4)await encoder.flush();
      } else mux.add(await webpFrame(canvas),timestamp,true);
      if(error)throw error;
    },
    async finish(duration) {if(encoder)await encoder.flush();if(error)throw error;return mux.finish(duration);},
    close(){if(encoder&&encoder.state!=='closed')encoder.close();mux.frames=[];}
  };
}
