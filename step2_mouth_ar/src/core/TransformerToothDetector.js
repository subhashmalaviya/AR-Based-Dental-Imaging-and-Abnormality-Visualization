import { ToothDetector, registerDetector } from './ToothDetector.js';
import { loadOrt, readModel } from './LearnedToothDetector.js';
import { traceLabelContours } from '../eval/vectorBoundary.js';
import { instanceStats } from '../eval/toothInstances.js';

export function validateTransformerCard(c) {
  if(c?.schema!=='dental-segformer-onnx/1'||c.architecture!=='segformer'||c.modality!=='rgb-intraoral')
    throw new Error('Requires a SegFormer card for RGB intraoral photos');
  const {width:w,height:h,mean,std}=c.input??{};
  if(![w,h].every(v=>Number.isInteger(v)&&v>=32&&v<=1024)||!Array.isArray(mean)||!Array.isArray(std)
    ||mean.length!==3||std.length!==3||!mean.every(Number.isFinite)||!std.every(v=>Number.isFinite(v)&&v>0))
    throw new Error('Invalid Transformer input size or normalization');
  if(!Number.isInteger(c.classes)||c.classes<2||c.classes>256||!Array.isArray(c.toothClasses)||!c.toothClasses.length||c.toothClasses.length>=c.classes
    ||new Set(c.toothClasses).size!==c.toothClasses.length||!c.toothClasses.every(v=>Number.isInteger(v)&&v>=0&&v<c.classes))
    throw new Error('Explicit tooth class IDs are required');
  if(!Number.isInteger(c.minAreaPixels)||c.minAreaPixels<1||c.minAreaPixels>w*h)throw new Error('Invalid component area threshold');
  return c;
}

// Standard SegFormer output: [1, classes, height/4, width/4] raw logits.
// Bilinear resize logits FIRST (align_corners=false), then argmax/softmax.
export function decodeSegformer(tensor,card,aperture) {
  validateTransformerCard(card);
  const W=card.input.width,H=card.input.height,C=card.classes,[batch,channels,h,w]=tensor.dims;
  if(batch!==1||channels!==C||![w,h].every(v=>Number.isInteger(v)&&v>0)||tensor.data.length!==C*w*h)
    throw new Error('Expected NCHW SegFormer logits with the configured class count');
  if(aperture.length!==W*H)throw new Error('Aperture dimensions differ');
  const classes=new Int32Array(W*H).fill(-1),prob=new Float32Array(W*H), tooth=new Set(card.toothClasses);
  const logits=new Float64Array(C), at=(c,x,y)=>tensor.data[c*w*h+Math.max(0,Math.min(h-1,y))*w+Math.max(0,Math.min(w-1,x))];
  for(let y=0;y<H;y++)for(let x=0;x<W;x++) {
    const i=y*W+x;if(!aperture[i])continue;
    const u=(x+.5)*w/W-.5,v=(y+.5)*h/H-.5,x0=Math.floor(u),y0=Math.floor(v),fx=u-x0,fy=v-y0;
    let best=0;
    for(let c=0;c<C;c++) {
      logits[c]=(1-fy)*((1-fx)*at(c,x0,y0)+fx*at(c,x0+1,y0))+fy*((1-fx)*at(c,x0,y0+1)+fx*at(c,x0+1,y0+1));
      if(!Number.isFinite(logits[c]))throw new Error('Non-finite Transformer logits');
      if(logits[c]>logits[best])best=c;
    }
    if(!tooth.has(best))continue;
    let sum=0;for(let c=0;c<C;c++)sum+=Math.exp(logits[c]-logits[best]);
    classes[i]=best;prob[i]=1/sum;
  }
  const labels=new Int32Array(W*H),seen=new Uint8Array(W*H),queue=new Int32Array(W*H);let id=0;
  const classById={};
  for(let i=0;i<labels.length;i++)if(classes[i]>=0&&!seen[i]) {
    let head=0,tail=1;queue[0]=i;seen[i]=1;
    while(head<tail) {
      const p=queue[head++],x=p%W,y=Math.floor(p/W);
      for(const q of [x? p-1:-1,x<W-1?p+1:-1,y?p-W:-1,y<H-1?p+W:-1])
        if(q>=0&&!seen[q]&&classes[q]===classes[i]){seen[q]=1;queue[tail++]=q;}
    }
    if(tail<card.minAreaPixels)continue;
    if(id>=256)throw new Error('More than 256 predicted regions; increase the model card minimum area');
    classById[++id]=classes[i];for(let k=0;k<tail;k++)labels[queue[k]]=id;
  }
  return {labels,prob,classById};
}

export class TransformerToothDetector extends ToothDetector {
  constructor({modelUrl,card,wasmBase}={}) {super();Object.assign(this,{modelUrl,card,wasmBase,ready:false,params:{},lastDebug:null});}
  get name(){return this.card?.name??'SegFormer candidate';}
  get isLearnedModel(){return true;}
  get isAsync(){return true;}
  get preferredRoiSize(){return {width:this.inputWidth,height:this.inputHeight,padding:.16,padTop:.16};}
  setParams() {} // The model card fixes preprocessing/decoding for fair comparisons.
  async init() {
    if(this.ready)return;
    this.info=validateTransformerCard(this.card);this.inputWidth=this.card.input.width;this.inputHeight=this.card.input.height;
    this.ort=await loadOrt('wasm',this.wasmBase);
    const bytes=await readModel(this.modelUrl);
    if(this.card.sha256) {
      const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(sha!==this.card.sha256.toLowerCase())throw new Error('ONNX file does not match the model card hash');
    }
    this.session=await this.ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
    try {
      this.inputName=this.session.inputNames[0];this.outputName=this.session.outputNames[0];
      await this.predict({width:this.inputWidth,height:this.inputHeight,data:new Uint8ClampedArray(this.inputWidth*this.inputHeight*4)},new Uint8Array(this.inputWidth*this.inputHeight).fill(255));
      this.backend='wasm';this.ready=true;
    }catch(e){await this.session.release();this.session=null;throw e;}
  }
  async predict(image,aperture) {
    const W=this.inputWidth,H=this.inputHeight,n=W*H;
    if(image.width!==W||image.height!==H)throw new Error('Transformer image dimensions differ');
    const data=new Float32Array(3*n),{mean,std}=this.card.input;
    for(let k=0;k<3;k++)for(let i=0;i<n;i++)data[k*n+i]=(image.data[4*i+k]/255-mean[k])/std[k];
    const start=performance.now(),outputs=await this.session.run({[this.inputName]:new this.ort.Tensor('float32',data,[1,3,H,W])});
    this.lastInferenceMs=performance.now()-start;
    return decodeSegformer(outputs[this.outputName],this.card,aperture);
  }
  async detectAsync(image,aperture,roi) {
    if(!this.ready)throw new Error('Transformer is not loaded');
    const {labels,prob,classById}=await this.predict(image,aperture),W=this.inputWidth,H=this.inputHeight;
    const contours=traceLabelContours(labels,W,H),stats=instanceStats(labels,W);
    const confidence=new Map();for(let i=0;i<labels.length;i++)if(labels[i])confidence.set(labels[i],(confidence.get(labels[i])??0)+prob[i]);
    const detections=stats.map(s=>{
      const area=c=>Math.abs(c.points.reduce((sum,p,i)=>{const q=c.points[(i+1)%c.points.length];return sum+p[0]*q[1]-p[1]*q[0]},0));
      const loops=contours.filter(c=>c.id===s.id),main=loops.sort((a,b)=>area(b)-area(a))[0];
      const contour=main.points.map(p=>roi.roiToLocal(...p)),center=roi.roiToLocal(s.x+.5,s.y+.5);
      const us=contour.map(p=>p.u),vs=contour.map(p=>p.v);
      return {center,contour,box:{u:Math.min(...us),v:Math.min(...vs),w:Math.max(...us)-Math.min(...us),h:Math.max(...vs)-Math.min(...vs)},
        area:s.area,confidence:confidence.get(s.id)/s.area,arch:s.y<H/2?'upper':'lower',classId:classById[s.id],
        visibility:'full',mask:{roiWidth:W,roiHeight:H,label:s.id}};
    });
    this.lastDebug={width:W,height:H,labels,toothCount:stats.length,inferenceMs:this.lastInferenceMs,maps:{teeth:prob,center:new Float32Array(W*H),boundary:new Float32Array(W*H)}};
    return detections;
  }
  dispose(){this.session?.release();this.session=null;this.ready=false;}
}
registerDetector('transformer',opts=>new TransformerToothDetector(opts),{label:'SegFormer candidate (local ONNX)',learned:true});
