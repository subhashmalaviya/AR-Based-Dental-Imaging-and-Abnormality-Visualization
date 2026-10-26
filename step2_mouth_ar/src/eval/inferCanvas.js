import { LearnedToothDetector } from '../core/LearnedToothDetector.js';
import { TransformerToothDetector } from '../core/TransformerToothDetector.js';
import { ToothSegmenter } from '../core/ToothSegmenter.js';
import { resizeInstanceLabels } from './toothInstances.js';
export function evaluationDetector(selection,modelUrl) {
  if(selection.key==='transformer')return new TransformerToothDetector(selection.options);
  if(selection.key==='learned')return new LearnedToothDetector({modelUrl});
  if(selection.key==='classical')return new ToothSegmenter();
  throw new Error('Unsupported detector');
}
export async function inferCanvas(detector,input) {
  const W=detector.inputWidth??192,H=detector.inputHeight??144;
  const c=document.createElement('canvas');c.width=W;c.height=H;const ctx=c.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(input,0,0,W,H);
  const roi={width:W,height:H,bounds:{u0:0,u1:1,v0:0,v1:1},roiToLocal:(x,y)=>({u:x/W,v:y/H})};
  const image=ctx.getImageData(0,0,W,H),aperture=new Uint8Array(W*H).fill(255),start=performance.now();
  let labels;
  if(detector.isAsync){await detector.detectAsync(image,aperture,roi);labels=detector.lastDebug.labels;}
  else {
    labels=new Int32Array(W*H);
    for(const [index,d] of detector.detect(image,aperture,roi).entries()) {
      ctx.clearRect(0,0,W,H);ctx.beginPath();d.contour.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](p.u*W,p.v*H));ctx.closePath();ctx.fill();
      const pixels=ctx.getImageData(0,0,W,H).data;for(let i=0;i<labels.length;i++)if(pixels[4*i+3]>=128)labels[i]=index+1;
    }
  }
  return {labels:resizeInstanceLabels(labels,W,H,input.width,input.height),elapsedMs:performance.now()-start,inputSize:[W,H]};
}
