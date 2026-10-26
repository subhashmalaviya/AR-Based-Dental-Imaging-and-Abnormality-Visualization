import { decodeLabels } from './toothInstances.js';
import { traceLabelContours, validateVectorContours, vectorBoundaryDistance } from './vectorBoundary.js';

export function sampleTimes(duration,fps,maxFrames=600) {
  if(!Number.isFinite(duration)||duration<=0||![1,2].includes(fps))throw new Error('Use a finite video and 1 or 2 samples/sec');
  const count=Math.ceil(duration*fps);
  if(count>maxFrames)throw new Error(`Video exceeds ${maxFrames} samples; use a shorter clip`);
  return Array.from({length:count},(_,i)=>Math.round(i*1000/fps));
}
export function validateCrop(crop,w,h,maxPixels=4_000_000) {
  if(!Array.isArray(crop)||crop.length!==4||!crop.every(Number.isInteger)||crop[0]<0||crop[1]<0||crop[2]<8||crop[3]<8||crop[0]+crop[2]>w||crop[1]+crop[3]>h||crop[2]*crop[3]>maxPixels)
    throw new Error(`Choose a mouth crop inside the video (up to ${maxPixels/1_000_000} megapixels)`);
  return crop.slice();
}
export function frameTruth(doc,context,w,h) {
  if(doc?.schema!=='tooth-labels/1'||doc.annotationStatus?.startsWith('unreviewed'))throw new Error('Use manually reviewed editor labels');
  if(!doc.frame||doc.frame.captureProtocol!==context.captureProtocol||doc.frame.videoSha256!==context.videoSha256||doc.frame.timeMs!==context.timeMs
    ||JSON.stringify(doc.frame.sourceCrop)!==JSON.stringify(context.sourceCrop)
    ||doc.width!==w||doc.height!==h||JSON.stringify(doc.crop)!==JSON.stringify([0,0,w,h]))throw new Error('GT does not match this video, timestamp and crop');
  const labels=decodeLabels(doc.labels,w*h),overrides=validateVectorContours(doc.vectorContours??[],w,h,labels);
  const ids=new Set(overrides.map(c=>c.id));
  return {labels,contours:[...traceLabelContours(labels,w,h).filter(c=>!ids.has(c.id)),...overrides]};
}
export function evaluateVideoFrame(labels,w,h,{truth=null,previous=null}={}) {
  const contours=traceLabelContours(labels,w,h);
  const gt=truth?vectorBoundaryDistance(truth.contours,contours,w,h):null;
  const temporal=previous?vectorBoundaryDistance(previous,contours,w,h):null;
  return {contours,gt,temporal,normalizedBoundaryError:gt?.normalizedAssd??null,
    temporalBoundaryChange:temporal?.normalizedAssd??null,
    status:gt?.status??'unlabelled — temporal change only'};
}
