import {decodeLabels} from './toothInstances.js';
export function rgbBase64(bytes){
  // Chunked conversion avoids JS argument limits for native-resolution frames.
  let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return btoa(binary);
}
export async function requestIop(payload){
  const response=await fetch(payload.action==='info'?'/__iop/info':'/__iop/infer',{
    method:payload.action==='info'?'GET':'POST',headers:{'X-Dental-Client':'1','Content-Type':'application/json'},
    ...(payload.action==='info'?{}:{body:JSON.stringify(payload)})});
  if(!response.headers.get('Content-Type')?.includes('application/json'))throw new Error('Local IOP requires npm run dev or npm run preview on this computer. It cannot run on a static hosted page.');
  const value=await response.json();if(!response.ok)throw new Error(value.error??'Local IOP failed');return value;
}
export async function loadIopBenchmarkModel(spec,request=requestIop,dispose=async()=>{}){
  const info=await request({action:'info'});
  if(spec.sha256&&spec.sha256!==info.sha256)throw new Error('IOP checkpoint hash mismatch');
  return {...info,async infer(rgb,s){
    if(rgb.length!==s.width*s.height*3)throw new Error('Invalid RGB input');
    const result=await request({action:'infer',width:s.width,height:s.height,rgb:rgbBase64(rgb)});
    if(result.width!==s.width||result.height!==s.height)throw new Error('IOP returned mismatched dimensions');
    return {teeth:decodeLabels(result.labels,s.width*s.height)};
  },dispose};
}
