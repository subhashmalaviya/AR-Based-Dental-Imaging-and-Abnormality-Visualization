import test from 'node:test';
import assert from 'node:assert/strict';
import {loadIopBenchmarkModel,rgbBase64} from '../src/eval/iopBenchmarkModel.js';
import {iopMiddleware,IopWorker,defaultIopRoot} from '../scripts/iop-bridge.mjs';
import {Readable} from 'node:stream';

test('IOP adapter sends exact native RGB and preserves instance labels',async()=>{
 const rgb=Uint8Array.from([0,128,255,255,1,4]);const calls=[];
 const model=await loadIopBenchmarkModel({},async p=>{calls.push(p);return p.action==='info'?{targets:['teeth'],sha256:'hash'}:{width:2,height:1,labels:[[1,1],[2,1]]};});
 const result=await model.infer(rgb,{width:2,height:1});
 assert.deepEqual([...result.teeth],[1,2]);assert.equal(result.gums,undefined);
 assert.deepEqual([...Buffer.from(calls[1].rgb,'base64')],[...rgb]);assert.equal(calls[1].width,2);
 assert.equal(rgbBase64(new Uint8Array(100000)),Buffer.alloc(100000).toString('base64'));
 await assert.rejects(()=>model.infer(rgb,{width:1,height:1}),/Invalid RGB/);
});
test('IOP adapter rejects wrong weight identity or output size',async()=>{
 await assert.rejects(()=>loadIopBenchmarkModel({sha256:'expected'},async()=>({sha256:'other'})),/hash mismatch/);
 const model=await loadIopBenchmarkModel({},async p=>p.action==='info'?{}:{width:3,height:1,labels:[[0,3]]});
 await assert.rejects(()=>model.infer(new Uint8Array(6),{width:2,height:1}),/mismatched/);
});
async function request({path='/__iop/info',address='127.0.0.1',headers={},method='GET',body=''}={}){
 let calls=0,code,value;
 const handler=iopMiddleware({request:async p=>{calls++;return {action:p.action};}});
 const req=Readable.from([Buffer.from(body)]);Object.assign(req,{url:path,method,headers:{host:'localhost:5173','x-dental-client':'1',...headers},socket:{remoteAddress:address}});
 const res={setHeader(){},set statusCode(v){code=v;},end(text){value=JSON.parse(text);}};
 await handler(req,res,()=>{});return {calls,code,value};
}
test('IOP route permits local requests, rejects cross-origin/LAN, fixes operation from URL',async()=>{
 assert.equal((await request()).code,200);
 assert.equal((await request({address:'192.168.1.12'})).code,403);
 assert.equal((await request({headers:{origin:'https://foreign.example'}})).calls,0);
 assert.equal((await request({headers:{'x-dental-client':undefined}})).code,403);
 const r=await request({path:'/__iop/infer',method:'POST',body:JSON.stringify({action:'info'})});assert.equal(r.value.action,'infer');
 assert.equal((await request({path:'/__iop/delete',method:'POST'})).code,404);
});
test('missing Python has an actionable error and worker can be closed safely',async()=>{
 assert.match(defaultIopRoot,/IOP_teeth_segmentation/);
 const worker=new IopWorker({python:'/nonexistent/python',timeoutMs:2000});
 try{await assert.rejects(()=>worker.request({action:'info'}),/Cannot start/);}finally{worker.close();}
});

test('native patient photo dimensions above 4 MP are valid benchmark inputs',async()=>{
 const {validateBenchmarkMedia}=await import('../src/eval/targetBenchmark.js');
 const sample={id:'patient',group:'patient',split:'test',width:2896,height:1944};
 assert.doesNotThrow(()=>validateBenchmarkMedia({schema:'evaluation-media/1',samples:[sample]}));
 assert.throws(()=>validateBenchmarkMedia({schema:'evaluation-media/1',samples:[{...sample,width:33000,height:1000}]}),/maximum 32 MP/);
});
