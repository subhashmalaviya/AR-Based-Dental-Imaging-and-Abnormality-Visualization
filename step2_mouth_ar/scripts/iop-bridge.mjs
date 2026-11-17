import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {join,resolve} from 'node:path';

export const defaultIopRoot=fileURLToPath(new URL('../../../IOP_teeth_segmentation/',import.meta.url));
export class IopWorker {
  constructor({root=process.env.DENTAL_IOP_ROOT??defaultIopRoot,python=process.env.DENTAL_IOP_PYTHON,timeoutMs=180000}={}) {
    this.root=resolve(root);this.python=python??join(this.root,'.venv/bin/python');this.timeoutMs=timeoutMs;
  }
  start(){
    if(this.child)return;
    const child=spawn(this.python,['-B','-u',fileURLToPath(new URL('./iop-worker.py',import.meta.url))],{
      env:{...process.env,DENTAL_IOP_ROOT:this.root,PYTHONDONTWRITEBYTECODE:'1'},stdio:['pipe','pipe','pipe']});
    this.child=child;this.stderr='';
    child.stderr.on('data',chunk=>{this.stderr=(this.stderr+chunk.toString()).slice(-3000);});
    const fail=message=>{if(this.child!==child)return;this.child=null;this.finish(new Error(message));};
    child.on('error',e=>fail(`Cannot start local IOP Python: ${e.message}. Set DENTAL_IOP_ROOT / DENTAL_IOP_PYTHON.`));
    child.on('exit',code=>fail(`Local IOP stopped (${code}): ${this.stderr}`));
    child.stdin.on('error',e=>fail(`IOP input failed: ${e.message}`));
    createInterface({input:child.stdout}).on('line',line=>{
      if(this.child!==child)return;
      try{const value=JSON.parse(line);this.finish(value.error?new Error(value.error):null,value.result);}
      catch{this.close();}
    });
  }
  finish(error,result){
    const pending=this.pending;this.pending=null;
    if(pending){clearTimeout(pending.timer);error?pending.reject(error):pending.resolve(result);}
  }
  request(payload){
    if(this.pending)return Promise.reject(new Error('IOP is busy; wait for the current frame'));
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>this.close('Local IOP inference timed out'),this.timeoutMs);
      this.pending={resolve,reject,timer};
      try{this.start();this.child.stdin.write(JSON.stringify(payload)+'\n');}catch(e){this.finish(e);}
    });
  }
  close(message='Local IOP worker closed'){
    const child=this.child;this.child=null;child?.kill();this.finish(new Error(message));
  }
}
export function iopMiddleware(worker){
  return async(req,res,next)=>{
    const path=new URL(req.url,'http://localhost').pathname;
    if(!path.startsWith('/__iop/'))return next();
    const json=(code,body)=>{res.statusCode=code;res.setHeader('Content-Type','application/json');res.setHeader('Cache-Control','no-store');res.end(JSON.stringify(body));};
    try{
      const local=['127.0.0.1','::1','::ffff:127.0.0.1'].includes(req.socket.remoteAddress);
      if(!local||req.headers['x-dental-client']!=='1'||(req.headers.origin&&new URL(req.headers.origin).host!==req.headers.host))return json(403,{error:'Local IOP is available only from this computer.'});
      if(path==='/__iop/info'&&req.method==='GET')return json(200,await worker.request({action:'info'}));
      if(path!=='/__iop/infer'||req.method!=='POST')return json(404,{error:'Unknown IOP operation'});
      let size=0;const chunks=[];
      for await(const chunk of req){size+=chunk.length;if(size>128_100_000)return json(413,{error:'IOP request exceeds 32 MP'});chunks.push(chunk);}
      const payload=JSON.parse(Buffer.concat(chunks).toString());
      return json(200,await worker.request({...payload,action:'infer'}));
    }catch(e){return json(503,{error:e.message});}
  };
}
export function iopBridge(){
  const configure=server=>{const worker=new IopWorker();server.middlewares.use(iopMiddleware(worker));server.httpServer?.once('close',()=>worker.close());};
  return {name:'local-iop-model',configureServer:configure,configurePreviewServer:configure};
}
