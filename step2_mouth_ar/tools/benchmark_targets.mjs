#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { runTargetBenchmark } from '../src/eval/targetBenchmark.js';
import { loadBenchmarkModel, sha256 } from '../src/eval/benchmarkModel.js';
import { boundaryOptions } from '../src/eval/boundaryOptions.js';
import { IopWorker } from '../scripts/iop-bridge.mjs';
import { loadIopBenchmarkModel } from '../src/eval/iopBenchmarkModel.js';
import { benchmarkCsv } from '../src/eval/benchmarkReport.js';
const [mediaFile,modelsFile,outFile]=process.argv.slice(2);
if(!outFile)throw new Error('Usage: node tools/benchmark_targets.mjs media/manifest.json models.json report.json');
const media=JSON.parse(await fs.readFile(mediaFile)),config=JSON.parse(await fs.readFile(modelsFile));
const root=path.dirname(mediaFile),modelRoot=path.dirname(modelsFile);
const models=await Promise.all(config.models.map(async m=>({...m,model:m.model?path.resolve(modelRoot,m.model):undefined,
  info:m.info?path.resolve(modelRoot,m.info):undefined,card:m.card?JSON.parse(await fs.readFile(path.resolve(modelRoot,m.card))):undefined})));
const report=await runTargetBenchmark({media,models,boundaryOptions:boundaryOptions(config.boundaryOptions),loadModel:async spec=>{if(spec.kind!=='iop')return loadBenchmarkModel(spec);const worker=new IopWorker({root:spec.root,python:spec.python});try{return await loadIopBenchmarkModel(spec,p=>worker.request(p),()=>worker.close());}catch(e){worker.close();throw e;}},
  onProgress:s=>console.log(s),readSample:async s=>{
    const rgb=await fs.readFile(path.resolve(root,s.rgb)),text=s.labels?await fs.readFile(path.resolve(root,s.labels)):null;
    return {rgb,doc:text?JSON.parse(text):null,rgbSha256:await sha256(rgb),labelsSha256:text?await sha256(text):null};
  }});
await fs.mkdir(path.dirname(outFile),{recursive:true});
await fs.writeFile(outFile,JSON.stringify(report,null,2));await fs.writeFile(outFile.replace(/\.json$/,'')+'.csv',benchmarkCsv(report));
console.log(JSON.stringify(report.models.map(m=>({id:m.id,error:m.error,summary:m.summary})),null,2));
if(report.models.some(m=>m.error||m.rows.some(r=>r.error)))process.exitCode=1;
