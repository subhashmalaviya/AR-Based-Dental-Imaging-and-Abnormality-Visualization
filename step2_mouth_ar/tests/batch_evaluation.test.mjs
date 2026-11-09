import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:http';
import { pairPatientFiles, parsePatientTruth, scoreBatchPrediction, summarizeBatch } from '../src/eval/batchEvaluation.js';
import { createBatchWorkbook } from '../src/eval/batchWorkbook.js';
import { scanPatientFolder, patientFolderMiddleware } from '../scripts/patient-folder.mjs';
import { encodeLabels } from '../src/eval/toothInstances.js';

const square = [[1,1],[4,1],[4,4],[1,4]];
test('pairs are directory-specific; ambiguous and orphan entries remain explicit', () => {
  const rows = pairPatientFiles(['P1/a.png','P1/a.json','P2/a.PNG','P2/a.JSON','P3/b.png','P4/c.json','P5/d.png','P5/d.jpg','P5/d.json','notes.txt'].map(path => ({ path })));
  assert.equal(rows.length, 5); assert.equal(rows.filter(r => !r.issue).length, 2);
  assert.equal(rows[0].annotation.path, 'P1/a.json'); assert.equal(rows[1].annotation.path, 'P2/a.JSON');
  assert.match(rows[2].issue, /Missing ground/); assert.match(rows[3].issue, /Missing clean/); assert.match(rows[4].issue, /Ambiguous/);
});
test('patient contours preserve all rings without duplicating contour + contours aliases', () => {
  const data = { annotation_count: 1, teeth: [{ contour: square, contours: [square, square.map(([x,y]) => [x+5,y])] }] };
  const truth = parsePatientTruth(data, 10, 10, 'a.png');
  assert.equal(truth.contours.length, 2); assert.deepEqual(truth.contours[0].points, square);
  assert.equal(new Set(truth.contours.map(c => c.id)).size, 1);
  assert.throws(() => parsePatientTruth(data, 6, 6, 'a.png'), /outside/);
  assert.throws(() => parsePatientTruth({ ...data, annotation_count: 2 }, 10, 10, 'a.png'), /annotation_count/);
  assert.throws(() => parsePatientTruth({ teeth: [{ contour: [[1,1],[1,1],[1,1]] }] }, 10, 10), /zero-area/);
  assert.throws(() => parsePatientTruth({ ...data, imageWidth: 99 }, 10, 10), /imageWidth/);
});
test('boundary score agrees with exact supplied vector geometry, and missing predictions are not zero', () => {
  const labels = new Int32Array(100); for (let y=1;y<4;y++) for(let x=1;x<4;x++) labels[y*10+x]=1;
  const truth = [{ id: 1, smooth: false, points: square }];
  assert.equal(scoreBatchPrediction(truth, labels, 10, 10).error, 0);
  const missed = scoreBatchPrediction(truth, new Int32Array(100), 10, 10);
  assert.equal(missed.error, null); assert.equal(missed.status, 'missing prediction');
  assert.deepEqual(summarizeBatch([{status:'ok',error:0},{status:'ok',error:.2},missed]), {total:3,scored:2,unavailable:1,mean:.1});
  assert.equal(summarizeBatch([missed]).mean, null);
});
test('saved editor labels honor their explicit crop and reject mismatched image names', () => {
  const doc = { schema:'tooth-labels/1', cleanImage:'a.png', crop:[2,3,5,5], width:5,height:5,labels:encodeLabels(new Int32Array(25)) };
  assert.deepEqual(parsePatientTruth(doc, 10, 10, 'a.png').crop, [2,3,5,5]);
  assert.throws(() => parsePatientTruth(doc, 10, 10, 'b.png'), /match/);
});
test('all ten supplied patient JSON files align within their PNG dimensions', async t => {
  const root = new URL('../../testvideophoto/', import.meta.url); let count = 0;
  for (const patient of ['Patient_1','Patient_2']) {
    let scan; try { scan = await scanPatientFolder(new URL(patient, root).pathname, root.pathname); }
    catch(e) { if(e.code==='ENOENT') {t.skip('Optional patient fixtures unavailable');return;} throw e; }
    const pairs = pairPatientFiles(scan.files); assert.equal(pairs.length, 5);
    for (const pair of pairs) {
      assert.equal(pair.issue,null);
      const bytes = await readFile(pair.image.absolute), doc = JSON.parse(await readFile(pair.annotation.absolute,'utf8'));
      const truth = parsePatientTruth(doc,bytes.readUInt32BE(16),bytes.readUInt32BE(20),pair.image.path);
      assert.equal(new Set(truth.contours.map(c=>c.id)).size,doc.annotation_count); count++;
    }
  }
  assert.equal(count,10);
});
test('local folder endpoint rejects cross-origin requests and paths outside its root; skips symlinks', async t => {
  const root = await mkdtemp(join(tmpdir(),'dental-batch-')); t.after(()=>rm(root,{recursive:true,force:true}));
  await mkdir(join(root,'Patient')); await writeFile(join(root,'Patient','a.json'),'{}');
  await symlink('/etc/passwd',join(root,'Patient','escape.json'));
  assert.equal((await scanPatientFolder(root,root)).files.length,1);
  await assert.rejects(scanPatientFolder(tmpdir(),root),/inside this project/);
  const server=createServer((req,res)=>patientFolder(req,res,()=>{res.statusCode=404;res.end();}));
  const patientFolder=patientFolderMiddleware(root);
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(()=>new Promise(resolve=>server.close(resolve)));
  const base=`http://127.0.0.1:${server.address().port}`;
  const headers={'X-Dental-Client':'1','Content-Type':'application/json'};
  assert.equal((await fetch(base+'/__batch/scan',{method:'POST',headers:{...headers,Origin:'https://example.org'},body:JSON.stringify({folder:root})})).status,403);
  assert.equal((await fetch(base+'/__batch/scan',{method:'POST',body:'{}'})).status,403);
  const response=await fetch(base+'/__batch/scan',{method:'POST',headers,body:JSON.stringify({folder:root})});
  const manifest=await response.json(); assert.equal(manifest.files.length,1); assert.equal(manifest.files[0].absolute,undefined);
  assert.equal(await (await fetch(base+manifest.files[0].url,{headers})).text(),'{}');
});
test('Excel round trip retains numeric errors, cached overall mean, unavailable rows and literal filenames', async () => {
  const rows=[{patient:'P1',image:'=1+1.png',annotation:'a.json',status:'ok',error:0},
    {patient:'P2',image:'b.png',annotation:'b.json',status:'ok',error:.2},
    {patient:'P3',image:'c.png',annotation:'c.json',status:'missing prediction',error:null}];
  const book=await createBatchWorkbook(rows,{timestamp:'2026-11-09T00:00:00Z',source:'fixture',detector:'learned',model:'test'});
  const {default:ExcelJS}=await import('exceljs'), restored=new ExcelJS.Workbook();
  await restored.xlsx.load(await book.xlsx.writeBuffer());
  assert.equal(restored.getWorksheet('Summary').getCell('B2').value.result,.1);
  assert.equal(restored.getWorksheet('Summary').getCell('B5').value.result,1);
  assert.equal(restored.getWorksheet('Images').getCell('E2').value,0);
  assert.equal(restored.getWorksheet('Images').getCell('E4').value,null);
  assert.equal(restored.getWorksheet('Images').getCell('C2').value,'=1+1.png');
  const empty=await createBatchWorkbook([],{timestamp:'2026-11-09T00:00:00Z',source:'empty',detector:'learned',model:'test'});
  assert.equal(empty.getWorksheet('Summary').getCell('B2').value,'Unavailable');
});
