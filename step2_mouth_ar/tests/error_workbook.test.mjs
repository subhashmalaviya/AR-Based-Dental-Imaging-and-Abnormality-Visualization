import test from 'node:test';
import assert from 'node:assert/strict';
import ExcelJS from 'exceljs';
import { scoreBatchPrediction } from '../src/eval/batchEvaluation.js';
import { createBatchWorkbook } from '../src/eval/batchWorkbook.js';
import { createComparisonWorkbook } from '../src/eval/errorWorkbook.js';
const curve = (id, x, y) => ({ id, smooth: false, points: [[x,y],[x+2,y],[x+2,y+2],[x,y+2]] });
function fixture() {
  const labels = new Int32Array(100);
  for (const [id,x,y] of [[5,1,1],[6,7,7]]) for(let dy=0;dy<2;dy++) for(let dx=0;dx<2;dx++) labels[(y+dy)*10+x+dx]=id;
  return scoreBatchPrediction([curve(1,1,1),curve(2,5,1)], labels, 10, 10, { pixelsPerMm: 2 });
}
async function reload(book) {
  const restored = new ExcelJS.Workbook(); await restored.xlsx.load(await book.xlsx.writeBuffer()); return restored;
}
function records(sheet) {
  const keys = sheet.getRow(1).values.slice(1), output=[];
  sheet.eachRow((row, i) => { if(i>1) output.push(Object.fromEntries(keys.map((key,j)=>[key,row.getCell(j+1).value]))); });
  return output;
}
test('batch computes individual errors, missed teeth and extras without losing overall errors', () => {
  const result=fixture();
  assert.equal(result.perTooth.rows.length,2);
  assert.equal(result.perTooth.rows[0].prediction,5);
  assert.equal(result.perTooth.rows[0].boundary.averageErrorPx,0);
  assert.equal(result.perTooth.rows[1].prediction,null);
  assert.equal(result.perTooth.rows[1].boundary.averageErrorPx,null);
  assert.equal(result.perTooth.rows[1].boundary.sdfPass,false);
  assert.equal(result.perTooth.extraPredictions[0].id,6);
  assert.ok(result.boundary.averageErrorPx>0);
  assert.equal(result.boundary.averageErrorMm,result.boundary.averageErrorPx/2);
  assert.ok(result.scores.fn>0);
});
test('multi-ring teeth retain all pixels and saved masks retain holes', () => {
  const labels=new Int32Array(100);
  for(const x of [1,5]) for(let y=1;y<3;y++) for(let dx=0;dx<2;dx++) labels[y*10+x+dx]=1;
  const result=scoreBatchPrediction([curve(1,1,1),curve(1,5,1)],labels,10,10);
  assert.equal(result.perTooth.rows.length,1); assert.equal(result.perTooth.rows[0].gtPixels,8);
  labels[11]=0;
  const saved=scoreBatchPrediction([curve(1,1,1),curve(1,5,1)],labels,10,10,{},labels);
  assert.equal(saved.perTooth.rows[0].gtPixels,7);assert.equal(saved.scores.iou,1);
});
test('Excel round trip includes all scalar metrics, per-image errors, tooth rows and unavailable results', async () => {
  const result=fixture();
  const book=await reload(await createBatchWorkbook([
    {patient:'Patient_1',image:'=1+1.png',annotation:'a.json',...result},
    {patient:'Patient_2',image:'failed.png',status:'failed',details:'Invalid labels'}
  ],{timestamp:'2026-11-25T00:00:00Z',detector:'learned'}));
  assert.equal(book.getWorksheet('Images').getCell('E2').value,result.error);
  assert.equal(book.getWorksheet('Images').getCell('C2').value,'=1+1.png');
  assert.equal(book.getWorksheet('Images').getCell('E3').value,null);
  const teeth=records(book.getWorksheet('Individual teeth'));
  assert.equal(teeth.length,2); assert.equal(teeth[0]['boundary.averageErrorMm'],0);
  assert.equal(teeth[1]['boundary.averageErrorMm'],null);assert.equal(teeth[1]['boundary.sdfPass'],false);
  assert.equal(records(book.getWorksheet('Extra predictions'))[0].prediction,6);
  const values=records(book.getWorksheet('All calculated values'));
  for(const [key,value] of Object.entries(result.boundary)) {
    const exported=values.find(r=>r.scope==='Image overall'&&r.metric===`boundary.${key}`);
    assert.ok(exported, key);assert.equal(exported.value,value,key);
  }
  const overall=records(book.getWorksheet('Overall metrics')).find(r=>r.metric==='boundary.averageErrorPx');
  assert.equal(overall.mean,result.boundary.averageErrorPx);assert.equal(overall.unavailableImages,1);
});
test('comparison uses edited result snapshot and exports diagnostics without large masks', async () => {
  const result=fixture();
  const book=await reload(await createComparisonWorkbook({cleanImage:'edited.png',groundTruthImage:'edited.json',originalSize:[10,10],evaluatedRegion:[0,0,10,10],
    ...result,manualEditOperations:7,matchedAverageErrorPx:0,scope:'Edited crop',predictionMask:[1,2,3]}));
  assert.equal(book.getWorksheet('Summary').getCell('B14').value,'Edited crop');
  const values=records(book.getWorksheet('All calculated values'));
  assert.equal(values.find(r=>r.metric==='manualEditOperations').value,7);
  assert.equal(values.find(r=>r.metric==='matchedAverageErrorPx').value,0);
  assert.ok(!values.some(r=>r.metric.includes('predictionMask')));
});
test('reference comparison and empty batch export clearly unavailable individual metrics',async()=>{
  const reference=await createComparisonWorkbook({mode:'live-reference',scores:{iou:0.5},perCard:[{iou:0.25}]});
  assert.equal(reference.getWorksheet('Individual teeth').rowCount,1);
  assert.ok(records(reference.getWorksheet('All calculated values')).some(r=>r.scope==='Reference region 1'&&r.value===0.25));
  const empty=await reload(await createBatchWorkbook([],{timestamp:'2026-11-25T00:00:00Z'}));
  assert.equal(empty.getWorksheet('Summary').getCell('B2').value,'Unavailable');
});
