import { summarizeBoundaryMetrics, boundaryOptions } from './boundaryOptions.js';
import { summarizeBatch } from './batchEvaluation.js';

// Lazy-loaded by the download button, keeping Excel code out of camera startup.
export async function createBatchWorkbook(rows, run) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook(); book.creator = 'Dental AR';
  book.created = new Date(run.timestamp); book.calcProperties.fullCalcOnLoad = true;
  const summary = book.addWorksheet('Summary'), images = book.addWorksheet('Images');
  const totals = summarizeBatch(rows), metrics = summarizeBoundaryMetrics(rows), options = boundaryOptions(run.boundaryOptions);
  summary.columns = [{ width: 38 }, { width: 95 }];
  summary.addRows([
    ['Dental AR batch evaluation', ''],
    ['Overall normalized boundary error', totals.mean ?? 'Unavailable'],
    ['Scored images', totals.scored], ['Total entries', totals.total], ['Entries without a score', totals.unavailable],
    ['Run status', run.cancelled ? 'Cancelled — partial results' : totals.unavailable ? 'Completed with unavailable results — check Images' : 'Completed'],
    ['Detector', run.detector], ['Model', run.model], ['Model input (pixels)', run.inputSize?.join(' × ') ?? 'Unavailable'],
    ['Started (UTC)', run.timestamp], ['Source folder', run.source],
    ['Definition', 'Arc-length weighted bidirectional Euclidean ASSD divided by evaluated region diagonal. 0 = exact match; lower is better.'],
    ['Overall calculation', 'Arithmetic mean of scored image errors; each image has equal weight. Missing predictions, invalid labels, unmatched files and cancelled entries have no numeric score and remain listed.'],
    ['Evaluation region', 'Whole native image for patient contour JSON; saved crop for tooth-labels/1. No crop is inferred from ground truth. Normalization depends on the evaluated region.'],
    ['Ground-truth coordinates', 'Patient contours are used as supplied, without smoothing or rescaling. JSON without dimensions must already align with its paired image.'],
    ['Parameters', JSON.stringify(run.parameters ?? {})],
    ['Mean image Hausdorff (px)', metrics.meanHausdorffPx ?? 'Unavailable'],
    ['Maximum image Hausdorff (px)', metrics.maxHausdorffPx ?? 'Unavailable'],
    ['Mean image SDF coverage', metrics.meanSdfCoverage ?? 'Unavailable'],
    ['SDF PASS images', metrics.sdfPassed], ['SDF FAIL images', metrics.sdfFailed], ['SDF unavailable entries', metrics.sdfUnavailable],
    ['SDF tolerance (px)', options.tolerancePx], ['SDF required coverage', options.requiredCoverage],
    ['Hausdorff definition', 'Maximum bidirectional Euclidean boundary mismatch in pixels; normalized value divides by evaluated region diagonal. Lower is better. Sampled maximum, including vertices, spacing <=0.05 px; curves flattened to 0.005 px.'],
    ['SDF definition', 'Proportion of boundary length in BOTH directions whose absolute signed distance to the other border is <= tolerance. Higher coverage is better; SDF error = 1 - coverage. PASS requires the configured coverage. Inside/outside sign is irrelevant to this absolute-distance test.'],
    ['Aggregation and missing data', 'Means weight each evaluated image equally, not each pixel. Missing predictions contribute 0 coverage and FAIL, but have no finite Hausdorff distance. Missing GT, failed and cancelled entries are unavailable, not PASS. Check coverage counts alongside averages.'],
    ['Units', 'Pixels are native evaluated-region pixels. Tolerance is not normalized; do not compare pixel errors across resolutions without considering scale. PASS is a tolerance criterion, not a clinical assessment.'],
  ]);
  images.columns = [
    ['Patient folder', 24], ['Patient ID', 14], ['Clean image', 36], ['Ground-truth JSON', 36],
    ['Normalized boundary error (0–1)', 24], ['Hausdorff distance (px)', 24], ['Normalized Hausdorff (0–1)', 26],
    ['SDF coverage', 20], ['SDF error (1 − coverage)', 24], ['SDF check', 18], ['SDF tolerance (px)', 22], ['Required SDF coverage', 24],
    ['Status', 29], ['Details', 65], ['GT teeth', 14], ['Predicted teeth', 18], ['Width (px)', 14], ['Height (px)', 14],
    ['Region x', 14], ['Region y', 14], ['Region width', 17], ['Region height', 17], ['Inference (ms)', 18],
  ].map(([header, width]) => ({ header, width }));
  for (const row of rows) images.addRow([row.patient, row.patientId ?? null, row.image, row.annotation,
    Number.isFinite(row.error) ? row.error : null,
    row.boundary?.hausdorffPx ?? null, row.boundary?.normalizedHausdorff ?? null,
    row.boundary?.sdfCoverage ?? null, row.boundary?.sdfError ?? null,
    row.boundary?.sdfPass == null ? 'Unavailable' : row.boundary.sdfPass ? 'PASS' : 'FAIL',
    row.boundary?.tolerancePx ?? options.tolerancePx, row.boundary?.requiredCoverage ?? options.requiredCoverage,
    row.status, row.details || null, row.gtTeeth ?? null, row.predictedTeeth ?? null, row.width ?? null, row.height ?? null,
    ...(row.crop ?? [null, null, null, null]), row.elapsedMs ?? null]);
  const end = rows.length + 1;
  if (rows.length) {
    summary.getCell('B2').value = { formula: `IF(COUNT('Images'!E2:E${end})=0,"Unavailable",AVERAGE('Images'!E2:E${end}))`, result: totals.mean ?? 'Unavailable' };
    summary.getCell('B3').value = { formula: `COUNT('Images'!E2:E${end})`, result: totals.scored };
    summary.getCell('B5').value = { formula: 'B4-B3', result: totals.unavailable };
    images.autoFilter = { from: 'A1', to: `W${end}` };
  }
  summary.getCell('B2').numFmt = '0.000000'; images.getColumn(5).numFmt = '0.000000';
  images.getColumn(23).numFmt = '0.0';
  for (const col of [6, 11]) images.getColumn(col).numFmt = '0.000';
  for (const col of [7, 9]) images.getColumn(col).numFmt = '0.000000';
  for (const col of [8, 12]) images.getColumn(col).numFmt = '0.00%';
  for (const cell of ['B19', 'B24']) summary.getCell(cell).numFmt = '0.00%';
  for (const cell of ['B17', 'B18']) summary.getCell(cell).numFmt = '0.000';
  if (rows.length) {
    for (const [cell, column, op, cached] of [['B17','F','AVERAGE',metrics.meanHausdorffPx],['B18','F','MAX',metrics.maxHausdorffPx],['B19','H','AVERAGE',metrics.meanSdfCoverage]])
      summary.getCell(cell).value = {formula:`IF(COUNT('Images'!${column}2:${column}${end})=0,"Unavailable",${op}('Images'!${column}2:${column}${end}))`,result:cached ?? 'Unavailable'};
    for (const [cell, value, count] of [['B20','PASS',metrics.sdfPassed],['B21','FAIL',metrics.sdfFailed],['B22','Unavailable',metrics.sdfUnavailable]])
      summary.getCell(cell).value = {formula:`COUNTIF('Images'!J2:J${end},"${value}")`,result:count};
  }
  for (const sheet of [summary, images]) {
    sheet.views = [{ state: 'frozen', ySplit: 1, showGridLines: false }];
    sheet.eachRow((row, index) => {
      row.font = { name: 'Calibri', size: 11, color: { argb: 'FF203044' } };
      row.alignment = { vertical: 'top', wrapText: true }; row.height = sheet === summary && index >= 12 ? 48 : 32;
      if (index % 2 === 0) row.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFF0F5FA' } };
    });
    sheet.getRow(1).font = { name: 'Calibri', size: 12, bold: true, color: { argb: 'FFFFFFFF' } };
    sheet.getRow(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF173451' } };
    sheet.getRow(1).height = 36;
  }
  for (let i = 0; i < rows.length; i++) if (rows[i].status !== 'ok')
    images.getCell(i + 2, 13).font = { color: { argb: 'FF9C4020' }, bold: true };
  return book;
}
