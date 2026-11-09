import { summarizeBatch } from './batchEvaluation.js';

// Lazy-loaded by the download button, keeping Excel code out of camera startup.
export async function createBatchWorkbook(rows, run) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook(); book.creator = 'Dental AR';
  book.created = new Date(run.timestamp); book.calcProperties.fullCalcOnLoad = true;
  const summary = book.addWorksheet('Summary'), images = book.addWorksheet('Images');
  const totals = summarizeBatch(rows);
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
  ]);
  images.columns = [
    ['Patient folder', 30], ['Patient ID', 18], ['Clean image', 43], ['Ground-truth JSON', 43],
    ['Normalized boundary error (0–1)', 24], ['Status', 29], ['Details', 65],
    ['GT teeth', 14], ['Predicted teeth', 18], ['Width (px)', 14], ['Height (px)', 14],
    ['Region x', 14], ['Region y', 14], ['Region width', 17], ['Region height', 17], ['Inference (ms)', 18],
  ].map(([header, width]) => ({ header, width }));
  for (const row of rows) images.addRow([row.patient, row.patientId ?? '', row.image, row.annotation,
    Number.isFinite(row.error) ? row.error : null, row.status, row.details ?? '', row.gtTeeth ?? null,
    row.predictedTeeth ?? null, row.width ?? null, row.height ?? null, ...(row.crop ?? [null, null, null, null]), row.elapsedMs ?? null]);
  const end = rows.length + 1;
  if (rows.length) {
    summary.getCell('B2').value = { formula: `IF(COUNT('Images'!E2:E${end})=0,"Unavailable",AVERAGE('Images'!E2:E${end}))`, result: totals.mean ?? 'Unavailable' };
    summary.getCell('B3').value = { formula: `COUNT('Images'!E2:E${end})`, result: totals.scored };
    summary.getCell('B5').value = { formula: 'B4-B3', result: totals.unavailable };
    images.autoFilter = { from: 'A1', to: `P${end}` };
  }
  summary.getCell('B2').numFmt = '0.000000'; images.getColumn(5).numFmt = '0.000000';
  images.getColumn(16).numFmt = '0.0';
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
    images.getCell(i + 2, 6).font = { color: { argb: 'FF9C4020' }, bold: true };
  return book;
}
