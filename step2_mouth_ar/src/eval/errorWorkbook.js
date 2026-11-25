// Detail sheets deliberately omit masks/contours: exports contain measurements,
// evaluation settings and matching provenance, not millions of source pixels.
const scalar = value => value == null ? null : typeof value === 'number' ? (Number.isFinite(value) ? value : null)
  : typeof value === 'object' ? JSON.stringify(value) : value;
const flatten = (value, prefix = '', out = {}) => {
  for (const [key, item] of Object.entries(value ?? {})) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (item && typeof item === 'object' && !Array.isArray(item) && !ArrayBuffer.isView(item)) flatten(item, name, out);
    else out[name] = scalar(item);
  }
  return out;
};
function table(book, name, records, defaultKeys) {
  const sheet = book.addWorksheet(name);
  const keys = [...new Set([...defaultKeys, ...records.flatMap(Object.keys)])];
  sheet.columns = keys.map(key => ({ header: key, key, width: /image|annotation|details|metric/i.test(key) ? 36 : 24 }));
  for (const record of records) sheet.addRow(record);
  sheet.views = [{ state: 'frozen', ySplit: 1, showGridLines: false }];
  sheet.autoFilter = { from: { row: 1, column: 1 }, to: { row: Math.max(1, sheet.rowCount), column: keys.length } };
  sheet.eachRow((row, index) => {
    row.alignment = { vertical: 'top', wrapText: true };
    row.font = { name: 'Calibri', size: 11, color: { argb: index === 1 ? 'FFFFFFFF' : 'FF203044' }, bold: index === 1 };
    row.height = index === 1 ? 42 : 30;
    if (index === 1 || index % 2 === 0) row.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: index === 1 ? 'FF173451' : 'FFF0F5FA' } };
    row.eachCell(cell => { if (typeof cell.value === 'number') cell.numFmt = '0.########'; });
  });
  return sheet;
}
export function addErrorDetails(book, entries, run) {
  const teeth = [], extras = [], values = [], aggregates = new Map();
  for (const entry of entries) {
    const identity = { patient: entry.patient ?? '', image: entry.image ?? '', annotation: entry.annotation ?? '' };
    const recordValues = (scope, record, ids = {}) => {
      for (const [metric, value] of Object.entries(flatten(record)))
        values.push({ ...identity, scope, ...ids, metric, value, availability: value == null ? 'Unavailable' : 'Available' });
    };
    recordValues('Image overall', { boundary: entry.boundary, scores: entry.scores, error: entry.error, status: entry.status, details: entry.details });
    if (entry.comparisonDetails) recordValues('Comparison metadata and diagnostics', entry.comparisonDetails);
    if (entry.perCard) entry.perCard.forEach((card, i) => recordValues(`Reference region ${i + 1}`, card));
    const perTooth = entry.perTooth;
    if (perTooth) {
      const { rows, extraPredictions, ...matching } = perTooth;
      recordValues('Instance matching summary', matching);
      for (const row of rows) {
        teeth.push({ ...identity, ...flatten(row) });
        recordValues('Individual tooth', row, { groundTruth: row.groundTruth, prediction: row.prediction });
      }
      for (const extra of extraPredictions) extras.push({ ...identity, prediction: extra.id, areaPixels: extra.area, centroidX: extra.x, centroidY: extra.y, status: 'Unmatched prediction; no paired GT distance' });
    } else recordValues('Instance matching summary', { status: 'Individual-tooth results unavailable in this evaluation mode or entry' });
    for (const [metric, value] of Object.entries(flatten({ boundary: entry.boundary, scores: entry.scores }))) {
      // Settings, counts and geometry are exported above; only errors, coverage
      // and overlap ratios are averaged, each image weighted equally.
      if (!/^(boundary\.(assd|averageError|normalized|hausdorff|maxError|absoluteMaxError|sdfCoverage|sdfError|truthWithinTolerance|predictionWithinTolerance|truthToPrediction|predictionToTruth)|scores\.(iou|dice|precision|recall))/.test(metric)) continue;
      if (!aggregates.has(metric)) aggregates.set(metric, []);
      if (typeof value === 'number' && Number.isFinite(value)) aggregates.get(metric).push(value);
    }
  }
  table(book, 'Individual teeth', teeth, ['patient', 'image', 'annotation', 'groundTruth', 'prediction', 'status', 'normalizedBoundaryError']);
  table(book, 'Extra predictions', extras, ['patient', 'image', 'annotation', 'prediction', 'areaPixels', 'centroidX', 'centroidY', 'status']);
  table(book, 'Overall metrics', [...aggregates].map(([metric, numbers]) => ({ metric,
    mean: numbers.length ? numbers.reduce((a, b) => a + b, 0) / numbers.length : null,
    maximum: numbers.length ? Math.max(...numbers) : null,
    scoredImages: numbers.length, unavailableImages: entries.length - numbers.length,
    aggregation: 'Equal weight per image; unavailable values excluded. Not a mean of individual teeth.' })), ['metric', 'mean', 'maximum', 'scoredImages', 'unavailableImages', 'aggregation']);
  table(book, 'All calculated values', values, ['patient', 'image', 'annotation', 'scope', 'groundTruth', 'prediction', 'metric', 'value', 'availability']);
  table(book, 'Evaluation settings', Object.entries(flatten(run)).map(([setting, value]) => ({ setting, value })), ['setting', 'value']);
}

export async function createComparisonWorkbook(result) {
  const { createBatchWorkbook } = await import('./batchWorkbook.js');
  const { predictionMask, alignedReferenceMask, perTooth, boundary, scores, perCard, ...details } = result;
  const entry = {
    patient: '', image: result.cleanImage ?? result.source ?? result.mode ?? 'Comparison',
    annotation: result.groundTruthImage ?? result.groundTruth ?? result.referenceView ?? '',
    error: boundary?.normalizedAssd ?? result.normalizedBoundaryError ?? null,
    boundary, scores, perTooth, perCard, comparisonDetails: details,
    status: boundary?.status ?? 'unavailable', width: result.originalSize?.[0] ?? result.width,
    height: result.originalSize?.[1] ?? result.height, crop: result.evaluatedRegion,
    gtTeeth: perTooth?.groundTruthCount, predictedTeeth: perTooth?.predictionCount
  };
  const book = await createBatchWorkbook([entry], {
    timestamp: result.timestamp ?? new Date().toISOString(), source: entry.image,
    detector: result.detector, model: result.model ?? result.detector,
    parameters: result.parameters, inputSize: result.modelInput, boundaryOptions: boundary,
    scope: result.scope, manualEditOperations: result.manualEditOperations
  });
  const summary = book.getWorksheet('Summary');
  summary.getCell('A1').value = 'Dental AR ground-truth comparison';
  summary.getCell('B14').value = result.scope ?? 'Current evaluated image region';
  summary.getCell('B15').value = 'Uses the current ground-truth snapshot, including edits. No inference or extraction is repeated for export.';
  return book;
}
