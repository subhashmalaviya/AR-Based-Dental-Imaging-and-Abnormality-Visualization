export function boundaryOptions({ tolerancePx = 0.1, requiredCoverage = 0.95 } = {}) {
  if (!Number.isFinite(tolerancePx) || tolerancePx < 0) throw new Error('SDF tolerance must be a non-negative pixel distance');
  if (!Number.isFinite(requiredCoverage) || requiredCoverage < 0 || requiredCoverage > 1) throw new Error('Required SDF coverage must be between 0 and 1');
  return { tolerancePx, requiredCoverage };
}
export const formatDistance = n => Number.isFinite(n) ? n.toFixed(3) : '—';
export const formatCoverage = n => Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : '—';
export const formatNormalized = n => Number.isFinite(n) ? `${n.toPrecision(4)} (${(n * 100).toFixed(2)}%)` : '—';
export const formatPass = value => value == null ? 'Unavailable' : value ? 'PASS' : 'FAIL';
export function summarizeBoundaryMetrics(rows) {
  const metrics = rows.map(r => r.boundary).filter(Boolean);
  const mean = key => { const values = metrics.map(b => b[key]).filter(Number.isFinite); return values.length ? values.reduce((a,b)=>a+b,0)/values.length : null; };
  const hd = metrics.map(b=>b.hausdorffPx ?? b.maxErrorPx).filter(Number.isFinite);
  const normHd = metrics.map(b=>b.normalizedHausdorff ?? b.normalizedMaxError).filter(Number.isFinite);
  return {
    meanAverageErrorPx: mean('averageErrorPx') ?? mean('assdPx'),
    meanNormalizedAverageError: mean('normalizedAverageError') ?? mean('normalizedAssd'),
    meanHausdorffPx: mean('hausdorffPx') ?? mean('maxErrorPx'),
    maxHausdorffPx: hd.length ? Math.max(...hd) : null,
    meanNormalizedHausdorff: mean('normalizedHausdorff') ?? mean('normalizedMaxError'),
    maxNormalizedHausdorff: normHd.length ? Math.max(...normHd) : null,
    meanSdfCoverage: mean('sdfCoverage'),
    sdfPassed: metrics.filter(b=>b.sdfPass===true).length,
    sdfFailed: metrics.filter(b=>b.sdfPass===false).length,
    sdfUnavailable: rows.length-metrics.filter(b=>b.sdfPass!=null).length
  };
}
