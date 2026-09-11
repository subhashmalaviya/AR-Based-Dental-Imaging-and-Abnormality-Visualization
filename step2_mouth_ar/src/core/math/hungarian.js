/**
 * hungarian.js — optimal one-to-one assignment (Kuhn–Munkres, O(n^3)).
 *
 * Used twice: by ToothTracker to associate detections with tracks, and by the
 * evaluation code to match detections with ground-truth teeth. With ~10–30
 * teeth per frame the cubic cost is a few microseconds, so there is no reason
 * to accept greedy matching's occasional wrong answer — greedy is exactly what
 * swaps IDs between two adjacent crowns that both overlap the same track.
 *
 * @param {number[][]} cost  n x m cost matrix (lower is better). Use a large
 *   finite value (e.g. FORBIDDEN) for pairs that must never be matched.
 * @returns {number[]} rowToCol: for each row, its assigned column or -1.
 */
export const FORBIDDEN = 1e6;

export function hungarian(cost) {
  const n = cost.length;
  if (!n) return [];
  const m = cost[0].length;
  if (!m) return new Array(n).fill(-1);

  // The algorithm below needs rows <= cols; transpose if not.
  const transpose = n > m;
  const C = transpose
    ? Array.from({ length: m }, (_, j) => Array.from({ length: n }, (_, i) => cost[i][j]))
    : cost;
  const N = C.length, M = C[0].length;

  const INF = Number.POSITIVE_INFINITY;
  const u = new Float64Array(N + 1);
  const v = new Float64Array(M + 1);
  const p = new Int32Array(M + 1);     // p[j] = row matched to column j (1-based), 0 = free
  const way = new Int32Array(M + 1);

  for (let i = 1; i <= N; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(M + 1).fill(INF);
    const used = new Uint8Array(M + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= M; j++) {
        if (used[j]) continue;
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= M; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const rowToCol = new Array(N).fill(-1);
  for (let j = 1; j <= M; j++) if (p[j]) rowToCol[p[j] - 1] = j - 1;
  if (!transpose) return rowToCol;

  const out = new Array(n).fill(-1);
  rowToCol.forEach((col, r) => { if (col >= 0) out[col] = r; });
  return out;
}

/**
 * Convenience: solve, then drop any pair whose cost reached `gate` (i.e. the
 * solver was forced to use a forbidden pair because nothing else was left).
 * @returns {Array<[number, number]>} matched [row, col] pairs
 */
export function assign(cost, gate = FORBIDDEN) {
  const r2c = hungarian(cost);
  const pairs = [];
  r2c.forEach((c, r) => { if (c >= 0 && cost[r][c] < gate) pairs.push([r, c]); });
  return pairs;
}
