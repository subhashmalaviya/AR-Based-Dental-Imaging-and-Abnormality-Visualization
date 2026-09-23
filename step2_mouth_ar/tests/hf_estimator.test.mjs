/**
 * hf_estimator.test.mjs — Unit tests for HFToothEstimator.js
 *
 * These tests mock globalThis.fetch so they run headlessly without any real
 * network or browser APIs.
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';

// --------------------------------------------------------------------------
// Stub browser APIs that the module (or its imports) might call at module load.
// We only need fetch for the actual tests; other DOM APIs used at runtime
// (document, canvas, URL) are never exercised in these headless tests.
// --------------------------------------------------------------------------

let originalFetch;

before(() => {
  originalFetch = globalThis.fetch;
});
after(() => {
  globalThis.fetch = originalFetch;
});

// Helper: build a mock fetch that returns the given JSON payload.
function mockFetch(payload, status = 200) {
  globalThis.fetch = async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  });
}

// Lazy-import so the mock is set before module code executes fetch.
async function loadEstimator() {
  const mod = await import('../src/core/HFToothEstimator.js');
  return mod;
}

// --------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------

/** Simulates YOLOS output for a frontal photo (8 upper, 6 lower, all high-conf). */
const MOCK_DETECTIONS_14 = [
  // Upper row (cy ~ 100-180, image height ~480)
  { score: 0.92, label: 'tooth', box: { xmin: 200, ymin: 90, xmax: 240, ymax: 190 } },  // upper CI-R
  { score: 0.88, label: 'tooth', box: { xmin: 240, ymin: 95, xmax: 275, ymax: 188 } },  // upper LI-R
  { score: 0.85, label: 'tooth', box: { xmin: 275, ymin: 100, xmax: 310, ymax: 185 } }, // upper C-R
  { score: 0.83, label: 'tooth', box: { xmin: 310, ymin: 105, xmax: 355, ymax: 183 } }, // upper PM1-R
  { score: 0.80, label: 'tooth', box: { xmin: 160, ymin: 90, xmax: 200, ymax: 190 } },  // upper CI-L
  { score: 0.78, label: 'tooth', box: { xmin: 120, ymin: 95, xmax: 160, ymax: 188 } },  // upper LI-L
  { score: 0.77, label: 'tooth', box: { xmin: 80,  ymin: 100, xmax: 120, ymax: 185 } }, // upper C-L
  { score: 0.75, label: 'tooth', box: { xmin: 40,  ymin: 105, xmax: 80,  ymax: 183 } }, // upper PM1-L
  // Lower row (cy ~ 320-400)
  { score: 0.90, label: 'tooth', box: { xmin: 210, ymin: 315, xmax: 248, ymax: 400 } }, // lower CI-R
  { score: 0.87, label: 'tooth', box: { xmin: 248, ymin: 315, xmax: 280, ymax: 397 } }, // lower LI-R
  { score: 0.84, label: 'tooth', box: { xmin: 280, ymin: 315, xmax: 315, ymax: 395 } }, // lower C-R
  { score: 0.70, label: 'tooth', box: { xmin: 170, ymin: 315, xmax: 210, ymax: 400 } }, // lower CI-L
  { score: 0.68, label: 'tooth', box: { xmin: 135, ymin: 315, xmax: 170, ymax: 397 } }, // lower LI-L
  { score: 0.65, label: 'tooth', box: { xmin: 100, ymin: 315, xmax: 135, ymax: 395 } }, // lower C-L
];

it('estimateFromBlob: returns count, upperCount, lowerCount correctly', async () => {
  mockFetch(MOCK_DETECTIONS_14);
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  const result = await estimateFromBlob(blob, 10, 'hf_faketoken');

  assert.equal(result.count, 14);
  assert.equal(result.upperCount, 8);
  assert.equal(result.lowerCount, 6);
  assert.ok(Array.isArray(result.raw));
});

it('estimateFromBlob: upper arch CI is the closest tooth to midline', async () => {
  mockFetch(MOCK_DETECTIONS_14);
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  const result = await estimateFromBlob(blob, 10, 'hf_faketoken');

  // CI should have been assigned — the two centremost upper teeth
  assert.ok(result.upper.CI, 'upper.CI should be populated');
  assert.ok(result.upper.LI, 'upper.LI should be populated');
  assert.ok(result.lower.CI, 'lower.CI should be populated');
});

it('estimateFromBlob: mm values are box_px / pixelsPerMm', async () => {
  mockFetch(MOCK_DETECTIONS_14);
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  const ppm = 5;
  const result = await estimateFromBlob(blob, ppm, 'hf_faketoken');

  // Verify one raw detection was correctly converted
  const raw = result.raw[0];
  const expectedW = raw.width_px;
  const expectedH = raw.height_px;
  assert.ok(Math.abs(raw.width_mm  - expectedW / ppm) < 0.001, 'width_mm = px / ppm');
  assert.ok(Math.abs(raw.height_mm - expectedH / ppm) < 0.001, 'height_mm = px / ppm');
});

it('estimateFromBlob: filters out low-confidence detections (< MIN_SCORE)', async () => {
  const lowConfDetections = [
    { score: 0.10, label: 'tooth', box: { xmin: 200, ymin: 90, xmax: 240, ymax: 190 } },
    { score: 0.92, label: 'tooth', box: { xmin: 240, ymin: 95, xmax: 275, ymax: 188 } },
  ];
  mockFetch(lowConfDetections);
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  const result = await estimateFromBlob(blob, 10, 'hf_faketoken');

  // Only 1 tooth should survive the score filter
  assert.equal(result.count, 1, 'only high-confidence detections kept');
});

it('estimateFromBlob: throws when pixelsPerMm is 0', async () => {
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  await assert.rejects(
    () => estimateFromBlob(blob, 0, 'hf_token'),
    /pixelsPerMm must be a positive number/
  );
});

it('estimateFromBlob: throws when token is empty', async () => {
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  await assert.rejects(
    () => estimateFromBlob(blob, 10, ''),
    /API token is required/
  );
});

it('estimateFromBlob: throws on HTTP 401 with informative message', async () => {
  globalThis.fetch = async () => ({
    ok: false,
    status: 401,
    text: async () => JSON.stringify({ error: 'Authorization header is invalid' }),
  });
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  await assert.rejects(
    () => estimateFromBlob(blob, 10, 'hf_invalid'),
    /HTTP 401/
  );
});

it('estimateFromBlob: returns empty upper/lower when no teeth detected', async () => {
  mockFetch([]);
  const { estimateFromBlob } = await loadEstimator();
  const blob = new Blob(['fake'], { type: 'image/jpeg' });
  const result = await estimateFromBlob(blob, 10, 'hf_faketoken');
  assert.equal(result.count, 0);
  assert.deepEqual(result.upper, {});
  assert.deepEqual(result.lower, {});
});

it('ARCH_TOOTH_KEYS has exactly 8 entries matching the half-arch order', async () => {
  const { ARCH_TOOTH_KEYS } = await loadEstimator();
  assert.deepEqual(ARCH_TOOTH_KEYS, ['CI', 'LI', 'C', 'PM1', 'PM2', 'M1', 'M2', 'M3']);
});
