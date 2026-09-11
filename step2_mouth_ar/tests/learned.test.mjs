/**
 * The learned detector, run exactly as the browser runs it (same module, same
 * ONNX Runtime Web package, Node build) on a real rectified mouth crop.
 *
 * Skipped when public/models/tooth_seg.onnx is absent (e.g. a fresh clone
 * before the model is trained); the decoder itself is covered by
 * step3v2.test.mjs with synthetic maps.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';
import path from 'node:path';

import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { pickRecordingFormat } from '../src/core/SessionRecorder.js';

const MODEL = path.resolve('public/models/tooth_seg.onnx');
const CROP = path.resolve('tests/eval/sample_roi_160x120.rgb.z');
const have = fs.existsSync(MODEL) && fs.existsSync(CROP);

test('no MediaRecorder -> recording reports unsupported instead of failing later', () => {
  assert.equal(pickRecordingFormat(), null);
});

test('model card documents datasets, licences and validation', { skip: !have && 'model not present' }, () => {
  const info = JSON.parse(fs.readFileSync(MODEL.replace(/\.onnx$/, '.json'), 'utf8'));
  assert.equal(info.input.width, 160);
  assert.equal(info.input.height, 120);
  const names = info.training.datasets.map((d) => d.name);
  assert.ok(names.includes('DentalAI') && names.includes('EasyPortrait'));
  for (const d of info.training.datasets) assert.ok(d.license && d.url, `${d.name} licence/url`);
  assert.ok(info.validation.ep_teeth_iou > 0, 'validation metrics recorded');
  assert.ok(info.onnx_max_abs_diff < 1e-3, 'exported graph matches the trained network');
});

test('learned detector finds separate teeth in a real mouth crop', { skip: !have && 'model not present' }, async () => {
  const det = new LearnedToothDetector({ modelUrl: MODEL });
  await det.init();
  const W = det.inputWidth, H = det.inputHeight;
  const rgb = zlib.inflateSync(fs.readFileSync(CROP));
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[4 * i] = rgb[3 * i]; data[4 * i + 1] = rgb[3 * i + 1]; data[4 * i + 2] = rgb[3 * i + 2]; data[4 * i + 3] = 255;
  }
  const b = { u0: -0.6, u1: 0.6, v0: -0.45, v1: 0.45 };
  const roi = {
    width: W, height: H, bounds: b,
    roiToLocal: (x, y) => ({ u: b.u0 + (x / W) * (b.u1 - b.u0), v: b.v0 + (y / H) * (b.v1 - b.v0) }),
  };
  const aperture = new Uint8Array(W * H).fill(255);
  const dets = await det.detectAsync({ data, width: W, height: H }, aperture, roi);

  // The sample is a clenched smile with ~12 visible teeth (see tests/eval).
  assert.ok(dets.length >= 6, `expected several separate teeth, got ${dets.length}`);
  for (const d of dets) {
    assert.ok(d.confidence > 0 && d.confidence <= 1);
    assert.ok(d.arch === 'upper' || d.arch === 'lower');
    assert.ok(d.contour.length >= 4, 'each tooth has its own outline');
    assert.ok(d.box.w > 0 && d.box.h > 0);
  }
  assert.ok(dets.some((d) => d.arch === 'upper') && dets.some((d) => d.arch === 'lower'),
    'both arches found in a clenched smile');
  assert.ok(det.lastDebug.maps.teeth.length === W * H, 'probability maps exposed for the debug view');
});
