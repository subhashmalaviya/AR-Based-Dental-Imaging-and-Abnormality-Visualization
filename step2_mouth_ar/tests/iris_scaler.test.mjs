import test from 'node:test';
import assert from 'node:assert/strict';
import { IrisScaler, IRIS_LANDMARKS, IRIS_DIAMETER_MM } from '../src/core/IrisScaler.js';

test('IrisScaler: extracts iris bounding boxes and computes pixelsPerMm', () => {
  const scaler = new IrisScaler({ smoothingAlpha: 0 }); // no EMA for deterministic test

  // Create a synthetic 478-landmark array
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));

  // Frame size: 1000 x 1000
  // Left eye iris points: center at (0.4, 0.3), left edge at (0.38, 0.3), right edge at (0.42, 0.3)
  // Distance = 0.04 * 1000 = 40 px
  landmarks[IRIS_LANDMARKS.LEFT_CENTER] = { x: 0.40, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.LEFT_LEFT_EDGE] = { x: 0.38, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.LEFT_RIGHT_EDGE] = { x: 0.42, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.LEFT_TOP_EDGE] = { x: 0.40, y: 0.28, z: 0 };
  landmarks[IRIS_LANDMARKS.LEFT_BOTTOM_EDGE] = { x: 0.40, y: 0.32, z: 0 };

  // Right eye iris points: center at (0.6, 0.3), left edge at (0.58, 0.3), right edge at (0.62, 0.3)
  landmarks[IRIS_LANDMARKS.RIGHT_CENTER] = { x: 0.60, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.RIGHT_LEFT_EDGE] = { x: 0.58, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.RIGHT_RIGHT_EDGE] = { x: 0.62, y: 0.30, z: 0 };
  landmarks[IRIS_LANDMARKS.RIGHT_TOP_EDGE] = { x: 0.60, y: 0.28, z: 0 };
  landmarks[IRIS_LANDMARKS.RIGHT_BOTTOM_EDGE] = { x: 0.60, y: 0.32, z: 0 };

  const ok = scaler.update(landmarks, 1000, 1000);
  assert.equal(ok, true);
  assert.equal(scaler.isReady(), true);

  const info = scaler.info();
  assert.equal(info.source, 'iris');
  assert.ok(Math.abs(info.leftIrisDiamPx - 40) < 0.1);
  assert.ok(Math.abs(info.rightIrisDiamPx - 40) < 0.1);

  // Expected pixelsPerMm = 40 / 11.7 ~ 3.4188
  const expectedPpm = 40 / IRIS_DIAMETER_MM;
  assert.ok(Math.abs(scaler.pixelsPerMm - expectedPpm) < 0.01);

  // Check visual geometry
  assert.ok(scaler.irises != null);
  assert.ok(scaler.irises.left != null);
  assert.ok(scaler.irises.right != null);
  assert.equal(scaler.irises.left.box.width, 40);
  assert.equal(scaler.irises.left.box.height, 40);
  assert.equal(scaler.irises.left.center.x, 400);
  assert.equal(scaler.irises.left.center.y, 300);
});

test('IrisScaler: falls back to mouth width when iris landmarks absent', () => {
  const scaler = new IrisScaler({ smoothingAlpha: 0 });

  // 468 landmarks only (no iris points)
  const landmarks = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5, z: 0 }));

  // Mouth corners: 61 and 291
  // Left corner at x=0.45, right corner at x=0.55 -> width = 0.10 * 1000 = 100 px
  landmarks[61] = { x: 0.45, y: 0.65, z: 0 };
  landmarks[291] = { x: 0.55, y: 0.65, z: 0 };

  const ok = scaler.update(landmarks, 1000, 1000);
  assert.equal(ok, true);
  assert.equal(scaler.info().source, 'mouth_width');
  assert.ok(Math.abs(scaler.info().mouthWidthPx - 100) < 0.1);
  // Default mouth width is 50 mm -> 100 / 50 = 2.0 px/mm
  assert.ok(Math.abs(scaler.pixelsPerMm - 2.0) < 0.01);
  assert.ok(scaler.mouthRef != null);
  assert.equal(scaler.mouthRef.widthPx, 100);
});
