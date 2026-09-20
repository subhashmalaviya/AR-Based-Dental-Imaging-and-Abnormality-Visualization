import test from 'node:test';
import assert from 'node:assert/strict';
import { DentalReferenceModel, DEFAULT_HALF_ARCH, TOOTH_TYPE_COLORS } from '../src/core/DentalReferenceModel.js';

test('DentalReferenceModel: generates 32 teeth symmetrically from half-arch', () => {
  const model = new DentalReferenceModel();
  const full = model.getFull32TeethModel();

  // Exactly 32 teeth
  assert.equal(Object.keys(full).length, 32);

  // Check Upper Right Central Incisor (FDI 11)
  assert.equal(full[11].fdi, 11);
  assert.equal(full[11].arch, 'upper');
  assert.equal(full[11].side, 'right');
  assert.equal(full[11].typeKey, 'CI');
  assert.equal(full[11].width, DEFAULT_HALF_ARCH.upper.CI.width);
  assert.equal(full[11].height, DEFAULT_HALF_ARCH.upper.CI.height);

  // Check Upper Left Central Incisor (FDI 21) - symmetrical
  assert.equal(full[21].fdi, 21);
  assert.equal(full[21].arch, 'upper');
  assert.equal(full[21].side, 'left');
  assert.equal(full[21].width, full[11].width);
  assert.equal(full[21].height, full[11].height);

  // Check Lower Right Canine (FDI 43)
  assert.equal(full[43].fdi, 43);
  assert.equal(full[43].arch, 'lower');
  assert.equal(full[43].typeKey, 'C');
  assert.equal(full[43].width, DEFAULT_HALF_ARCH.lower.C.width);
  assert.equal(full[43].height, DEFAULT_HALF_ARCH.lower.C.height);

  // Check Lower Left Canine (FDI 33) - symmetrical
  assert.equal(full[33].width, full[43].width);
  assert.equal(full[33].height, full[43].height);
});

test('DentalReferenceModel: allows customizing individual tooth dimensions', () => {
  const model = new DentalReferenceModel();
  model.setDimensions({
    upper: {
      CI: { width: 9.2, height: 11.0 },
    },
  });

  const full = model.getFull32TeethModel();
  assert.equal(full[11].width, 9.2);
  assert.equal(full[11].height, 11.0);
  assert.equal(full[21].width, 9.2);
  assert.equal(full[21].height, 11.0);

  // Other teeth remain at default
  assert.equal(full[12].width, DEFAULT_HALF_ARCH.upper.LI.width);

  // Reset defaults restores original
  model.resetDefaults();
  assert.equal(model.getReference('upper', 'CI').width, DEFAULT_HALF_ARCH.upper.CI.width);
});

test('DentalReferenceModel: classifies teeth from midline outwards along arch', () => {
  const model = new DentalReferenceModel();

  // Synthetic tracks in mouth-local coordinates for upper arch
  // Midline is at u=0.
  // u = 0.04 (Right Central Incisor)
  // u = -0.04 (Left Central Incisor)
  // u = 0.12 (Right Lateral Incisor)
  // u = -0.12 (Left Lateral Incisor)
  const tracks = [
    { id: 1, arch: 'upper', smoothed: { center: { u: 0.04, v: -0.1 } } },
    { id: 2, arch: 'upper', smoothed: { center: { u: -0.04, v: -0.1 } } },
    { id: 3, arch: 'upper', smoothed: { center: { u: 0.12, v: -0.1 } } },
    { id: 4, arch: 'upper', smoothed: { center: { u: -0.12, v: -0.1 } } },
  ];

  const classified = model.classifyArchTracks(tracks, 'upper');
  assert.equal(classified.length, 4);

  const t1 = classified.find((c) => c.id === 1);
  const t2 = classified.find((c) => c.id === 2);
  const t3 = classified.find((c) => c.id === 3);
  const t4 = classified.find((c) => c.id === 4);

  // Central Incisors closest to midline
  assert.equal(t1.toothType, 'CI');
  assert.equal(t2.toothType, 'CI');

  // Lateral Incisors next
  assert.equal(t3.toothType, 'LI');
  assert.equal(t4.toothType, 'LI');
});

test('DentalReferenceModel: computes particular errors against specific tooth types', () => {
  const model = new DentalReferenceModel();

  const classified = [
    { id: 10, arch: 'upper', toothType: 'CI', group: 'incisor', label: 'U-CI', refWidth: 8.5, refHeight: 10.5 },
    { id: 11, arch: 'upper', toothType: 'LI', group: 'incisor', label: 'U-LI', refWidth: 6.5, refHeight: 9.0  },
  ];

  const sizes = [
    { id: 10, wMm: 8.8, hMm: 10.2 }, // error: wAbs=0.3, hAbs=0.3
    { id: 11, wMm: 6.7, hMm: 9.2  }, // error: wAbs=0.2, hAbs=0.2
  ];

  const errResult = model.computeParticularErrors(classified, sizes);
  assert.ok(errResult != null);
  assert.equal(errResult.overall.n, 2);

  // Mean width error: (0.3 + 0.2) / 2 = 0.25
  assert.ok(Math.abs(errResult.overall.widthMAE - 0.25) < 0.001);
  // Mean height error: (0.3 + 0.2) / 2 = 0.25
  assert.ok(Math.abs(errResult.overall.heightMAE - 0.25) < 0.001);

  // Check per-tooth errors
  const e10 = errResult.teeth.find((e) => e.id === 10);
  assert.ok(Math.abs(e10.wAbs - 0.3) < 0.001);
  assert.ok(Math.abs(e10.hAbs - 0.3) < 0.001);
});
