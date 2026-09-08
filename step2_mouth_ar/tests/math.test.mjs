/**
 * Headless verification of the tracking maths.
 *
 * The camera-facing layers (CameraManager / FaceTracker) need a real device and
 * are verified on-device. Everything downstream of the landmarks is pure
 * geometry, so it is tested here against synthetic landmark sets with known
 * ground truth: a canonical face mesh is generated at a chosen pose, pushed
 * through MouthTracker -> MouthARAnchor, and the recovered pose is compared to
 * what was synthesised.
 *
 * Run:  npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  basisToEuler, basisToQuaternion, cross, dot, length, normalize,
  orthonormalBasis, quaternionToBasis, sub, v3,
} from '../src/core/math/vec3.js';
import { OneEuroFilter } from '../src/core/filters/OneEuroFilter.js';
import { PoseSmoother } from '../src/core/filters/PoseSmoother.js';
import { MouthTracker } from '../src/core/MouthTracker.js';
import { MouthARAnchor } from '../src/core/MouthARAnchor.js';
import {
  LIPS_ALL, LIPS_INNER_RING, LIPS_OUTER_RING, LOWER_LIP_INNER, LOWER_LIP_OUTER,
  MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, UPPER_LIP_INNER, UPPER_LIP_OUTER,
} from '../src/landmarks/FaceLandmarkIndices.js';

const W = 1280, H = 720;
const DEG = Math.PI / 180;

/** Rotation matrix from yaw (about Y), pitch (about X), roll (about Z). */
function rotationYPR(yawDeg, pitchDeg, rollDeg) {
  const cy = Math.cos(yawDeg * DEG), sy = Math.sin(yawDeg * DEG);
  const cp = Math.cos(pitchDeg * DEG), sp = Math.sin(pitchDeg * DEG);
  const cr = Math.cos(rollDeg * DEG), sr = Math.sin(rollDeg * DEG);
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
  const Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
  const mul = (A, B) => A.map((r, i) => B[0].map((_, j) =>
    A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));
  return mul(mul(Ry, Rx), Rz);
}
const applyR = (R, p) => v3(
  R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z,
  R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z,
  R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z,
);

/**
 * Build a synthetic 478-landmark array for a mouth of known pose.
 * Mouth-local model: corners at (+/-0.5, 0, 0), lips above/below in Y.
 */
function synthesiseLandmarks({
  yaw = 0, pitch = 0, roll = 0,
  centerPx = { x: 640, y: 400 }, widthPx = 200, open = 0.0, noisePx = 0,
  seed = 1,
} = {}) {
  const R = rotationYPR(yaw, pitch, roll);
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff) * 2 - 1; };

  const local = new Map();
  local.set(MOUTH_CORNER_LEFT, v3(-0.5, 0, 0));
  local.set(MOUTH_CORNER_RIGHT, v3(0.5, 0, 0));
  local.set(UPPER_LIP_OUTER, v3(0, -0.22, 0.02));
  local.set(UPPER_LIP_INNER, v3(0, -0.5 * open - 0.02, 0.01));
  local.set(LOWER_LIP_INNER, v3(0, 0.5 * open + 0.02, 0.01));
  local.set(LOWER_LIP_OUTER, v3(0, 0.24, 0.02));

  // Elliptical outer/inner rings so the ring centroid sits at local origin and
  // the inner-ring area responds to `open`.
  LIPS_OUTER_RING.forEach((idx, i) => {
    const a = (i / LIPS_OUTER_RING.length) * 2 * Math.PI;
    if (!local.has(idx)) local.set(idx, v3(0.5 * Math.cos(a), 0.23 * Math.sin(a), 0.02));
  });
  LIPS_INNER_RING.forEach((idx, i) => {
    const a = (i / LIPS_INNER_RING.length) * 2 * Math.PI;
    if (!local.has(idx)) {
      local.set(idx, v3(0.36 * Math.cos(a), (0.02 + 0.5 * open) * Math.sin(a), 0.01));
    }
  });

  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (const [idx, p] of local) {
    const r = applyR(R, p);
    const xPx = centerPx.x + widthPx * r.x + noisePx * rnd();
    const yPx = centerPx.y + widthPx * r.y + noisePx * rnd();
    const zPx = widthPx * r.z + noisePx * rnd();
    lm[idx] = { x: xPx / W, y: yPx / H, z: zPx / W };
  }
  return lm;
}

// --------------------------------------------------------------- vec3 basics
test('orthonormalBasis produces a right-handed orthonormal frame', () => {
  const b = orthonormalBasis(v3(2, 0.3, -0.1), v3(0.2, 1, 0.4));
  for (const a of [b.x, b.y, b.z]) assert.ok(Math.abs(length(a) - 1) < 1e-9);
  assert.ok(Math.abs(dot(b.x, b.y)) < 1e-9);
  assert.ok(Math.abs(dot(b.x, b.z)) < 1e-9);
  assert.ok(Math.abs(dot(b.y, b.z)) < 1e-9);
  const handed = sub(cross(b.x, b.y), b.z);
  assert.ok(length(handed) < 1e-9, 'x cross y must equal z');
});

test('quaternion <-> basis round-trips', () => {
  const b = orthonormalBasis(v3(1, 0.4, 0.2), v3(-0.3, 1, 0.1));
  const back = quaternionToBasis(basisToQuaternion(b));
  for (const k of ['x', 'y', 'z']) {
    assert.ok(length(sub(b[k], back[k])) < 1e-8, `axis ${k} round-trip`);
  }
});

// ------------------------------------------------------- mouth measurements
test('MouthTracker recovers mouth width and centre', () => {
  const lm = synthesiseLandmarks({ widthPx: 240, centerPx: { x: 500, y: 300 } });
  const m = new MouthTracker().track(lm, W, H);
  assert.ok(m, 'should track');
  assert.ok(Math.abs(m.mouthWidth - 240) < 1e-6, `width ${m.mouthWidth}`);
  assert.ok(Math.abs(m.origin.x - 500) < 1.0, `origin.x ${m.origin.x}`);
  assert.ok(Math.abs(m.origin.y - 300) < 1.0, `origin.y ${m.origin.y}`);
});

test('mouth opening ratio is scale-invariant and tracks aperture', () => {
  const t = new MouthTracker();
  const closed = t.track(synthesiseLandmarks({ open: 0.0 }), W, H);
  const open = t.track(synthesiseLandmarks({ open: 0.6 }), W, H);
  assert.ok(open.opening.ratio > closed.opening.ratio + 0.4, 'opening must rise');
  assert.equal(closed.opening.isOpen, false);
  assert.equal(open.opening.isOpen, true);

  // same aperture, face twice as close -> ratio must be unchanged
  const near = t.track(synthesiseLandmarks({ open: 0.6, widthPx: 400 }), W, H);
  assert.ok(Math.abs(near.opening.ratio - open.opening.ratio) < 1e-6,
    'ratio must not depend on distance to camera');
  assert.ok(near.opening.gap > open.opening.gap, 'raw gap does scale with distance');
});

test('bounding box encloses every lip landmark', () => {
  const lm = synthesiseLandmarks({ roll: 15, widthPx: 260 });
  const m = new MouthTracker().track(lm, W, H);
  const bb = m.boundingBox;
  for (const i of LIPS_ALL) {
    const x = lm[i].x * W, y = lm[i].y * H;
    assert.ok(x >= bb.x - 1e-6 && x <= bb.x + bb.width + 1e-6, `x of ${i}`);
    assert.ok(y >= bb.y - 1e-6 && y <= bb.y + bb.height + 1e-6, `y of ${i}`);
  }
});

// ------------------------------------------------------------- anchor & pose
test('anchor recovers head roll from the mouth frame', () => {
  for (const roll of [-30, -10, 0, 12, 25]) {
    const m = new MouthTracker().track(synthesiseLandmarks({ roll }), W, H);
    const a = new MouthARAnchor({ smoothing: 'responsive' });
    // feed repeatedly so the smoother converges off its initial value
    let pose;
    for (let i = 0; i < 90; i++) pose = a.update(m, i / 60);
    assert.ok(Math.abs(pose.euler.roll - roll) < 1.5,
      `roll ${roll} -> ${pose.euler.roll.toFixed(2)}`);
  }
});

test('anchor recovers yaw and pitch', () => {
  for (const [yaw, pitch] of [[0, 0], [20, 0], [-25, 0], [0, 15], [15, -12]]) {
    const m = new MouthTracker().track(synthesiseLandmarks({ yaw, pitch }), W, H);
    const a = new MouthARAnchor({ smoothing: 'responsive' });
    let pose;
    for (let i = 0; i < 120; i++) pose = a.update(m, i / 60);
    assert.ok(Math.abs(pose.euler.yaw - yaw) < 2.5,
      `yaw ${yaw} -> ${pose.euler.yaw.toFixed(2)}`);
    assert.ok(Math.abs(pose.euler.pitch - pitch) < 2.5,
      `pitch ${pitch} -> ${pose.euler.pitch.toFixed(2)}`);
  }
});

test('localToScreen maps the mouth corners back onto the detected corners', () => {
  const lm = synthesiseLandmarks({ roll: 18, yaw: 12, widthPx: 220,
    centerPx: { x: 700, y: 380 } });
  const m = new MouthTracker().track(lm, W, H);
  const a = new MouthARAnchor({ smoothing: 'responsive' });
  let pose;
  for (let i = 0; i < 120; i++) pose = a.update(m, i / 60);

  const left = a.localToScreen(v3(-0.5, 0, 0));
  const right = a.localToScreen(v3(0.5, 0, 0));
  const errL = Math.hypot(left.x - m.corners.left.x, left.y - m.corners.left.y);
  const errR = Math.hypot(right.x - m.corners.right.x, right.y - m.corners.right.y);
  assert.ok(errL < 4, `left corner reprojection ${errL.toFixed(2)} px`);
  assert.ok(errR < 4, `right corner reprojection ${errR.toFixed(2)} px`);
});

test('localToScreen / screenToLocal round-trip in the mouth plane', () => {
  const m = new MouthTracker().track(synthesiseLandmarks({ roll: -22 }), W, H);
  const a = new MouthARAnchor({ smoothing: 'responsive' });
  for (let i = 0; i < 120; i++) a.update(m, i / 60);
  for (const p of [v3(0, 0, 0), v3(0.3, -0.2, 0), v3(-0.45, 0.15, 0)]) {
    const back = a.screenToLocal(a.localToScreen(p));
    assert.ok(Math.hypot(back.x - p.x, back.y - p.y) < 1e-6, 'round-trip');
  }
});

test('anchor scale follows distance, so local units stay constant', () => {
  const t = new MouthTracker();
  const far = t.track(synthesiseLandmarks({ widthPx: 120 }), W, H);
  const near = t.track(synthesiseLandmarks({ widthPx: 360 }), W, H);
  const mk = (m) => {
    const a = new MouthARAnchor({ smoothing: 'responsive' });
    let p; for (let i = 0; i < 120; i++) p = a.update(m, i / 60);
    return { a, p };
  };
  const F = mk(far), N = mk(near);
  assert.ok(Math.abs(F.p.scale - 120) < 2, `far scale ${F.p.scale}`);
  assert.ok(Math.abs(N.p.scale - 360) < 2, `near scale ${N.p.scale}`);

  // a fixed local offset must project 3x larger when the face is 3x closer
  const dFar = Math.hypot(...['x', 'y'].map((k) =>
    F.a.localToScreen(v3(0.4, 0, 0))[k] - F.a.localToScreen(v3(0, 0, 0))[k]));
  const dNear = Math.hypot(...['x', 'y'].map((k) =>
    N.a.localToScreen(v3(0.4, 0, 0))[k] - N.a.localToScreen(v3(0, 0, 0))[k]));
  assert.ok(Math.abs(dNear / dFar - 3) < 0.05, `scale ratio ${(dNear / dFar).toFixed(3)}`);
});

// ----------------------------------------------------------------- filtering
test('OneEuroFilter suppresses jitter on a stationary signal', () => {
  const f = new OneEuroFilter({ minCutoff: 0.6, beta: 0.01 });
  let s = 7;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff) * 2 - 1; };
  const raw = [], filt = [];
  for (let i = 0; i < 300; i++) {
    const x = 100 + 3 * rnd();          // stationary + noise
    raw.push(x);
    filt.push(f.filter(x, i / 60));
  }
  const sd = (a) => {
    const w = a.slice(100);
    const mu = w.reduce((p, c) => p + c, 0) / w.length;
    return Math.sqrt(w.reduce((p, c) => p + (c - mu) ** 2, 0) / w.length);
  };
  assert.ok(sd(filt) < sd(raw) * 0.35,
    `jitter ${sd(filt).toFixed(3)} vs raw ${sd(raw).toFixed(3)}`);
});

test('OneEuroFilter keeps lag low on fast motion (the point of 1-euro)', () => {
  const mk = (beta) => new OneEuroFilter({ minCutoff: 0.6, beta });
  const ramp = (f) => {
    let out = 0;
    for (let i = 0; i < 120; i++) out = f.filter(i * 10, i / 60); // fast ramp
    return out;
  };
  const target = 119 * 10;
  const lagPlain = target - ramp(mk(0.0));     // EMA-like, no speed adaptation
  const lagAdaptive = target - ramp(mk(0.05)); // speed-adaptive
  assert.ok(lagAdaptive < lagPlain,
    `adaptive lag ${lagAdaptive.toFixed(1)} should beat ${lagPlain.toFixed(1)}`);
});

test('OneEuroFilter stays finite and bounded across a dropped-frame gap', () => {
  // A long stall (backgrounded tab) must not produce NaN/Inf or overshoot.
  // Note it *is* expected to move most of the way to the new sample: after a
  // multi-second gap the face genuinely has moved, and 1-euro is a motion
  // tracker, not an outlier rejector. Outlier rejection is MouthARAnchor's job
  // (see the gating test below) -- an earlier version of this test asserted
  // the opposite and was simply wrong about what the filter is for.
  const f = new OneEuroFilter({ minCutoff: 0.6, beta: 0.01 });
  for (let i = 0; i < 60; i++) f.filter(100, i / 60);
  const after = f.filter(400, 5.0);
  assert.ok(Number.isFinite(after), 'must stay finite');
  assert.ok(after > 100 && after <= 400, `must stay within the span, got ${after}`);
});

test('anchor rejects an implausible jump, then re-anchors if it persists', () => {
  const t = new MouthTracker();
  const good = t.track(synthesiseLandmarks({ centerPx: { x: 500, y: 300 } }), W, H);
  const a = new MouthARAnchor({ smoothing: 'responsive', gating: true });
  for (let i = 0; i < 60; i++) a.update(good, i / 60);
  const settled = a.getPose().origin.x;

  // one-frame teleport across the screen: must be vetoed
  const jump = t.track(synthesiseLandmarks({ centerPx: { x: 1100, y: 300 } }), W, H);
  const held = a.update(jump, 61 / 60);
  assert.ok(Math.abs(held.origin.x - settled) < 5,
    `single outlier must be held off, moved to ${held.origin.x.toFixed(1)}`);
  assert.ok(a.rejectedFrames > 0, 'rejection should be counted');

  // but if the subject really is there, the anchor must follow rather than
  // stay stuck forever
  let pose;
  for (let i = 62; i < 160; i++) pose = a.update(jump, i / 60);
  assert.ok(Math.abs(pose.origin.x - 1100) < 15,
    `must re-anchor to the sustained position, got ${pose.origin.x.toFixed(1)}`);
});

test('gating does not impede ordinary fast head motion', () => {
  const t = new MouthTracker();
  const a = new MouthARAnchor({ smoothing: 'responsive', gating: true });
  let pose;
  // sweep the mouth across the frame at ~14 px/frame (a brisk head turn)
  for (let i = 0; i < 120; i++) {
    const m = t.track(synthesiseLandmarks({ centerPx: { x: 300 + i * 6, y: 320 } }), W, H);
    pose = a.update(m, i / 60);
  }
  assert.equal(a.rejectedFrames, 0, 'normal motion must never be rejected');
  assert.ok(Math.abs(pose.origin.x - (300 + 119 * 6)) < 25,
    `should keep up with the sweep, at ${pose.origin.x.toFixed(1)}`);
});

test('PoseSmoother handles quaternion sign flips without swinging', () => {
  const s = new PoseSmoother('responsive');
  const b = orthonormalBasis(v3(1, 0, 0), v3(0, 1, 0));
  const q = basisToQuaternion(b);
  const pose = (qq) => ({ origin: v3(10, 20, 0), quaternion: qq, scale: 100, mouthOpen: 0 });
  let out;
  for (let i = 0; i < 40; i++) out = s.smooth(pose(q), i / 60);
  const before = { ...out.quaternion };
  // same rotation, negated representation
  const neg = { w: -q.w, x: -q.x, y: -q.y, z: -q.z };
  for (let i = 40; i < 50; i++) out = s.smooth(pose(neg), i / 60);
  const dot4 = before.w * out.quaternion.w + before.x * out.quaternion.x
    + before.y * out.quaternion.y + before.z * out.quaternion.z;
  assert.ok(Math.abs(dot4) > 0.999,
    `orientation must be unchanged by a sign flip, |dot|=${Math.abs(dot4).toFixed(5)}`);
});

test('anchor coasts briefly through dropouts, then reports lost', () => {
  const m = new MouthTracker().track(synthesiseLandmarks({}), W, H);
  const a = new MouthARAnchor({ smoothing: 'balanced' });
  for (let i = 0; i < 30; i++) a.update(m, i / 60);
  assert.ok(a.isValid());
  for (let i = 0; i < 5; i++) assert.ok(a.update(null, (30 + i) / 60), 'coasts');
  a.update(null, 36 / 60);
  assert.equal(a.getPose(), null, 'declares lost after the coast window');
});

test('end-to-end: noisy landmarks give a materially steadier anchor', () => {
  const t = new MouthTracker();
  const raw = new MouthARAnchor({ smoothing: 'responsive' });
  const sm = new MouthARAnchor({ smoothing: 'smooth' });
  const rawXs = [], smXs = [];
  for (let i = 0; i < 240; i++) {
    const lm = synthesiseLandmarks({ noisePx: 2.5, seed: i + 1 });
    const m = t.track(lm, W, H);
    const pr = raw.update(m, i / 60);
    const ps = sm.update(m, i / 60);
    if (i > 120) { rawXs.push(pr.origin.x); smXs.push(ps.origin.x); }
  }
  const sd = (a) => {
    const mu = a.reduce((p, c) => p + c, 0) / a.length;
    return Math.sqrt(a.reduce((p, c) => p + (c - mu) ** 2, 0) / a.length);
  };
  assert.ok(sd(smXs) < sd(rawXs), `smooth ${sd(smXs).toFixed(3)} < responsive ${sd(rawXs).toFixed(3)}`);
});
