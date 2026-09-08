/**
 * 3D tooth-anchor verification.
 *
 * The claim under test is specifically "these are real per-tooth 3D anchors,
 * not 2D boxes with a z bolted on". Three properties separate those cases, and
 * a fake implementation fails all three:
 *
 *   1. ROUND-TRIP  — an anchor projected back to the image must land on the
 *      tooth it was built from. Catches transform/compose/invert errors.
 *
 *   2. RIGIDITY    — teeth are rigid relative to the skull, so a tooth's
 *      position in the FACE frame must stay put as the head rotates. This is
 *      checked against a forward-simulated scene: ground-truth 3D teeth are
 *      projected to synthetic 2D observations, those observations are fed to
 *      the estimator, and the recovered 3D is compared to the truth. A control
 *      run without the head matrix (the "2D-ish" implementation) is measured
 *      alongside, so the difference is demonstrated rather than asserted.
 *
 *   3. PROVENANCE  — every axis says whether it was tracked, measured,
 *      estimated or assumed. Depth must never claim to be measured.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ToothPoseEstimator, intrinsicsForFrame, basisFromHeadMatrix, headMatrixDepthM,
  PROVENANCE, DEPTH_SOURCE,
} from '../src/core/ToothPoseEstimator.js';
import { Tooth3DAnchor, Tooth3DAnchorSet } from '../src/core/Tooth3DAnchor.js';
import {
  compose, invertRigid, multiply, projectPoint, transformPoint,
} from '../src/core/math/mat4.js';
import { orthonormalBasis, v3 } from '../src/core/math/vec3.js';

const W = 1280, H = 720;
const intr = intrinsicsForFrame(W, H);
const DEG = Math.PI / 180;
const MOUTH_W = 0.050;               // true mouth width used by the simulator

// ===========================================================================
// A synthetic scene: ground-truth 3D teeth, projected to 2D observations.
// ===========================================================================

/** Face basis in CAMERA coordinates for a given yaw/pitch, frontal = C. */
function faceBasisCam(yawDeg = 0, pitchDeg = 0) {
  // Frontal: face +X -> cam +X, face +Y(up) -> cam -Y, face +Z(out) -> cam -Z.
  const F0 = { x: v3(1, 0, 0), y: v3(0, -1, 0), z: v3(0, 0, -1) };
  const rotY = (v, a) => v3(Math.cos(a) * v.x + Math.sin(a) * v.z, v.y,
    -Math.sin(a) * v.x + Math.cos(a) * v.z);
  const rotX = (v, a) => v3(v.x, Math.cos(a) * v.y - Math.sin(a) * v.z,
    Math.sin(a) * v.y + Math.cos(a) * v.z);
  const R = (v) => rotX(rotY(v, yawDeg * DEG), pitchDeg * DEG);
  return { x: R(F0.x), y: R(F0.y), z: R(F0.z) };
}

/** The MediaPipe-style 4x4 that would produce that pose (GL frame, cm). */
function headMatrix(yawDeg = 0, pitchDeg = 0, T = v3(0, 0, 0.30)) {
  const F = faceBasisCam(yawDeg, pitchDeg);
  // basisFromHeadMatrix left-multiplies by C = diag(1,-1,-1); C is its own
  // inverse, so undo it here to build the matrix MediaPipe would hand us.
  const c = (v) => v3(v.x, -v.y, -v.z);
  const gx = c(F.x), gy = c(F.y), gz = c(F.z), gt = c(T);
  return new Float32Array([
    gx.x, gx.y, gx.z, 0,
    gy.x, gy.y, gy.z, 0,
    gz.x, gz.y, gz.z, 0,
    gt.x * 100, gt.y * 100, gt.z * 100, 1,   // MediaPipe's model is in cm
  ]);
}

const project = (p) => ({
  x: intr.cx + (intr.fx * p.x) / p.z,
  y: intr.cy + (intr.fy * p.y) / p.z,
});

/**
 * Build the 2D observations a perfect tracker would report for a scene.
 * @returns {{mouthPose, tracks, faceToCamera, truth}}
 */
function simulate({ yaw = 0, pitch = 0, T = v3(0, 0, 0.30),
                    teeth = [[-0.30, -0.10], [0, -0.12], [0.28, -0.08]],
                    archRatio = 0.28 } = {}) {
  const F = faceBasisCam(yaw, pitch);
  const toCam = (p) => v3(
    T.x + F.x.x * p.x + F.y.x * p.y + F.z.x * p.z,
    T.y + F.x.y * p.x + F.y.y * p.y + F.z.y * p.z,
    T.z + F.x.z * p.x + F.y.z * p.y + F.z.z * p.z,
  );
  const archZ = (u) => -archRatio * (2 * u) * (2 * u);
  // ground-truth positions in the face frame (metres, +Y up)
  const truth = teeth.map(([u, v]) =>
    v3(u * MOUTH_W, -v * MOUTH_W, archZ(u) * MOUTH_W));

  // observed mouth: corners at u = +/-0.5 on the same arch
  const pxL = project(toCam(v3(-0.5 * MOUTH_W, 0, archZ(-0.5) * MOUTH_W)));
  const pxR = project(toCam(v3(0.5 * MOUTH_W, 0, archZ(0.5) * MOUTH_W)));
  const pxO = project(toCam(v3(0, 0, 0)));
  const widthPx = Math.hypot(pxR.x - pxL.x, pxR.y - pxL.y);
  const bx = { x: (pxR.x - pxL.x) / widthPx, y: (pxR.y - pxL.y) / widthPx };
  const by = { x: -bx.y, y: bx.x };          // image-plane "down"

  const mouthPose = {
    origin: { x: pxO.x, y: pxO.y, z: 0 },
    basis: { x: v3(bx.x, bx.y, 0), y: v3(by.x, by.y, 0), z: v3(0, 0, 1) },
    scale: widthPx,
    mouthOpen: 0.4,
  };

  // what the 2D tracker would report for each tooth
  const tracks = truth.map((p, i) => {
    const px = project(toCam(p));
    const dx = (px.x - pxO.x) / widthPx, dy = (px.y - pxO.y) / widthPx;
    return {
      id: i + 1, arch: 'upper', status: 'stable',
      smoothed: {
        center: { u: dx * bx.x + dy * bx.y, v: dx * by.x + dy * by.y },
        box: { u: 0, v: 0, w: 0.09, h: 0.11 },
        confidence: 0.8,
      },
    };
  });

  const faceToCamera = compose(T, F, 1);
  return { mouthPose, tracks, faceToCamera, truth, headMatrix: headMatrix(yaw, pitch, T) };
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const mm = (m) => `${(m * 1000).toFixed(2)} mm`;

// ===========================================================================
// mat4 / conversion basics
// ===========================================================================
test('mat4 compose/invert round-trips a rigid transform', () => {
  const b = orthonormalBasis(v3(1, 0.3, -0.2), v3(-0.1, 1, 0.25));
  const m = compose(v3(0.03, -0.02, 0.4), b, 1);
  const back = transformPoint(multiply(invertRigid(m), m), v3(0.01, 0.02, 0.03));
  assert.ok(dist(back, v3(0.01, 0.02, 0.03)) < 1e-6, 'inv(M)*M must be identity');
});

test('projectPoint rejects points behind the camera', () => {
  assert.equal(projectPoint({ x: 0, y: 0, z: -0.1 }, intr), null);
  assert.ok(projectPoint({ x: 0, y: 0, z: 0.4 }, intr));
});

test('MediaPipe GL head matrix converts into the OpenCV camera frame', () => {
  const b = basisFromHeadMatrix(headMatrix(0));
  // frontal face: +X right, +Y(up) -> camera -Y, +Z(out) -> camera -Z
  assert.ok(dist(b.x, v3(1, 0, 0)) < 1e-6, `x: ${JSON.stringify(b.x)}`);
  assert.ok(dist(b.y, v3(0, -1, 0)) < 1e-6, `y: ${JSON.stringify(b.y)}`);
  assert.ok(dist(b.z, v3(0, 0, -1)) < 1e-6, `z: ${JSON.stringify(b.z)}`);
});

test('head-matrix depth accepts centimetres or metres and rejects nonsense', () => {
  assert.ok(Math.abs(headMatrixDepthM(headMatrix(0, 0, v3(0, 0, 0.3))) - 0.3) < 1e-6);
  const metres = headMatrix(0, 0, v3(0, 0, 0.3)); metres[14] = -0.3;   // metres build
  assert.ok(Math.abs(headMatrixDepthM(metres) - 0.3) < 1e-6);
  const absurd = headMatrix(0); absurd[14] = -900000;
  assert.equal(headMatrixDepthM(absurd), null, 'implausible depth must be refused');
});

// ===========================================================================
// The anchors are genuinely 3D
// ===========================================================================
test('each tooth gets its OWN distinct transform, and the arch actually curves', () => {
  const est = new ToothPoseEstimator();
  const sim = simulate();
  const poses = sim.tracks.map((t) => est.estimate(t, sim.mouthPose, sim.headMatrix, intr));

  assert.equal(poses.length, 3);
  const xs = poses.map((p) => p.positionCamera.x.toFixed(6));
  assert.equal(new Set(xs).size, 3, 'a distinct X per tooth');
  const zs = poses.map((p) => p.positionCamera.z);
  assert.ok(zs[0] > zs[1] && zs[2] > zs[1],
    `outer teeth must sit further from the camera: ${zs.map((z) => z.toFixed(4))}`);
  // and the matrices themselves must differ, not just a position field
  assert.notDeepEqual([...poses[0].toothToCamera], [...poses[2].toothToCamera]);
});

test('ROUND-TRIP: every anchor reprojects onto the tooth it was built from', () => {
  const est = new ToothPoseEstimator();
  let worst = 0;
  for (const yaw of [-20, 0, 15]) {
    const sim = simulate({ yaw });
    for (const t of sim.tracks) {
      const a = new Tooth3DAnchor(est.estimate(t, sim.mouthPose, sim.headMatrix, intr));
      const scr = a.toScreen({ x: 0, y: 0, z: 0 }, intr);
      assert.ok(scr, 'anchor must be in front of the camera');
      const s = t.smoothed;
      const exp = {
        x: sim.mouthPose.origin.x + sim.mouthPose.scale
          * (s.center.u * sim.mouthPose.basis.x.x + s.center.v * sim.mouthPose.basis.y.x),
        y: sim.mouthPose.origin.y + sim.mouthPose.scale
          * (s.center.u * sim.mouthPose.basis.x.y + s.center.v * sim.mouthPose.basis.y.y),
      };
      worst = Math.max(worst, Math.hypot(scr.x - exp.x, scr.y - exp.y));
    }
  }
  // Exact by construction: the anchor is placed on the measured viewing ray,
  // so only its distance along that ray is inferred.
  assert.ok(worst < 0.01, `reprojection error ${worst.toFixed(4)} px must be ~0`);
});

test('RIGIDITY: recovered teeth stay put in face space while the head turns', () => {
  const est = new ToothPoseEstimator();
  const control = new ToothPoseEstimator();   // same code, no head matrix
  const yaws = [-25, -15, 0, 15, 25];
  const real = [[], [], []];
  const naive = [[], [], []];

  for (const yaw of yaws) {
    const sim = simulate({ yaw });
    const camToFaceTruth = invertRigid(sim.faceToCamera);
    sim.tracks.forEach((t, i) => {
      const p = est.estimate(t, sim.mouthPose, sim.headMatrix, intr);
      real[i].push(transformPoint(camToFaceTruth, p.positionCamera));
      // Control: no head pose at all — the mouth's 2D basis stands in for 3D
      // orientation, which is what a "2D box with a z" implementation does.
      const q = control.estimate(t, sim.mouthPose, null, intr);
      naive[i].push(transformPoint(camToFaceTruth, q.positionCamera));
    });
  }

  const spread = (pts) => {
    const n = pts.length;
    const m = pts.reduce((a, p) => v3(a.x + p.x / n, a.y + p.y / n, a.z + p.z / n), v3());
    return Math.max(...pts.map((p) => dist(p, m)));
  };

  const realDrift = Math.max(...real.map(spread));
  const naiveDrift = Math.max(...naive.map(spread));

  // The head-tracked anchors barely move in the skull's own frame.
  assert.ok(realDrift < 0.003,
    `tracked anchors must be rigid in face space; drifted ${mm(realDrift)}`);
  // And they are decisively better than the orientation-free version, which is
  // the point: this is 3D anchoring, not a repositioned 2D overlay.
  assert.ok(naiveDrift > realDrift * 5,
    `control should smear (${mm(naiveDrift)}) vs tracked (${mm(realDrift)})`);
});

test('ACCURACY vs ground truth: recovered 3D matches the simulated teeth', () => {
  const est = new ToothPoseEstimator();
  let worst = 0;
  for (const yaw of [-20, 0, 20]) {
    const sim = simulate({ yaw });
    const camToFaceTruth = invertRigid(sim.faceToCamera);
    sim.tracks.forEach((t, i) => {
      const p = est.estimate(t, sim.mouthPose, sim.headMatrix, intr);
      worst = Math.max(worst, dist(transformPoint(camToFaceTruth, p.positionCamera), sim.truth[i]));
    });
  }
  // NOTE: this measures the geometry pipeline only. The simulator uses the
  // same arch prior the estimator assumes, so a small residual here means the
  // transforms are right — it says nothing about how well the prior matches a
  // real mouth, which is the dominant real-world error and is not measurable
  // from RGB at all.
  assert.ok(worst < 0.003, `pipeline error vs ground truth: ${mm(worst)}`);
});

test('orientation follows the head, not the image plane', () => {
  const est = new ToothPoseEstimator();
  const s0 = simulate({ yaw: 0 }), s1 = simulate({ yaw: 30 });
  const a = new Tooth3DAnchor(est.estimate(s0.tracks[2], s0.mouthPose, s0.headMatrix, intr));
  const b = new Tooth3DAnchor(est.estimate(s1.tracks[2], s1.mouthPose, s1.headMatrix, intr));
  const d = Math.abs(b.rotationEuler.ry - a.rotationEuler.ry);
  assert.ok(d > 20, `a 30 deg head yaw must rotate the anchor; got ${d.toFixed(1)} deg`);
});

test('depth responds to distance, and physical tooth size does not', () => {
  const est = new ToothPoseEstimator();
  const near = simulate({ T: v3(0, 0, 0.20) });
  const far = simulate({ T: v3(0, 0, 0.50) });
  const pn = est.estimate(near.tracks[1], near.mouthPose, near.headMatrix, intr);
  const pf = est.estimate(far.tracks[1], far.mouthPose, far.headMatrix, intr);
  assert.ok(pf.positionCamera.z > pn.positionCamera.z * 2,
    `leaning back must increase depth: ${pn.positionCamera.z.toFixed(3)} -> ${pf.positionCamera.z.toFixed(3)}`);
  // Metric size is apparent size x distance, so it must stay put as the
  // subject moves. The residual is pure perspective on the mouth corners
  // (the arch is not a fronto-parallel segment), not a scale drift.
  const drift = Math.abs(pn.scale.x - pf.scale.x) / pn.scale.x;
  assert.ok(drift < 0.005,
    `a real tooth does not shrink when you lean back; drifted ${(drift * 100).toFixed(2)}%`);
});

// ===========================================================================
// Honesty of the numbers
// ===========================================================================
test('every pose declares what is tracked / measured / estimated / assumed', () => {
  const est = new ToothPoseEstimator();
  const sim = simulate();
  const p = est.estimate(sim.tracks[0], sim.mouthPose, sim.headMatrix, intr);
  assert.equal(p.provenance.position_xy, PROVENANCE.MEASURED);
  assert.equal(p.provenance.position_z, PROVENANCE.ESTIMATED, 'depth is NOT measured');
  assert.equal(p.provenance.orientation, PROVENANCE.TRACKED);
  assert.equal(p.provenance.scale_z, PROVENANCE.ESTIMATED);
  assert.equal(p.provenance.metric_scale, PROVENANCE.ASSUMED);
  assert.equal(p.depthSource, DEPTH_SOURCE.HEAD_MATRIX);

  // Degraded input must downgrade its own claims rather than stay silent.
  const noHead = est.estimate(sim.tracks[0], sim.mouthPose, null, intr);
  assert.equal(noHead.provenance.orientation, PROVENANCE.ESTIMATED);
  assert.equal(noHead.depthSource, DEPTH_SOURCE.APPARENT_WIDTH);
});

test('metric output scales linearly with the assumed mouth width', () => {
  const sim = simulate();
  const a = new ToothPoseEstimator({ assumedMouthWidthM: 0.050, useHeadMatrixDepth: false });
  const b = new ToothPoseEstimator({ assumedMouthWidthM: 0.100, useHeadMatrixDepth: false });
  const pa = a.estimate(sim.tracks[0], sim.mouthPose, sim.headMatrix, intr);
  const pb = b.estimate(sim.tracks[0], sim.mouthPose, sim.headMatrix, intr);
  const ratio = pb.positionCamera.z / pa.positionCamera.z;
  assert.ok(Math.abs(ratio - 2) < 1e-6,
    `doubling the assumption must double metric depth, got ${ratio.toFixed(4)}`);
});

test('the depth prior is a knob, not a hidden constant', () => {
  const sim = simulate({ archRatio: 0.28 });
  const flat = new ToothPoseEstimator({ archDepthRatio: 0 });
  const curved = new ToothPoseEstimator({ archDepthRatio: 0.28 });
  const zf = sim.tracks.map((t) => flat.estimate(t, sim.mouthPose, sim.headMatrix, intr).positionCamera.z);
  const zc = sim.tracks.map((t) => curved.estimate(t, sim.mouthPose, sim.headMatrix, intr).positionCamera.z);
  assert.ok(Math.max(...zf) - Math.min(...zf) < 1e-6, 'ratio 0 must give a flat plane');
  assert.ok(Math.max(...zc) - Math.min(...zc) > 0.002, 'the prior must actually bend the arch');
});

// ===========================================================================
// Anchor lifetime and the attachment seam
// ===========================================================================
test('anchors persist per tooth ID and are released when the tooth goes', () => {
  const est = new ToothPoseEstimator();
  const set = new Tooth3DAnchorSet();
  const sim = simulate();
  const poses = sim.tracks.map((t) => est.estimate(t, sim.mouthPose, sim.headMatrix, intr));

  set.update(poses);
  assert.equal(set.size, 3);
  const a2 = set.get(2);
  assert.ok(a2);

  set.update(poses.slice(0, 2));
  assert.equal(set.size, 2, "a vanished tooth's anchor must be released");
  assert.equal(set.get(2), a2, 'surviving anchors keep their identity');
  assert.equal(set.get(3), null);
});

test('an anchor can carry an attached 3D model (the Step-4 seam)', () => {
  const est = new ToothPoseEstimator();
  const sim = simulate();
  const a = new Tooth3DAnchor(est.estimate(sim.tracks[0], sim.mouthPose, sim.headMatrix, intr));
  const mesh = { kind: 'tooth-mesh', vertices: [] };
  assert.equal(a.attach(mesh).attachment, mesh);
  // content authored in tooth-local metres maps into camera space and back
  const tip = { x: 0, y: 0.004, z: 0.002 };
  const back = a.fromCamera(a.toCamera(tip));
  assert.ok(dist(back, tip) < 1e-6, 'tooth-local <-> camera must round-trip');
});

test('toJSON exposes the transform, the tracking state and the provenance', () => {
  const est = new ToothPoseEstimator();
  const sim = simulate();
  const a = new Tooth3DAnchor(est.estimate(sim.tracks[1], sim.mouthPose, sim.headMatrix, intr));
  const j = a.toJSON();
  for (const k of ['tooth_id', 'arch', 'position_m', 'rotation_deg', 'scale_m',
    'confidence', 'tracking', 'provenance', 'depth_source']) {
    assert.ok(k in j, `toJSON must include ${k}`);
  }
  assert.equal(j.tooth_id, 2);
});

test('face-frame rotation reads as arch splay, not a 180 deg rest pose', () => {
  const est = new ToothPoseEstimator();
  const sim = simulate({ teeth: [[0, -0.1], [0.35, -0.1]] });
  const mid = new Tooth3DAnchor(est.estimate(sim.tracks[0], sim.mouthPose, sim.headMatrix, intr));
  const outer = new Tooth3DAnchor(est.estimate(sim.tracks[1], sim.mouthPose, sim.headMatrix, intr));
  const r = mid.rotationInFace;
  assert.ok(Math.abs(r.rx) < 1 && Math.abs(r.ry) < 1 && Math.abs(r.rz) < 1,
    `a central tooth must sit near zero in the face frame, got ${JSON.stringify(r)}`);
  // an outer tooth is turned away from the midline by the arch
  assert.ok(Math.abs(outer.rotationInFace.ry) > 10,
    `an outer tooth must splay; got ${outer.rotationInFace.ry.toFixed(1)} deg`);
  // and it stays a face-frame quantity: turning the head barely changes it.
  // Not exactly, and honestly so — splay comes from the arch prior evaluated
  // at the tooth's OBSERVED position across the mouth, and a yawed head shifts
  // that observation slightly. The residual is the prior's, not the frame's.
  const t2 = simulate({ yaw: 25, teeth: [[0, -0.1], [0.35, -0.1]] });
  const turned = new Tooth3DAnchor(est.estimate(t2.tracks[1], t2.mouthPose, t2.headMatrix, intr));
  const d = Math.abs(turned.rotationInFace.ry - outer.rotationInFace.ry);
  assert.ok(d < 8, `splay belongs to the tooth, not the head direction; moved ${d.toFixed(1)} deg`);
});
