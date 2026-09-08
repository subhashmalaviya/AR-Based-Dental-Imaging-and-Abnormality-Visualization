/**
 * ToothPoseEstimator.js — turns a tracked 2D tooth into a full 3D pose.
 *
 * =========================================================================
 * WHAT IS MEASURED, WHAT IS ESTIMATED, WHAT IS ASSUMED
 * =========================================================================
 * An RGB camera cannot measure the depth of a tooth. Nothing in this file
 * pretends otherwise. Every field of the resulting pose carries a
 * `provenance` tag so a caller — and the UI — can tell them apart:
 *
 *   'tracked'   MediaPipe's own per-frame output. The head's 3D rotation comes
 *               from its facial transformation matrix, a real fit of a 3D face
 *               model to this image.
 *
 *   'measured'  Derived from actual image pixels this frame. The tooth's
 *               VIEWING RAY and its apparent size come from the segmenter's
 *               contour (ToothSegmenter), not from a template. The anchor is
 *               placed *on* that ray, so it reprojects exactly onto the tooth
 *               that was segmented.
 *
 *   'estimated' Model-based inference, not observation. Only the DISTANCE
 *               ALONG the measured ray is estimated: the mouth's depth, plus a
 *               parametric dental-arch offset (a human arch curves back toward
 *               the molars, so a tooth further from the midline sits further
 *               from the camera). A plausible anatomical prior, NOT a
 *               measurement of this person.
 *
 *   'assumed'   A fixed constant. Absolute scale is unobservable from one RGB
 *               camera (a big mouth far away and a small mouth near by are
 *               pixel-identical), so metric depth is ultimately pinned by an
 *               average adult face/mouth size. Every metric distance inherits
 *               that assumption's error — roughly +/-10% between adults.
 *
 * Consequently this is NOT medical-grade registration and must not be
 * presented as such. It is a correct 3D *framework* — real transforms, real
 * coordinate spaces, real head-pose tracking, exact reprojection — with an
 * anatomically-motivated depth prior standing in for the depth sensing a later
 * stage would provide (CBCT registration, a TrueDepth/LiDAR camera, or stereo).
 *
 * The split is deliberate and is what makes the prior replaceable: swap
 * `archDepthAt()` and the mouth-depth source for real measurements and every
 * transform downstream keeps working unchanged.
 *
 * =========================================================================
 * COORDINATE SPACES
 * =========================================================================
 *   camera      metres, OpenCV convention: +X right, +Y DOWN, +Z into the
 *               scene (away from the viewer). This is what `projectPoint`
 *               assumes and what the canvas draws.
 *
 *   face        metric, rigid to the skull, right-handed model convention:
 *               +X to image right, +Y UP, +Z out of the face toward the
 *               camera. Origin at the mouth centre. Teeth are STATIC in this
 *               frame — that is exactly what makes it the right frame to
 *               anchor in, and it is what the rigidity test checks.
 *
 *               NOTE this is not the same handedness as MouthARAnchor's 2D
 *               pixel-space mouth-local frame (+Y down, units of mouth width).
 *               That frame is a weak-perspective screen convenience; this one
 *               is metric 3D. `estimate()` is the only place they meet, and it
 *               converts explicitly (v -> -Y, mouth widths -> metres).
 *
 *   tooth-local metres, right-handed: +X across the arch, +Y along the long
 *               axis from crown toward root, +Z out of the labial (outward)
 *               surface. Origin at the crown centre. THIS is where a future 3D
 *               tooth mesh, root canal or lesion volume attaches.
 *
 * MediaPipe reports its facial transformation matrix in an OpenGL-style frame
 * (+Y up, camera looking down -Z). `basisFromHeadMatrix` converts it into the
 * OpenCV camera frame above with C = diag(1, -1, -1); that conversion is the
 * one place a sign error would silently mirror everything, so it is isolated
 * and unit-tested.
 */
import { compose, invertRigid, transformPoint } from './math/mat4.js';
import { cross, dot, normalize, scale as vscale, sub, v3 } from './math/vec3.js';

/** MediaPipe's face-geometry camera assumes a 63 degree vertical FOV. */
export const MEDIAPIPE_VFOV_DEG = 63;

export function intrinsicsForFrame(width, height, vfovDeg = MEDIAPIPE_VFOV_DEG) {
  const fy = (height / 2) / Math.tan((vfovDeg * Math.PI) / 360);
  return { fx: fy, fy, cx: width / 2, cy: height / 2, width, height };
}

export const PROVENANCE = {
  TRACKED: 'tracked',
  MEASURED: 'measured',
  ESTIMATED: 'estimated',
  ASSUMED: 'assumed',
};

/** How the mouth-plane distance was obtained this frame. */
export const DEPTH_SOURCE = {
  HEAD_MATRIX: 'mediapipe-metric-head-model',
  APPARENT_WIDTH: 'apparent-mouth-width',
};

/** OpenGL(MediaPipe) -> OpenCV(camera) axis flip: +Y up -> +Y down, +Z out -> +Z in. */
const GL_TO_CV = { x: 1, y: -1, z: -1 };

export class ToothPoseEstimator {
  constructor(opts = {}) {
    // Average adult inter-commissure (mouth) width. With no head matrix this
    // is the single constant converting pixels to metres; it is an ASSUMPTION,
    // and all metric output scales linearly with it.
    this.assumedMouthWidthM = opts.assumedMouthWidthM ?? 0.050;

    // Dental-arch depth prior: how far the arch curves back, in mouth widths,
    // going from the midline out to the corner of the mouth.
    this.archDepthRatio = opts.archDepthRatio ?? 0.28;

    // Crown thickness (labial-lingual), as a fraction of mouth width.
    this.crownThicknessRatio = opts.crownThicknessRatio ?? 0.09;

    // Use MediaPipe's metric head translation for the mouth distance when it
    // looks sane. It is far more stable under head yaw than depth-from-
    // apparent-width, because a yawed mouth is foreshortened in the image.
    this.useHeadMatrixDepth = opts.useHeadMatrixDepth ?? true;
    this.depthRangeM = opts.depthRangeM ?? [0.08, 2.5];

    this.vfovDeg = opts.vfovDeg ?? MEDIAPIPE_VFOV_DEG;
    this.lastDepthSource = null;
  }

  setParams(p) { Object.assign(this, p); }

  /**
   * Outward (+Z) offset of the dental arch at position u, in mouth widths.
   * Parabolic: 0 at the midline, curving away from the camera toward the
   * corners. Negative = further from the camera, behind the incisal plane.
   */
  archDepthAt(u) {
    return -this.archDepthRatio * (2 * u) * (2 * u);
  }

  /** Outward (labial) normal of the arch at u, in face-frame axes. */
  archNormalAt(u) {
    // arch curve z(u) = -k*(2u)^2  ->  dz/du = -8k*u
    const dzdu = -8 * this.archDepthRatio * u;
    // tangent (1, 0, dz/du); the outward normal is perpendicular to it in the
    // XZ plane with a positive Z component.
    return normalize(v3(-dzdu, 0, 1));
  }

  /**
   * Distance from the camera to the mouth plane, in metres.
   * Prefers MediaPipe's metric head translation; falls back to apparent size.
   * @returns {{depthM:number, source:string}}
   */
  mouthDepth(mouthPose, headMatrix) {
    if (this.useHeadMatrixDepth && headMatrix) {
      const z = headMatrixDepthM(headMatrix, this.depthRangeM);
      if (z !== null) return { depthM: z, source: DEPTH_SOURCE.HEAD_MATRIX };
    }
    // widthPx = f * widthM / Z  ->  Z = f * widthM / widthPx
    return { depthM: null, source: DEPTH_SOURCE.APPARENT_WIDTH };
  }

  /**
   * @param {object} track      a ToothTrack (uses .smoothed when present)
   * @param {object} mouthPose  MouthARAnchor pose (pixel space, mouth widths)
   * @param {Float32Array|null} headMatrix  MediaPipe 4x4 facial transform
   * @param {object} intr       camera intrinsics for this frame
   * @returns {object|null} a 3D pose with provenance
   */
  estimate(track, mouthPose, headMatrix, intr) {
    if (!track || !mouthPose || !intr) return null;
    const s = track.smoothed ?? track;
    const mouthWidthPx = mouthPose.scale;
    if (!(mouthWidthPx > 1e-6)) return null;

    // --- distance to the mouth plane -----------------------------------
    let { depthM: mouthDepthM, source: depthSource } =
      this.mouthDepth(mouthPose, headMatrix);
    if (mouthDepthM === null) {
      mouthDepthM = (intr.fx * this.assumedMouthWidthM) / mouthWidthPx;
      depthSource = DEPTH_SOURCE.APPARENT_WIDTH;
    }
    this.lastDepthSource = depthSource;

    // --- rotation of the face frame in camera space ---------------------
    // TRACKED when MediaPipe gives a head matrix, so anchors rotate with the
    // real head rather than with a 2D bounding box.
    const faceBasisCam = headMatrix
      ? basisFromHeadMatrix(headMatrix)
      : basisFromMouthPose(mouthPose);

    // --- metres per mouth-width -----------------------------------------
    // Apparent size at the estimated distance, DIVIDED BY the foreshortening
    // of the mouth's width axis. That correction matters: a turned head makes
    // the mouth narrower in the image, and without it every tooth would shrink
    // as the subject looks away — and the 3D proxy would visibly drift off the
    // tooth it belongs to. `foreshorten` is the length of the face's +X axis
    // after projection onto the image plane, so it comes from the same tracked
    // rotation used everywhere else. Only the *distance* varies per frame; a
    // mouth does not change size.
    const foreshorten = Math.max(0.35, Math.hypot(faceBasisCam.x.x, faceBasisCam.x.y));
    // The corners of the mouth are not on the mouth plane: the same arch prior
    // puts them further from the camera than the midline, which makes the
    // mouth look narrower than it is — noticeably so close up. Solve for the
    // size that is consistent with the corners' own depth (two fixed-point
    // iterations; the correction is a few percent and converges immediately).
    const cornerZ = this.archDepthAt(0.5) * faceBasisCam.z.z;
    let measuredMpw = (mouthDepthM * mouthWidthPx) / (intr.fx * foreshorten);
    for (let i = 0; i < 2; i++) {
      measuredMpw = ((mouthDepthM + cornerZ * measuredMpw) * mouthWidthPx)
        / (intr.fx * foreshorten);
    }
    // Guard: a bad head matrix must not be able to inflate the whole mouth.
    const metresPerMouthWidth =
      (measuredMpw > 0.020 && measuredMpw < 0.120) ? measuredMpw : this.assumedMouthWidthM;

    // --- the tooth in the face frame ------------------------------------
    const u = s.center.u;               // across the arch, MEASURED
    const vv = s.center.v;              // down the mouth, MEASURED
    const archZ = this.archDepthAt(u);  // outward offset, ESTIMATED
    // mouth-local (+Y down) -> face frame (+Y up)
    const posFace = v3(
      u * metresPerMouthWidth,
      -vv * metresPerMouthWidth,
      archZ * metresPerMouthWidth,
    );

    // --- position in camera space ---------------------------------------
    // The tooth is placed ON ITS MEASURED VIEWING RAY: only the distance along
    // that ray is estimated. So the anchor reprojects exactly onto the tooth
    // the segmenter found, while depth ordering still comes from the arch.
    const toothPx = localToPixel(mouthPose, u, vv);
    const offsetCam = rotateVector(faceBasisCam, posFace);
    const faceOriginCam = backProject(mouthPose.origin, mouthDepthM, intr);
    const toothDepthM = faceOriginCam.z + offsetCam.z;
    if (!(toothDepthM > 1e-3)) return null;      // behind the camera: reject
    const positionCam = backProject(toothPx, toothDepthM, intr);

    // --- orientation of the tooth ---------------------------------------
    // +Z follows the arch normal (ESTIMATED); +Y runs crown->root, which is up
    // for the upper arch and down for the lower.
    const zFace = this.archNormalAt(u);
    const yHint = v3(0, track.arch === 'lower' ? -1 : 1, 0);
    const yFace = normalize(sub(yHint, vscale(zFace, dot(yHint, zFace))));
    const xFace = cross(yFace, zFace);          // right-handed by construction
    const toothBasisFace = { x: xFace, y: yFace, z: zFace };
    const toothBasisCam = rotateBasis(faceBasisCam, toothBasisFace);

    // --- scale: width/height MEASURED, thickness ESTIMATED --------------
    const scaleM = {
      x: Math.max(1e-4, s.box.w * metresPerMouthWidth),
      y: Math.max(1e-4, s.box.h * metresPerMouthWidth),
      z: this.crownThicknessRatio * metresPerMouthWidth,
    };

    const faceToCamera = compose(faceOriginCam, faceBasisCam, 1);
    const toothToCamera = compose(positionCam, toothBasisCam, 1);

    return {
      id: track.id,
      arch: track.arch,
      // full transforms (column-major, WebGL / three.js ready)
      toothToCamera,
      faceToCamera,
      cameraToFace: invertRigid(faceToCamera),
      // convenience read-outs
      positionCamera: positionCam,
      positionFace: transformPoint(invertRigid(faceToCamera), positionCam),
      basisFace: toothBasisFace,
      scale: scaleM,
      mouthDepthM,
      metresPerMouthWidth,
      foreshorten,
      depthSource,
      pixel: toothPx,
      confidence: s.confidence,
      trackingState: track.status,
      provenance: {
        position_xy: PROVENANCE.MEASURED,    // viewing ray, from the contour
        position_z: PROVENANCE.ESTIMATED,    // arch prior + mouth distance
        orientation: headMatrix ? PROVENANCE.TRACKED : PROVENANCE.ESTIMATED,
        scale_xy: PROVENANCE.MEASURED,
        scale_z: PROVENANCE.ESTIMATED,
        metric_scale: PROVENANCE.ASSUMED,    // average adult face/mouth size
        depth_source: depthSource,
      },
    };
  }
}

/**
 * Rotation of the head from MediaPipe's facial transformation matrix,
 * converted from its OpenGL-style frame into our OpenCV camera frame.
 * Scale is divided out so a non-unit matrix cannot silently resize teeth.
 */
export function basisFromHeadMatrix(m) {
  const sx = Math.hypot(m[0], m[1], m[2]) || 1;
  const sy = Math.hypot(m[4], m[5], m[6]) || 1;
  const sz = Math.hypot(m[8], m[9], m[10]) || 1;
  // Rows are flipped (left-multiplication by C = diag(1,-1,-1)); the columns
  // stay the face model's own axes.
  return {
    x: v3(GL_TO_CV.x * m[0] / sx, GL_TO_CV.y * m[1] / sx, GL_TO_CV.z * m[2] / sx),
    y: v3(GL_TO_CV.x * m[4] / sy, GL_TO_CV.y * m[5] / sy, GL_TO_CV.z * m[6] / sy),
    z: v3(GL_TO_CV.x * m[8] / sz, GL_TO_CV.y * m[9] / sz, GL_TO_CV.z * m[10] / sz),
  };
}

/**
 * Camera-space distance to the head, from MediaPipe's translation.
 *
 * MediaPipe's canonical face model is metric but expressed in CENTIMETRES,
 * while some builds hand back metres; rather than guessing from the version we
 * pick the interpretation that lands in a physically sensible range and reject
 * the frame outright if neither does. Returns metres, or null.
 */
export function headMatrixDepthM(m, range = [0.08, 2.5]) {
  const zGl = m[14];
  if (!Number.isFinite(zGl) || zGl === 0) return null;
  const raw = Math.abs(zGl);                 // camera looks down -Z in GL
  for (const candidate of [raw / 100, raw]) { // centimetres, then metres
    if (candidate >= range[0] && candidate <= range[1]) return candidate;
  }
  return null;
}

/**
 * Fallback orientation when MediaPipe supplies no head matrix: the mouth
 * anchor's own basis, with its +Y-down / +Z-toward-camera screen convention
 * mapped into the face frame. Less reliable (it is fitted to lip landmarks in
 * the image plane), hence tagged 'estimated' rather than 'tracked'.
 */
function basisFromMouthPose(pose) {
  const b = pose.basis;
  return {
    x: v3(b.x.x, b.x.y, -b.x.z),
    y: v3(-b.y.x, -b.y.y, b.y.z),
    z: v3(-b.z.x, -b.z.y, b.z.z),
  };
}

/** Rotate a vector by a basis given as three camera-space column vectors. */
function rotateVector(B, v) {
  return v3(
    B.x.x * v.x + B.y.x * v.y + B.z.x * v.z,
    B.x.y * v.x + B.y.y * v.y + B.z.y * v.z,
    B.x.z * v.x + B.y.z * v.y + B.z.z * v.z,
  );
}

/** Compose two bases: express `b`'s columns in the frame `B` sits in. */
function rotateBasis(B, b) {
  return { x: rotateVector(B, b.x), y: rotateVector(B, b.y), z: rotateVector(B, b.z) };
}

/** Mouth-local (u, v) in mouth widths -> pixels, matching MouthARAnchor. */
function localToPixel(pose, u, v) {
  const { origin, basis, scale } = pose;
  return {
    x: origin.x + scale * (u * basis.x.x + v * basis.y.x),
    y: origin.y + scale * (u * basis.x.y + v * basis.y.y),
  };
}

/** Back-project a pixel to camera-space metres at a known depth. */
function backProject(px, depthM, intr) {
  return v3(
    ((px.x - intr.cx) / intr.fx) * depthM,
    ((px.y - intr.cy) / intr.fy) * depthM,
    depthM,
  );
}
