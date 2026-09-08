/**
 * MouthARAnchor.js — a stable AR coordinate system attached to the mouth.
 *
 * This is the Step-2 analogue of Step 1's cup pose, and it follows the same
 * principle that made the cup overlay work: **content is defined once in the
 * object's own coordinates, and only the pose changes per frame.** Nothing the
 * overlay draws is positioned in screen space.
 *
 * ---------------------------------------------------------------------------
 * MOUTH-LOCAL COORDINATE SYSTEM  (the contract Step 3 registers against)
 * ---------------------------------------------------------------------------
 *   origin  centroid of the outer lip ring
 *   +X      towards the subject's right mouth corner   (mouth width direction)
 *   +Y      downwards, towards the chin
 *   +Z      outwards from the face, towards the camera
 *   units   MOUTH WIDTHS — 1.0 == corner-to-corner distance.
 *
 * So (-0.5, 0, 0) and (+0.5, 0, 0) are the two mouth corners, and the teeth
 * live at slightly negative Z (behind the lip plane). Because the unit is the
 * mouth width, any content authored in this frame automatically keeps the
 * right size as the subject moves nearer or further from the camera.
 *
 * Projection is **weak perspective**: a point is placed in pixel space by
 * origin + scale*(lx*X + ly*Y + lz*Z) and its z dropped. That is deliberate.
 * MediaPipe's landmark z is a relative depth, not metric, so a full projective
 * model would be false precision; over a region as small as a mouth the
 * orthographic approximation is accurate and far steadier. `getMatrix()` still
 * exposes a full 4x4 so Step 3 can swap in a metric camera without changing
 * any caller.
 */
import { basisToEuler, basisToQuaternion, quaternionToBasis } from './math/vec3.js';
import { PoseSmoother } from './filters/PoseSmoother.js';

export class MouthARAnchor {
  constructor({ smoothing = 'balanced', gating = true } = {}) {
    this.smoother = new PoseSmoother(smoothing);
    this.pose = null;
    this.rawPose = null;
    this.lostFrames = 0;
    this.maxCoastFrames = 5;

    // --- outlier gating -------------------------------------------------
    // The 1-euro filter deliberately opens its cutoff as speed rises, so it
    // tracks fast head motion without lag -- but that also means it will pass
    // a single wild measurement straight through. Rejecting implausible jumps
    // is therefore a separate job, done here, mirroring the plausibility gate
    // that proved necessary in the Step-1 cup tracker.
    this.gating = gating;
    this.maxJumpInWidths = 0.8;   // origin may move < 0.8 mouth-widths / frame
    this.scaleRange = [0.6, 1.7]; // and change size by < ~1.7x / frame
    this.rejectRun = 0;
    this.maxRejectRun = 6;        // then believe the new measurement instead
    this.rejectedFrames = 0;
  }

  setSmoothing(preset) { this.smoother.setPreset(preset); }

  reset() {
    this.smoother.reset();
    this.pose = null;
    this.rawPose = null;
    this.lostFrames = 0;
    this.rejectRun = 0;
  }

  /**
   * Is this measurement a plausible continuation of the current pose?
   * Returns true when there is nothing to compare against yet.
   */
  _isPlausible(mouth) {
    if (!this.gating || !this.pose) return true;
    const ref = this.pose.scale || 1;
    const dx = mouth.origin.x - this.pose.origin.x;
    const dy = mouth.origin.y - this.pose.origin.y;
    if (Math.hypot(dx, dy) > this.maxJumpInWidths * ref) return false;
    const r = mouth.mouthWidth / ref;
    return r >= this.scaleRange[0] && r <= this.scaleRange[1];
  }

  /**
   * @param {object|null} mouth output of MouthTracker.track (null when no face)
   * @param {number} tSec monotonic timestamp in seconds
   * @returns {object|null} the smoothed anchor pose
   */
  update(mouth, tSec) {
    if (!mouth) {
      // Brief detection dropouts are common (blink, motion blur). Hold the last
      // pose for a few frames instead of making the overlay flicker off/on,
      // then declare the anchor lost.
      this.lostFrames += 1;
      if (this.lostFrames > this.maxCoastFrames) {
        this.reset();
        return null;
      }
      return this.pose;
    }

    this.lostFrames = 0;

    if (!this._isPlausible(mouth)) {
      this.rejectRun += 1;
      this.rejectedFrames += 1;
      // Hold the last good pose. But if the "outlier" persists, it is not an
      // outlier -- the subject really did move, or the detector re-acquired a
      // different face -- so stop vetoing it and re-anchor. Recovery must not
      // be gated on the state that went stale.
      if (this.rejectRun <= this.maxRejectRun) return this.pose;
      this.smoother.reset();
      this.rejectRun = 0;
    } else {
      this.rejectRun = 0;
    }

    const quaternion = basisToQuaternion(mouth.basis);
    this.rawPose = {
      origin: mouth.origin,
      quaternion,
      basis: mouth.basis,
      scale: mouth.mouthWidth,
      mouthOpen: mouth.opening.ratio,
    };

    const s = this.smoother.smooth(this.rawPose, tSec);
    this.pose = {
      ...s,
      euler: basisToEuler(s.basis),
      valid: true,
    };
    return this.pose;
  }

  isValid() { return this.pose !== null; }
  getPose() { return this.pose; }

  /**
   * Map a point from mouth-local coordinates (mouth widths) to pixels.
   * @param {{x:number,y:number,z:number}} p
   * @param {object} [pose] defaults to the current smoothed pose
   */
  localToScreen(p, pose = this.pose) {
    if (!pose) return null;
    const { origin, basis, scale } = pose;
    const lz = p.z ?? 0;
    return {
      x: origin.x + scale * (p.x * basis.x.x + p.y * basis.y.x + lz * basis.z.x),
      y: origin.y + scale * (p.x * basis.x.y + p.y * basis.y.y + lz * basis.z.y),
      // depth kept so callers can z-sort; not used by the weak-perspective drop
      depth: origin.z + scale * (p.x * basis.x.z + p.y * basis.y.z + lz * basis.z.z),
    };
  }

  /** Inverse of localToScreen for z=0, useful for hit-testing / picking. */
  screenToLocal(pt, pose = this.pose) {
    if (!pose) return null;
    const { origin, basis, scale } = pose;
    const dx = (pt.x - origin.x) / scale;
    const dy = (pt.y - origin.y) / scale;
    // Project the screen offset onto the (already orthonormal) in-plane axes.
    return {
      x: dx * basis.x.x + dy * basis.x.y,
      y: dx * basis.y.x + dy * basis.y.y,
      z: 0,
    };
  }

  /**
   * Column-major 4x4 mouth-local -> pixel-space matrix.
   *
   * Provided for Step 3: a dental mesh or CT-derived model expressed in mouth
   * coordinates can be pushed straight into a WebGL/three.js model matrix
   * without touching this module.
   */
  getMatrix(pose = this.pose) {
    if (!pose) return null;
    const { origin, basis, scale } = pose;
    const s = scale;
    return new Float32Array([
      basis.x.x * s, basis.x.y * s, basis.x.z * s, 0,
      basis.y.x * s, basis.y.y * s, basis.y.z * s, 0,
      basis.z.x * s, basis.z.y * s, basis.z.z * s, 0,
      origin.x, origin.y, origin.z, 1,
    ]);
  }

  /** Convenience for the HUD. */
  getReadout() {
    if (!this.pose) return null;
    const { origin, scale, euler, mouthOpen } = this.pose;
    return {
      x: origin.x, y: origin.y, depthProxy: scale,
      yaw: euler.yaw, pitch: euler.pitch, roll: euler.roll,
      mouthOpen,
    };
  }
}

export { quaternionToBasis };
