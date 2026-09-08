/**
 * PoseSmoother.js — temporal smoothing of a full 6-DoF-style anchor pose.
 *
 * Smooths position, scale and orientation with 1-euro filters. Orientation is
 * carried as a **quaternion with hemisphere alignment**, not as Euler angles:
 * filtering yaw/pitch/roll independently corrupts every wrap through +/-180
 * degrees and is not a linear space, so the overlay would flip or swing wildly
 * near those boundaries. (Same lesson as the Step-1 cup tracker, which smooths
 * rotations as quaternions for exactly this reason.)
 */
import { OneEuroFilter, OneEuroVectorFilter } from './OneEuroFilter.js';
import { quaternionToBasis } from '../math/vec3.js';

export const SMOOTHING_PRESETS = {
  responsive: { minCutoff: 2.2, beta: 0.05, dCutoff: 1.0 },
  balanced: { minCutoff: 1.0, beta: 0.02, dCutoff: 1.0 },
  smooth: { minCutoff: 0.45, beta: 0.008, dCutoff: 1.0 },
};

export class PoseSmoother {
  constructor(preset = 'balanced') {
    const p = SMOOTHING_PRESETS[preset] ?? SMOOTHING_PRESETS.balanced;
    this.origin = new OneEuroVectorFilter(3, p);
    // Scale drives the overlay's size; a touch steadier than position because
    // size flicker is more visually obvious than a small positional wobble.
    this.scale = new OneEuroFilter({ ...p, minCutoff: p.minCutoff * 0.8 });
    this.quat = new OneEuroVectorFilter(4, p);
    this.mouthOpen = new OneEuroFilter({ ...p, minCutoff: p.minCutoff * 1.5 });
    this.prevQuat = null;
  }

  setPreset(name) {
    const p = SMOOTHING_PRESETS[name] ?? SMOOTHING_PRESETS.balanced;
    this.origin.setParams(p);
    this.scale.setParams({ ...p, minCutoff: p.minCutoff * 0.8 });
    this.quat.setParams(p);
    this.mouthOpen.setParams({ ...p, minCutoff: p.minCutoff * 1.5 });
  }

  reset() {
    this.origin.reset();
    this.scale.reset();
    this.quat.reset();
    this.mouthOpen.reset();
    this.prevQuat = null;
  }

  /**
   * @param {{origin:{x,y,z}, quaternion:{w,x,y,z}, scale:number, mouthOpen:number}} pose
   * @param {number} tSec
   */
  smooth(pose, tSec) {
    const [ox, oy, oz] = this.origin.filter(
      [pose.origin.x, pose.origin.y, pose.origin.z], tSec);
    const scale = this.scale.filter(pose.scale, tSec);
    const mouthOpen = this.mouthOpen.filter(pose.mouthOpen ?? 0, tSec);

    // q and -q are the same rotation; flip into the previous hemisphere before
    // filtering, otherwise a sign flip is filtered as a huge jump.
    let q = pose.quaternion;
    if (this.prevQuat) {
      const d = q.w * this.prevQuat.w + q.x * this.prevQuat.x
        + q.y * this.prevQuat.y + q.z * this.prevQuat.z;
      if (d < 0) q = { w: -q.w, x: -q.x, y: -q.y, z: -q.z };
    }
    const [qw, qx, qy, qz] = this.quat.filter([q.w, q.x, q.y, q.z], tSec);
    const n = Math.hypot(qw, qx, qy, qz) || 1;
    const quaternion = { w: qw / n, x: qx / n, y: qy / n, z: qz / n };
    this.prevQuat = quaternion;

    return {
      origin: { x: ox, y: oy, z: oz },
      quaternion,
      basis: quaternionToBasis(quaternion),
      scale,
      mouthOpen,
    };
  }
}
