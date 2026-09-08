/**
 * vec3.js — minimal 3-vector helpers.
 *
 * Vectors are plain {x, y, z} objects so landmark data from MediaPipe can be
 * used directly without copying into typed arrays.
 */

export const v3 = (x = 0, y = 0, z = 0) => ({ x, y, z });

export const add = (a, b) => v3(a.x + b.x, a.y + b.y, a.z + b.z);
export const sub = (a, b) => v3(a.x - b.x, a.y - b.y, a.z - b.z);
export const scale = (a, s) => v3(a.x * s, a.y * s, a.z * s);
export const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;

export const cross = (a, b) => v3(
  a.y * b.z - a.z * b.y,
  a.z * b.x - a.x * b.z,
  a.x * b.y - a.y * b.x,
);

export const length = (a) => Math.hypot(a.x, a.y, a.z);

export function normalize(a) {
  const n = length(a);
  return n > 1e-12 ? scale(a, 1 / n) : v3(0, 0, 0);
}

export function centroid(points) {
  if (!points.length) return v3();
  let x = 0, y = 0, z = 0;
  for (const p of points) { x += p.x; y += p.y; z += p.z ?? 0; }
  const n = points.length;
  return v3(x / n, y / n, z / n);
}

/**
 * Gram-Schmidt: orthonormal right-handed basis from a primary direction and a
 * rough secondary direction. `up` only has to be approximately correct; its
 * component along `right` is removed.
 */
export function orthonormalBasis(right, up) {
  const x = normalize(right);
  const yRaw = sub(up, scale(x, dot(up, x)));
  const y = normalize(yRaw);
  const z = normalize(cross(x, y));
  return { x, y, z };
}

/** Rotation matrix (basis as columns) -> quaternion {w,x,y,z}. */
export function basisToQuaternion(b) {
  // Column-major: m[col][row]
  const m00 = b.x.x, m01 = b.y.x, m02 = b.z.x;
  const m10 = b.x.y, m11 = b.y.y, m12 = b.z.y;
  const m20 = b.x.z, m21 = b.y.z, m22 = b.z.z;

  const tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1.0) * 2;
    w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  return { w: w / n, x: x / n, y: y / n, z: z / n };
}

/** Quaternion -> orthonormal basis (inverse of basisToQuaternion). */
export function quaternionToBasis(q) {
  const { w, x, y, z } = q;
  return {
    x: v3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
    y: v3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
    z: v3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
  };
}

/**
 * Intrinsic yaw/pitch/roll (degrees) of a basis expressed in camera axes
 * (+x right, +y down, +z into the screen).
 *
 * Convention: R = Ry(yaw) * Rx(pitch) * Rz(roll), with the basis vectors held
 * as the COLUMNS of R. Expanding that product gives
 *
 *   b.z = ( sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch) )
 *   b.x.y = cos(pitch)sin(roll),   b.y.y = cos(pitch)cos(roll)
 *
 * hence the extraction below. Reading the terms off the transpose instead
 * (b.x.z / b.y.z / b.y.x) yields every angle negated -- which is exactly what
 * the round-trip test caught, reporting yaw 20 deg as -20 deg.
 */
export function basisToEuler(b) {
  const pitch = Math.asin(Math.max(-1, Math.min(1, -b.z.y)));
  const yaw = Math.atan2(b.z.x, b.z.z);
  const roll = Math.atan2(b.x.y, b.y.y);
  const deg = (r) => (r * 180) / Math.PI;
  return { yaw: deg(yaw), pitch: deg(pitch), roll: deg(roll) };
}
