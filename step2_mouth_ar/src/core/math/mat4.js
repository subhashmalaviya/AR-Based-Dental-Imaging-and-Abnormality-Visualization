/**
 * mat4.js — column-major 4x4 matrix maths for the 3D anchor framework.
 *
 * Column-major, matching WebGL/three.js and MediaPipe's facial transformation
 * matrix, so a matrix produced here can be handed straight to a GPU renderer or
 * an `Object3D.matrix` in a later step without repacking.
 *
 * Layout: m[col * 4 + row]. Translation lives in m[12], m[13], m[14].
 */

export const identity = () => new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

/** a * b (apply b first, then a). */
export function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4]
        + a[4 + r] * b[c * 4 + 1]
        + a[8 + r] * b[c * 4 + 2]
        + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

/** Transform a point (w = 1); returns {x,y,z}. */
export function transformPoint(m, p) {
  return {
    x: m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
    y: m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
    z: m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
  };
}

/** Transform a direction (w = 0) — ignores translation. */
export function transformDirection(m, v) {
  return {
    x: m[0] * v.x + m[4] * v.y + m[8] * v.z,
    y: m[1] * v.x + m[5] * v.y + m[9] * v.z,
    z: m[2] * v.x + m[6] * v.y + m[10] * v.z,
  };
}

/** Compose translation * rotation(basis columns) * scale. */
export function compose(position, basis, scale) {
  const s = typeof scale === 'number' ? { x: scale, y: scale, z: scale } : scale;
  return new Float32Array([
    basis.x.x * s.x, basis.x.y * s.x, basis.x.z * s.x, 0,
    basis.y.x * s.y, basis.y.y * s.y, basis.y.z * s.y, 0,
    basis.z.x * s.z, basis.z.y * s.z, basis.z.z * s.z, 0,
    position.x, position.y, position.z, 1,
  ]);
}

/**
 * Inverse of a rigid-with-uniform-scale transform.
 * Falls back to the general inverse if the basis is not orthogonal.
 */
export function invertRigid(m) {
  // column lengths = per-axis scale
  const sx = Math.hypot(m[0], m[1], m[2]);
  const sy = Math.hypot(m[4], m[5], m[6]);
  const sz = Math.hypot(m[8], m[9], m[10]);
  if (sx < 1e-9 || sy < 1e-9 || sz < 1e-9) return identity();

  // R^-1 = R^T (after removing scale), t^-1 = -R^-1 * t
  const r = [
    m[0] / sx, m[1] / sx, m[2] / sx,
    m[4] / sy, m[5] / sy, m[6] / sy,
    m[8] / sz, m[9] / sz, m[10] / sz,
  ];
  const tx = m[12], ty = m[13], tz = m[14];
  // transpose of R, then divide by scale again to undo it
  const i0 = r[0] / sx, i1 = r[3] / sy, i2 = r[6] / sz;
  const i4 = r[1] / sx, i5 = r[4] / sy, i6 = r[7] / sz;
  const i8 = r[2] / sx, i9 = r[5] / sy, i10 = r[8] / sz;

  return new Float32Array([
    i0, i1, i2, 0,
    i4, i5, i6, 0,
    i8, i9, i10, 0,
    -(i0 * tx + i4 * ty + i8 * tz),
    -(i1 * tx + i5 * ty + i9 * tz),
    -(i2 * tx + i6 * ty + i10 * tz),
    1,
  ]);
}

/** Extract the rotation basis (unit column vectors) from a TRS matrix. */
export function basisOf(m) {
  const sx = Math.hypot(m[0], m[1], m[2]) || 1;
  const sy = Math.hypot(m[4], m[5], m[6]) || 1;
  const sz = Math.hypot(m[8], m[9], m[10]) || 1;
  return {
    x: { x: m[0] / sx, y: m[1] / sx, z: m[2] / sx },
    y: { x: m[4] / sy, y: m[5] / sy, z: m[6] / sy },
    z: { x: m[8] / sz, y: m[9] / sz, z: m[10] / sz },
  };
}

export const positionOf = (m) => ({ x: m[12], y: m[13], z: m[14] });

export const scaleOf = (m) => ({
  x: Math.hypot(m[0], m[1], m[2]),
  y: Math.hypot(m[4], m[5], m[6]),
  z: Math.hypot(m[8], m[9], m[10]),
});

/** Intrinsic XYZ Euler angles (degrees) from a rotation basis. */
export function eulerOf(m) {
  const b = basisOf(m);
  const deg = (r) => (r * 180) / Math.PI;
  const pitch = Math.asin(Math.max(-1, Math.min(1, -b.z.y)));
  return {
    rx: deg(pitch),
    ry: deg(Math.atan2(b.z.x, b.z.z)),
    rz: deg(Math.atan2(b.x.y, b.y.y)),
  };
}

/**
 * Pinhole projection of a camera-space point to pixels.
 * Returns null when the point is behind the camera.
 */
export function projectPoint(p, intr) {
  if (!(p.z > 1e-6)) return null;
  return {
    x: intr.fx * (p.x / p.z) + intr.cx,
    y: intr.fy * (p.y / p.z) + intr.cy,
    depth: p.z,
  };
}
