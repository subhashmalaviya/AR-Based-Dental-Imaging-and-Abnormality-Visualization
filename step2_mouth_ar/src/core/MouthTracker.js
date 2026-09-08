/**
 * MouthTracker.js — turns a face-landmark set into mouth measurements.
 *
 * Consumes the raw landmark array from FaceTracker and produces the mouth's
 * geometry: corners, lip centres, an opening measure, a bounding box, and the
 * orthonormal 3D frame that MouthARAnchor turns into a stable AR pose.
 *
 * All outputs are in **pixel** coordinates for x/y (so they can be drawn
 * directly) and pixel-equivalent units for z, so the frame is not skewed by a
 * non-square video aspect ratio. MediaPipe returns x/y normalised to [0,1] by
 * width/height independently, and z on roughly the same scale as x; converting
 * both by the *same* factor (the video width) would distort y, so x and z use
 * width and y uses height.
 */
import {
  LIPS_ALL, LIPS_INNER_RING, LIPS_OUTER_RING, LOWER_LIP_INNER, LOWER_LIP_OUTER,
  MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, UPPER_LIP_INNER, UPPER_LIP_OUTER,
} from '../landmarks/FaceLandmarkIndices.js';
import {
  centroid, cross, length, normalize, orthonormalBasis, sub, v3,
} from './math/vec3.js';

/** Polygon area via the shoelace formula (x/y only). */
function polygonArea(points) {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const q = points[(i + 1) % points.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
}

export class MouthTracker {
  /**
   * @param {Array<{x:number,y:number,z:number}>} landmarks normalised MediaPipe landmarks
   * @param {number} width  video frame width in px
   * @param {number} height video frame height in px
   */
  track(landmarks, width, height) {
    if (!landmarks || landmarks.length < 468) return null;

    // Scale into pixel space. z shares x's scale in MediaPipe's convention.
    const px = (i) => {
      const p = landmarks[i];
      return v3(p.x * width, p.y * height, (p.z ?? 0) * width);
    };

    const cornerLeft = px(MOUTH_CORNER_LEFT);
    const cornerRight = px(MOUTH_CORNER_RIGHT);
    const upperOuter = px(UPPER_LIP_OUTER);
    const upperInner = px(UPPER_LIP_INNER);
    const lowerInner = px(LOWER_LIP_INNER);
    const lowerOuter = px(LOWER_LIP_OUTER);

    const outerRing = LIPS_OUTER_RING.map(px);
    const innerRing = LIPS_INNER_RING.map(px);
    const allPoints = LIPS_ALL.map(px);

    // --- the mouth's own coordinate frame -------------------------------
    // +X runs corner-to-corner (the most stable direction on the mouth),
    // +Y runs from upper to lower lip, +Z is the outward facial normal.
    const widthAxis = sub(cornerRight, cornerLeft);
    const mouthWidth = length(widthAxis);
    if (!(mouthWidth > 1e-6)) return null;

    const upAxisRough = sub(lowerOuter, upperOuter);
    const basis = orthonormalBasis(widthAxis, upAxisRough);

    // Origin at the centroid of the outer lip ring rather than the midpoint of
    // the two corners: averaging 20 points instead of 2 measurably reduces the
    // jitter that a single noisy landmark would inject into the anchor.
    const origin = centroid(outerRing);

    // --- mouth opening ---------------------------------------------------
    // Two complementary measures, both normalised by mouth width so they are
    // invariant to how close the face is to the camera:
    //   gap   — inner-lip separation, the direct "how far open" signal
    //   area  — inner-ring area, more robust when lips are partly occluded
    const gap = length(sub(lowerInner, upperInner));
    const openingRatio = gap / mouthWidth;
    const areaRatio = polygonArea(innerRing) / (mouthWidth * mouthWidth);

    // --- bounding box (axis-aligned, in pixels) ---------------------------
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of allPoints) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    return {
      origin,
      basis,
      mouthWidth,
      corners: { left: cornerLeft, right: cornerRight },
      lips: {
        upperOuter, upperInner, lowerInner, lowerOuter,
        outerRing, innerRing,
      },
      opening: { gap, ratio: openingRatio, areaRatio, isOpen: openingRatio > 0.12 },
      boundingBox: {
        x: minX, y: minY, width: maxX - minX, height: maxY - minY,
        cx: (minX + maxX) / 2, cy: (minY + maxY) / 2,
      },
    };
  }
}
