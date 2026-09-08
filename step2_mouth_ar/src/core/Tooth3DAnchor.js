/**
 * Tooth3DAnchor.js — one independent spatial anchor per tracked tooth.
 *
 * This is the object a later stage attaches real geometry to: a tooth mesh,
 * root canal geometry, a nerve canal from CBCT, a lesion volume. Each anchor
 * owns a full transform, so content authored once in tooth-local coordinates
 * follows that specific tooth — not the mouth, and not a screen rectangle.
 *
 *     Real tooth  ->  Tooth3DAnchor  ->  attached 3D model  ->  AR render
 *
 * Deliberately NOT a 2D box with a z value bolted on. The anchor holds a
 * genuine 4x4 tooth-to-camera transform composed from the head's tracked
 * rotation and the tooth's own measured position; `getMatrix()` is directly
 * usable as a three.js `Object3D.matrix`.
 *
 * Read ToothPoseEstimator's header before trusting any number here: the depth
 * axis is an anatomical estimate, not a measurement, and metric scale rests on
 * an assumed mouth width.
 */
import {
  basisOf, compose, eulerOf, invertRigid, multiply, positionOf, projectPoint,
  transformPoint,
} from './math/mat4.js';

export class Tooth3DAnchor {
  constructor(pose) {
    this.id = pose.id;
    this.arch = pose.arch;
    this.update(pose);
    /** Slot for a future 3D model (mesh, nerve, lesion volume). */
    this.attachment = null;
  }

  update(pose) {
    this.pose = pose;
    this.toothToCamera = pose.toothToCamera;
    this.confidence = pose.confidence;
    this.trackingState = pose.trackingState;
    this.provenance = pose.provenance;
  }

  // ---- the transform, in the forms a renderer or a 3D engine wants ----
  getMatrix() { return this.toothToCamera; }
  get position() { return positionOf(this.toothToCamera); }
  get rotationEuler() { return eulerOf(this.toothToCamera); }
  get basis() { return basisOf(this.toothToCamera); }
  get scale() { return this.pose.scale; }

  /**
   * Rotation of the tooth WITHIN the face, in degrees.
   *
   * Prefer this for anything a human reads. The camera-frame Euler angles are
   * correct but read as +/-180 at rest, because the face frame is Y-up while
   * the camera frame is Y-down — a rest pose that looks alarming and means
   * nothing. In the face frame a tooth sits near zero and the number you see
   * is the arch splay: how far this tooth is turned away from the midline.
   */
  get rotationInFace() {
    return eulerOf(compose({ x: 0, y: 0, z: 0 }, this.pose.basisFace, 1));
  }

  /** Transform (T, R, S) as the spec asks for it. */
  getTransform() {
    return {
      position: this.position,
      rotation: this.rotationEuler,        // camera frame — what a renderer wants
      rotationInFace: this.rotationInFace, // face frame — what a person reads
      scale: this.scale,
    };
  }

  // ---- coordinate-space conversions -----------------------------------
  /** tooth-local metres -> camera metres */
  toCamera(pLocal) { return transformPoint(this.toothToCamera, pLocal); }

  /** camera metres -> tooth-local metres */
  fromCamera(pCam) { return transformPoint(invertRigid(this.toothToCamera), pCam); }

  /** tooth-local metres -> pixels (null if behind the camera) */
  toScreen(pLocal, intr) { return projectPoint(this.toCamera(pLocal), intr); }

  /**
   * tooth-local -> face frame (metric, rigid to the skull).
   *
   * This is the frame in which a tooth should be STATIONARY while the head
   * moves, so it is the one to inspect when asking "is this anchor really
   * attached in 3D?" — see tests/anchor3d.test.mjs.
   */
  toFace(pLocal) {
    return transformPoint(multiply(this.pose.cameraToFace, this.toothToCamera), pLocal);
  }

  /** This tooth's origin in the face frame. */
  get positionFace() { return this.pose.positionFace; }

  /**
   * Attach 3D content to this specific tooth.
   * @param {object} model anything with its own local geometry; Step 4 would
   *   pass a mesh, a segmentation volume, or an abnormality marker.
   */
  attach(model) { this.attachment = model; return this; }

  /** A compact record, suitable for logging or handing to a later stage. */
  toJSON() {
    const t = this.getTransform();
    return {
      tooth_id: this.id,
      arch: this.arch,
      position_m: { x: +t.position.x.toFixed(5), y: +t.position.y.toFixed(5), z: +t.position.z.toFixed(5) },
      rotation_deg: { rx: +t.rotation.rx.toFixed(2), ry: +t.rotation.ry.toFixed(2), rz: +t.rotation.rz.toFixed(2) },
      rotation_in_face_deg: {
        rx: +t.rotationInFace.rx.toFixed(2),
        ry: +t.rotationInFace.ry.toFixed(2),
        rz: +t.rotationInFace.rz.toFixed(2),
      },
      scale_m: { x: +t.scale.x.toFixed(5), y: +t.scale.y.toFixed(5), z: +t.scale.z.toFixed(5) },
      depth_source: this.pose.depthSource,
      confidence: +this.confidence.toFixed(3),
      tracking: this.trackingState,
      provenance: this.provenance,
    };
  }
}

/**
 * Keeps one anchor per tracked tooth ID, so anchors persist with their tooth
 * across frames rather than being rebuilt (and renumbered) every frame.
 */
export class Tooth3DAnchorSet {
  constructor() { this.anchors = new Map(); }

  /** @param {Array} poses output of ToothPoseEstimator for this frame */
  update(poses) {
    const alive = new Set();
    for (const p of poses) {
      alive.add(p.id);
      const existing = this.anchors.get(p.id);
      if (existing) existing.update(p);
      else this.anchors.set(p.id, new Tooth3DAnchor(p));
    }
    for (const id of [...this.anchors.keys()]) {
      if (!alive.has(id)) this.anchors.delete(id);   // tooth gone: drop its anchor
    }
    return this.list();
  }

  list() { return [...this.anchors.values()]; }
  get(id) { return this.anchors.get(id) ?? null; }
  get size() { return this.anchors.size; }
  clear() { this.anchors.clear(); }
}
