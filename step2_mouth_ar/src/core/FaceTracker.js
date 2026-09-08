/**
 * FaceTracker.js — continuous face detection and landmarking.
 *
 * ===========================================================================
 * WHERE THE FACE TRACKING ACTUALLY HAPPENS
 * ===========================================================================
 * Here, via **MediaPipe Face Landmarker** (Tasks Vision) running in VIDEO mode
 * on the device GPU. Each call to `detect()` returns 478 3D face landmarks
 * plus a 4x4 head transformation matrix.
 *
 * Two things are worth knowing:
 *
 * 1. VIDEO running mode is not the same as calling an image detector in a
 *    loop. MediaPipe keeps state between timestamps and runs its own
 *    detect-then-track pipeline internally: a face detector locates the face
 *    once, and a lighter landmark model follows it on subsequent frames,
 *    re-detecting only when tracking confidence drops. That is what makes it
 *    fast enough for phones, and it is why timestamps must be monotonically
 *    increasing — going backwards makes the task throw.
 *
 * 2. The model bundle and WASM runtime are served from this app's own origin
 *    (public/models, public/wasm), not a CDN, so the app works offline and
 *    inside a Capacitor WebView.
 *
 * Everything downstream (MouthTracker, MouthARAnchor) consumes only the plain
 * landmark array, so swapping this for ARKit/ARCore face tracking or another
 * landmarker means reimplementing this file alone.
 */
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export class FaceTracker {
  constructor({
    wasmPath = 'wasm',
    modelPath = 'models/face_landmarker.task',
    numFaces = 1,
    minFaceDetectionConfidence = 0.5,
    minFacePresenceConfidence = 0.5,
    minTrackingConfidence = 0.5,
  } = {}) {
    Object.assign(this, {
      wasmPath, modelPath, numFaces,
      minFaceDetectionConfidence, minFacePresenceConfidence, minTrackingConfidence,
    });
    this.landmarker = null;
    this.delegate = null;
    this.ready = false;
    this._lastTimestamp = -1;
    this.lastResult = null;
  }

  async init(onProgress = () => {}) {
    onProgress('Loading MediaPipe runtime…');
    const fileset = await FilesetResolver.forVisionTasks(this.wasmPath);

    const options = (delegate) => ({
      baseOptions: { modelAssetPath: this.modelPath, delegate },
      runningMode: 'VIDEO',
      numFaces: this.numFaces,
      minFaceDetectionConfidence: this.minFaceDetectionConfidence,
      minFacePresenceConfidence: this.minFacePresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
      outputFaceBlendshapes: false,          // not needed in Step 2; costs time
      outputFacialTransformationMatrixes: true,
    });

    // Prefer GPU; fall back to CPU. Some Android WebViews advertise WebGL but
    // fail to create the GPU delegate, and a hard failure there would look
    // like "the app is broken" rather than "this device needs CPU".
    try {
      onProgress('Creating face landmarker (GPU)…');
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options('GPU'));
      this.delegate = 'GPU';
    } catch (gpuErr) {
      console.warn('[FaceTracker] GPU delegate unavailable, falling back to CPU:', gpuErr);
      onProgress('Creating face landmarker (CPU fallback)…');
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options('CPU'));
      this.delegate = 'CPU';
    }

    this.ready = true;
    return this.delegate;
  }

  /**
   * @param {HTMLVideoElement} video
   * @param {number} timestampMs must strictly increase between calls
   * @returns {{landmarks:Array|null, matrix:Float32Array|null, faceCount:number}}
   */
  detect(video, timestampMs) {
    if (!this.ready || !video || video.readyState < 2) {
      return { landmarks: null, matrix: null, faceCount: 0 };
    }
    // MediaPipe throws if a timestamp repeats or goes backwards; nudge instead.
    let ts = Math.round(timestampMs);
    if (ts <= this._lastTimestamp) ts = this._lastTimestamp + 1;
    this._lastTimestamp = ts;

    let result;
    try {
      result = this.landmarker.detectForVideo(video, ts);
    } catch (err) {
      console.warn('[FaceTracker] detectForVideo failed:', err);
      return { landmarks: null, matrix: null, faceCount: 0 };
    }
    this.lastResult = result;

    const faces = result?.faceLandmarks ?? [];
    if (!faces.length) return { landmarks: null, matrix: null, faceCount: 0 };

    const mats = result.facialTransformationMatrixes ?? [];
    return {
      landmarks: faces[0],
      matrix: mats.length ? mats[0].data : null,
      faceCount: faces.length,
    };
  }

  /**
   * Head pose from MediaPipe's 4x4 face transformation matrix, in degrees.
   *
   * This is a whole-head estimate fitted against MediaPipe's canonical face
   * model. It is reported alongside the mouth anchor's own orientation because
   * the two answer different questions: this one is "where is the head
   * pointing", the anchor's is "how is the mouth surface oriented". Step 3's
   * registration of a metric dental model will want this matrix.
   */
  static headPoseFromMatrix(m) {
    if (!m || m.length < 16) return null;
    // column-major: m[col*4 + row]
    const r = (row, col) => m[col * 4 + row];
    const sy = Math.hypot(r(0, 0), r(1, 0));
    const singular = sy < 1e-6;
    const deg = (v) => (v * 180) / Math.PI;
    if (singular) {
      return { yaw: 0, pitch: deg(Math.atan2(-r(1, 2), r(1, 1))), roll: 0 };
    }
    return {
      pitch: deg(Math.atan2(r(2, 1), r(2, 2))),
      yaw: deg(Math.atan2(-r(2, 0), sy)),
      roll: deg(Math.atan2(r(1, 0), r(0, 0))),
      tx: r(0, 3), ty: r(1, 3), tz: r(2, 3),
    };
  }

  close() {
    try { this.landmarker?.close?.(); } catch { /* already closed */ }
    this.landmarker = null;
    this.ready = false;
  }
}
