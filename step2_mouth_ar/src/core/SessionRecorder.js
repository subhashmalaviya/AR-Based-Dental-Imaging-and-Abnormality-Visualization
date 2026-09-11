/**
 * SessionRecorder.js — local recording of the live session for later analysis.
 *
 * Everything happens on the device. Frames are never uploaded: MediaRecorder
 * encodes in the browser and the result is handed back as a Blob that the
 * user downloads / shares. There is no server involved at any point.
 *
 * Two modes
 * ---------
 *  raw        MediaRecorder on the camera's own MediaStream. Exactly the
 *             sensor frames, unmirrored, unaffected by processing load. This
 *             is the right input for re-running detection offline and for
 *             building a dataset.
 *  annotated  The preview as you see it — video plus the analysis overlay
 *             (tooth masks, IDs, confidence, landmarks, AR, HUD) — composited
 *             into an offscreen canvas and recorded from canvas.captureStream().
 *             Composited once per *processed* frame, so every recorded frame
 *             shows the detections that were computed for it.
 *
 * The front-camera preview is mirrored on screen. The annotated recording is
 * mirrored the same way, because the overlay's text is drawn counter-flipped
 * for the mirrored preview and would otherwise read backwards. The raw
 * recording is NOT mirrored. Metadata coordinates are always in raw
 * (unmirrored) frame pixels; the JSON header records both facts.
 *
 * Format: WebM (VP9 > VP8) where supported — Chrome, Firefox, Android. Safari
 * (iOS/iPadOS/macOS) only records MP4/H.264, which is chosen automatically.
 */

const CANDIDATES = [
  { mimeType: 'video/webm;codecs=vp9', ext: 'webm' },
  { mimeType: 'video/webm;codecs=vp8', ext: 'webm' },
  { mimeType: 'video/webm', ext: 'webm' },
  { mimeType: 'video/mp4;codecs=avc1.42E01E', ext: 'mp4' },
  { mimeType: 'video/mp4', ext: 'mp4' },
];

export function pickRecordingFormat() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const c of CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(c.mimeType)) return c;
    } catch { /* some browsers throw on unknown codec strings */ }
  }
  return null;
}

/** dental_tracking_YYYYMMDD_HHMMSS, in local time. */
export function recordingBaseName(date = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `dental_tracking_${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}`
    + `_${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
}

export class SessionRecorder {
  /**
   * @param {object} deps
   * @param {() => MediaStream|null} deps.getCameraStream
   * @param {HTMLVideoElement} deps.video
   * @param {HTMLCanvasElement} deps.overlay
   * @param {() => boolean} deps.isMirrored
   */
  constructor({ getCameraStream, video, overlay, isMirrored }) {
    this.getCameraStream = getCameraStream;
    this.video = video;
    this.overlay = overlay;
    this.isMirrored = isMirrored;

    this.format = pickRecordingFormat();
    this.recorder = null;
    this.chunks = [];
    this.mode = 'raw';
    this.state = 'idle';            // idle | recording | stopping
    this.startedAt = 0;             // performance.now() at the 'start' event
    this.startedDate = null;
    this.framesComposited = 0;

    this._canvas = null;
    this._ctx = null;
    this._track = null;
  }

  get isSupported() { return !!this.format; }
  get isRecording() { return this.state === 'recording'; }
  get elapsedMs() { return this.isRecording ? performance.now() - this.startedAt : 0; }

  /**
   * @param {object} opts
   * @param {boolean} opts.annotated  record the overlay too
   * @param {number} [opts.fps=30]
   */
  async start({ annotated = false, fps = 30 } = {}) {
    if (!this.isSupported) throw new Error('Recording is not supported in this browser (no MediaRecorder).');
    if (this.isRecording) return;

    let stream;
    if (annotated) {
      const w = this.video.videoWidth, h = this.video.videoHeight;
      if (!w || !h) throw new Error('Camera is not running.');
      this._canvas = document.createElement('canvas');
      this._canvas.width = w;
      this._canvas.height = h;
      this._ctx = this._canvas.getContext('2d');
      // captureStream(0) + requestFrame() gives one recorded frame per
      // composited frame, keeping overlay and video in lockstep. Browsers
      // without requestFrame fall back to a fixed-rate capture.
      stream = this._canvas.captureStream(0);
      this._track = stream.getVideoTracks()[0];
      if (typeof this._track?.requestFrame !== 'function') {
        stream = this._canvas.captureStream(fps);
        this._track = null;
      }
      this.compose();                 // never start on an empty canvas
    } else {
      stream = this.getCameraStream();
      if (!stream) throw new Error('Camera is not running.');
    }

    this.mode = annotated ? 'annotated' : 'raw';
    this.chunks = [];
    this.framesComposited = 0;
    const bitrate = annotated ? 6_000_000 : 5_000_000;
    this.recorder = new MediaRecorder(stream, {
      mimeType: this.format.mimeType,
      videoBitsPerSecond: bitrate,
    });
    this.recorder.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };

    await new Promise((resolve, reject) => {
      this.recorder.onstart = () => resolve();
      this.recorder.onerror = (e) => reject(e.error ?? new Error('MediaRecorder error'));
      this.recorder.start(1000);      // 1 s chunks: bounded latency, no giant flush at stop
    });
    this.startedAt = performance.now();
    this.startedDate = new Date();
    this.state = 'recording';
  }

  /**
   * Composite the current preview into the recording canvas (annotated mode).
   * @param {(ctx, w, h) => void} [extras] draws unmirrored text on top
   *   (FPS, tracking status, timestamp) so it reads correctly in the file.
   */
  compose(extras) {
    if (!this._ctx) return;
    const ctx = this._ctx;
    const w = this._canvas.width, h = this._canvas.height;
    ctx.save();
    if (this.isMirrored()) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    try {
      ctx.drawImage(this.video, 0, 0, w, h);
      ctx.drawImage(this.overlay, 0, 0, w, h);
    } catch { /* video not ready */ }
    ctx.restore();
    if (extras) extras(ctx, w, h);
    this.framesComposited += 1;
    this._track?.requestFrame();
  }

  /** Timestamp for metadata, relative to the recording's own start. */
  now() { return this.isRecording ? performance.now() - this.startedAt : null; }

  /**
   * @returns {Promise<{blob:Blob,url:string,mimeType:string,ext:string,
   *   durationMs:number,baseName:string,mode:string}>}
   */
  stop() {
    if (!this.isRecording) return Promise.resolve(null);
    this.state = 'stopping';
    const durationMs = performance.now() - this.startedAt;
    const baseName = recordingBaseName(this.startedDate ?? new Date());
    return new Promise((resolve) => {
      this.recorder.onstop = () => {
        const blob = new Blob(this.chunks, { type: this.format.mimeType.split(';')[0] });
        this.chunks = [];
        this.state = 'idle';
        this._ctx = null;
        this._canvas = null;
        this._track = null;
        resolve({
          blob,
          url: URL.createObjectURL(blob),
          mimeType: this.format.mimeType,
          ext: this.format.ext,
          durationMs,
          baseName,
          mode: this.mode,
        });
      };
      this.recorder.stop();
    });
  }
}

/**
 * Save a Blob locally. Uses the Web Share sheet where it can take files (the
 * practical way to put a video into Photos/Files on iOS and Android), falling
 * back to a normal download link on desktop.
 */
export async function saveBlob(blob, filename, { preferShare = false } = {}) {
  const file = typeof File !== 'undefined' ? new File([blob], filename, { type: blob.type }) : null;
  if (preferShare && file && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: filename });
      return 'shared';
    } catch (e) {
      if (e?.name === 'AbortError') return 'cancelled';
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
  return 'downloaded';
}
