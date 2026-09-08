/**
 * CameraManager.js — live device camera acquisition.
 *
 * Wraps getUserMedia and the <video> element, handles front/back switching,
 * resolution negotiation and FPS measurement. This is the only module that
 * touches camera APIs, so porting to a different host (a native shell, a
 * different framework) means replacing this file alone.
 *
 * Notes that matter on real devices:
 *  - getUserMedia requires a **secure context**. https:// or http://localhost
 *    only; a phone hitting http://192.168.x.x will silently have no camera.
 *    See README "Testing on a phone".
 *  - iOS Safari needs `playsinline` on the <video> or it hijacks playback into
 *    the fullscreen player and the canvas gets nothing.
 *  - Requested resolution is a hint. Phones often hand back something else, so
 *    everything downstream reads videoWidth/videoHeight rather than assuming.
 */

export class CameraManager {
  constructor(videoElement, { width = 1280, height = 720, facingMode = 'user' } = {}) {
    this.video = videoElement;
    this.desired = { width, height };
    this.facingMode = facingMode;
    this.stream = null;
    this.running = false;

    // FPS is measured over a sliding window rather than from the last frame,
    // so the readout does not flicker on individual slow frames.
    this._frameTimes = [];
    this._fps = 0;
  }

  static isSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  static isSecureContext() {
    return window.isSecureContext === true;
  }

  async start() {
    if (!CameraManager.isSupported()) {
      throw new Error('getUserMedia is unavailable in this browser.');
    }
    if (!CameraManager.isSecureContext()) {
      throw new Error(
        'Camera needs a secure context (https:// or localhost). '
        + 'Open the HTTPS URL — see README "Testing on a phone".');
    }
    await this.stop();

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.facingMode },
        width: { ideal: this.desired.width },
        height: { ideal: this.desired.height },
        frameRate: { ideal: 30, max: 60 },
      },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      // Retry without the resolution hints; some devices reject over-specified
      // constraints outright with OverconstrainedError.
      if (err && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false, video: { facingMode: this.facingMode },
        });
      } else {
        throw CameraManager._describeError(err);
      }
    }

    this.video.srcObject = this.stream;
    this.video.setAttribute('playsinline', '');
    this.video.muted = true;
    await this.video.play();
    await this._waitForMetadata();

    this.running = true;
    this._frameTimes = [];
    return this.getSettings();
  }

  _waitForMetadata() {
    if (this.video.videoWidth > 0) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => { this.video.removeEventListener('loadeddata', done); resolve(); };
      this.video.addEventListener('loadeddata', done);
    });
  }

  static _describeError(err) {
    const map = {
      NotAllowedError: 'Camera permission was denied. Allow camera access and reload.',
      NotFoundError: 'No camera was found on this device.',
      NotReadableError: 'The camera is already in use by another app.',
      SecurityError: 'Blocked for security reasons — the page must be served over HTTPS.',
    };
    const msg = map[err?.name] || err?.message || 'Could not start the camera.';
    return new Error(msg);
  }

  async stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
    this.running = false;
  }

  async switchCamera() {
    this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
    return this.start();
  }

  /** Call once per processed frame to keep the FPS readout current. */
  tick(nowMs = performance.now()) {
    this._frameTimes.push(nowMs);
    while (this._frameTimes.length > 30) this._frameTimes.shift();
    const n = this._frameTimes.length;
    if (n >= 2) {
      const span = (this._frameTimes[n - 1] - this._frameTimes[0]) / 1000;
      this._fps = span > 0 ? (n - 1) / span : 0;
    }
    return this._fps;
  }

  get fps() { return this._fps; }

  get frameSize() {
    return { width: this.video?.videoWidth || 0, height: this.video?.videoHeight || 0 };
  }

  /** True when the front camera is active, so the preview should be mirrored. */
  get isMirrored() { return this.facingMode === 'user'; }

  getSettings() {
    const track = this.stream?.getVideoTracks?.()[0];
    const s = track?.getSettings?.() ?? {};
    return {
      width: this.video.videoWidth,
      height: this.video.videoHeight,
      facingMode: s.facingMode ?? this.facingMode,
      frameRate: s.frameRate,
      deviceLabel: track?.label ?? 'camera',
    };
  }
}
