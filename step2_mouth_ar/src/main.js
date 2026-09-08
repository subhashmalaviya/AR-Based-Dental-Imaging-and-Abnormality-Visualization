/**
 * main.js — application wiring for Step 2.
 *
 * Per-frame pipeline:
 *
 *   CameraManager  ->  FaceTracker  ->  MouthTracker  ->  MouthARAnchor
 *        (video)        (landmarks)      (mouth geom)     (smoothed pose)
 *                                                              |
 *                                       LandmarkRenderer  <----+
 *                                       AROverlayController <--+
 *
 * The render loop is driven by requestVideoFrameCallback where available so we
 * process each camera frame exactly once, rather than requestAnimationFrame
 * which can fire more often than the camera delivers (wasted inference) or
 * less often (dropped frames).
 */
import { CameraManager } from './core/CameraManager.js';
import { FaceTracker } from './core/FaceTracker.js';
import { MouthTracker } from './core/MouthTracker.js';
import { MouthARAnchor } from './core/MouthARAnchor.js';
import { AROverlayController } from './core/AROverlayController.js';
import { LandmarkRenderer } from './ui/LandmarkRenderer.js';
import { HUD } from './ui/HUD.js';
// --- Step 3 -------------------------------------------------------------
import { ToothPipeline } from './core/ToothPipeline.js';
import { ToothOverlayRenderer } from './ui/ToothOverlayRenderer.js';
import { DebugRenderer } from './ui/DebugRenderer.js';

const video = document.getElementById('camera');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const hud = new HUD(document);
const camera = new CameraManager(video, { width: 1280, height: 720, facingMode: 'user' });
const faceTracker = new FaceTracker({
  wasmPath: `${import.meta.env.BASE_URL}wasm`,
  modelPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
});
const mouthTracker = new MouthTracker();
const anchor = new MouthARAnchor({ smoothing: 'balanced' });
const overlay = new AROverlayController(ctx);
const landmarks = new LandmarkRenderer(ctx);

// --- Step 3 ---------------------------------------------------------------
const teeth = new ToothPipeline({ detector: 'classical', smoothing: 'balanced' });
const toothRenderer = new ToothOverlayRenderer(ctx);
const debugRenderer = new DebugRenderer(ctx);

const state = {
  running: false,
  showLandmarks: true,
  showMesh: false,
  showOverlay: true,
  mirrored: true,
  // Step 3 — the mouth AR quad is off by default now, because it would sit on
  // top of the teeth it is meant to let you see.
  showTeeth: true,
  toothDebug: false,   // hides Step-2 clutter + shows the segmenter's own view
};

// ---------------------------------------------------------------- overlay art
const logo = new Image();
logo.crossOrigin = 'anonymous';
logo.src = `${import.meta.env.BASE_URL}iitd_logo.png`;
logo.onload = () => overlay.setImage(logo);
logo.onerror = () => {
  console.warn('[main] overlay image missing, falling back to a rectangle');
  overlay.setMode('rect');
};

// ------------------------------------------------------------------ rendering
function resizeCanvas() {
  const { width, height } = camera.frameSize;
  if (!width || !height) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  // The preview is mirrored for the front camera (users expect a mirror), and
  // the canvas is mirrored with it via CSS so overlay pixels stay registered to
  // video pixels. Tracking maths always runs in unmirrored frame coordinates.
  const mirror = camera.isMirrored;
  const t = mirror ? 'scaleX(-1)' : 'none';
  video.style.transform = t;
  canvas.style.transform = t;
  state.mirrored = mirror;
  // Text/panels must be un-mirrored or every label reads backwards.
  toothRenderer.setMirrored(mirror);
  debugRenderer.setMirrored(mirror);
}

function drawFrame(landmarkList, mouth, toothTracks, toothStats) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const { width: w, height: h } = canvas;

  // Tooth-debug mode suppresses the Step-2 lip outline, anchor axes and AR
  // quad. Those are drawn in the same place as the teeth and, at a normal
  // camera distance, completely bury the tooth contours — which is the main
  // reason detection *looked* like it wasn't working.
  const clutter = !state.toothDebug;

  if (state.showLandmarks && clutter) {
    if (state.showMesh && landmarkList) landmarks.drawAllLandmarks(landmarkList, w, h);
    if (mouth && landmarkList) landmarks.drawMouth(mouth, landmarkList, w, h);
    landmarks.drawAnchor(anchor);
  }
  if (state.showOverlay && clutter) overlay.render(anchor);

  // Step 3 — drawn last so tooth contours are never painted over.
  debugRenderer.drawROI(teeth.roi, anchor);
  if (state.showTeeth && toothTracks?.length) {
    toothRenderer.render(toothTracks, anchor, teeth.selectedId);
  }
  debugRenderer.drawSegmenterView(
    teeth.lastRoiImage, teeth.debugData, toothTracks, teeth.roi, w, h);
  debugRenderer.drawStats(teeth.debugData, toothStats, teeth.timing, teeth.detectFps, w);
}

// ----------------------------------------------------------------- main loop
let rvfcHandle = null;
let rafHandle = null;

function processFrame(nowMs) {
  if (!state.running) return;

  resizeCanvas();
  const { width, height } = camera.frameSize;

  let landmarkList = null;
  let mouth = null;

  if (width && height) {
    const res = faceTracker.detect(video, nowMs);
    landmarkList = res.landmarks;
    if (landmarkList) mouth = mouthTracker.track(landmarkList, width, height);
  }

  // Anchor is updated every frame, including when the mouth is missing, so it
  // can run its own coast/lost logic rather than being frozen by the caller.
  anchor.update(mouth, nowMs / 1000);

  // Step 3 — runs after the anchor, and consumes its pose. Wrapped so a fault
  // in the tooth stage can never take down face/mouth tracking.
  let toothResult = { tracks: [], stats: teeth.tracker.stats() };
  try {
    toothResult = teeth.update(
      video, landmarkList, mouth, anchor, nowMs / 1000, width, height);
  } catch (err) {
    console.warn('[main] tooth pipeline error (face/mouth tracking unaffected):', err);
  }

  const fps = camera.tick(nowMs);
  drawFrame(landmarkList, mouth, toothResult.tracks, toothResult.stats);

  const readout = anchor.getReadout();
  hud.update({
    faceDetected: !!landmarkList,
    mouthTracked: !!mouth,
    anchorValid: anchor.isValid(),
    coasting: anchor.isValid() && !mouth,
    fps,
    delegate: faceTracker.delegate,
    resolution: { width, height },
    teeth: toothResult.stats,
    toothTiming: teeth.timing,
    toothReason: teeth.reason,
    selectedTooth: teeth.getSelected(),
    detectorName: teeth.detector.name,
    detectorIsLearned: teeth.detector.isLearnedModel,
    pose: readout ? { yaw: readout.yaw, pitch: readout.pitch, roll: readout.roll } : null,
    opening: readout?.mouthOpen,
  });

  scheduleNext();
}

function scheduleNext() {
  if (!state.running) return;
  if (typeof video.requestVideoFrameCallback === 'function') {
    rvfcHandle = video.requestVideoFrameCallback((now) => processFrame(now));
  } else {
    rafHandle = requestAnimationFrame((now) => processFrame(now));
  }
}

function stopLoop() {
  state.running = false;
  if (rvfcHandle && video.cancelVideoFrameCallback) {
    video.cancelVideoFrameCallback(rvfcHandle);
  }
  if (rafHandle) cancelAnimationFrame(rafHandle);
  rvfcHandle = rafHandle = null;
}

// --------------------------------------------------------------- lifecycle
async function start() {
  try {
    hud.setBanner('Requesting camera…', 'info');
    await camera.start();
    resizeCanvas();

    if (!faceTracker.ready) {
      hud.setBanner('Loading face model…', 'info');
      const delegate = await faceTracker.init((msg) => hud.setBanner(msg, 'info'));
      console.log(`[main] FaceLandmarker ready on ${delegate}`);
    }

    hud.setBanner('', 'info');
    anchor.reset();
    teeth.reset();
    state.running = true;
    scheduleNext();
    setButtons(true);
  } catch (err) {
    console.error(err);
    hud.setBanner(err.message || String(err), 'error');
    setButtons(false);
  }
}

async function stop() {
  stopLoop();
  await camera.stop();
  anchor.reset();
  teeth.reset();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hud.update({ faceDetected: false, mouthTracked: false, anchorValid: false, fps: 0 });
  setButtons(false);
}

function setButtons(running) {
  document.getElementById('startBtn').disabled = running;
  document.getElementById('stopBtn').disabled = !running;
  document.getElementById('switchBtn').disabled = !running;
}

// Release the camera when the tab is hidden; mobile browsers may otherwise
// keep the sensor hot, and resuming with a stale timestamp upsets MediaPipe.
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.running) stopLoop();
  else if (!document.hidden && camera.running && !state.running) {
    state.running = true;
    scheduleNext();
  }
});

// ----------------------------------------------------------------- controls
document.getElementById('startBtn').addEventListener('click', start);
document.getElementById('stopBtn').addEventListener('click', stop);
document.getElementById('switchBtn').addEventListener('click', async () => {
  stopLoop();
  await camera.switchCamera();
  anchor.reset();
  teeth.reset();
  state.running = true;
  scheduleNext();
});

document.getElementById('landmarksToggle').addEventListener('change', (e) => {
  state.showLandmarks = e.target.checked;
});
document.getElementById('meshToggle').addEventListener('change', (e) => {
  state.showMesh = e.target.checked;
});
document.getElementById('overlayToggle').addEventListener('change', (e) => {
  state.showOverlay = e.target.checked;
  overlay.setVisible(e.target.checked);
});
document.getElementById('smoothingSelect').addEventListener('change', (e) => {
  anchor.setSmoothing(e.target.value);
});
document.getElementById('opacityRange').addEventListener('input', (e) => {
  overlay.setOpacity(Number(e.target.value));
});
document.getElementById('sizeRange').addEventListener('input', (e) => {
  overlay.setPlacement({ width: Number(e.target.value) });
});
document.getElementById('modeSelect').addEventListener('change', (e) => {
  overlay.setMode(e.target.value);
});

// ------------------------------------------------------- Step 3 controls
const bind = (id, fn) => document.getElementById(id)?.addEventListener('change', fn);

bind('teethToggle', (e) => { state.showTeeth = e.target.checked; teeth.setEnabled(e.target.checked); });
bind('toothDebugToggle', (e) => {
  state.toothDebug = e.target.checked;
  // Turning debug on implies you want to see the segmenter's view and numbers.
  debugRenderer.setShow({ rectified: e.target.checked, stats: e.target.checked });
  const rect = document.getElementById('rectifiedToggle');
  const st = document.getElementById('statsToggle');
  if (rect) rect.checked = e.target.checked;
  if (st) st.checked = e.target.checked;
});
bind('statsToggle', (e) => debugRenderer.setShow({ stats: e.target.checked }));
bind('contoursToggle', (e) => toothRenderer.setShow({ contours: e.target.checked }));
bind('toothIdsToggle', (e) => toothRenderer.setShow({ ids: e.target.checked }));
bind('confToggle', (e) => toothRenderer.setShow({ confidence: e.target.checked }));
bind('boxesToggle', (e) => toothRenderer.setShow({ boxes: e.target.checked }));
bind('roiToggle', (e) => debugRenderer.setShow({ roi: e.target.checked }));
bind('rectifiedToggle', (e) => debugRenderer.setShow({ rectified: e.target.checked }));
bind('toothSmoothingSelect', (e) => teeth.setSmoothing(e.target.value));
bind('detectEveryNSelect', (e) => teeth.setDetectEveryN(Number(e.target.value)));

document.getElementById('splitRange')?.addEventListener('input', (e) => {
  teeth.detector.setParams?.({ splitRatio: Number(e.target.value) });
});
document.getElementById('logFrameBtn')?.addEventListener('click', () => {
  teeth.logNextFrame();
  hud.setBanner('Logged one frame of detections to the browser console (F12).', 'info');
  setTimeout(() => hud.setBanner('', 'info'), 3500);
});
document.getElementById('resetTeethBtn')?.addEventListener('click', () => {
  teeth.reset();
  hud.setSelectedTooth(null);
});

// Tap/click a tooth to select it. Canvas coordinates must be un-mirrored first,
// because the preview is CSS-mirrored for the front camera while the tracking
// maths lives in unmirrored frame coordinates.
canvas.style.pointerEvents = 'auto';
canvas.addEventListener('pointerdown', (ev) => {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  let nx = (ev.clientX - rect.left) / rect.width;
  const ny = (ev.clientY - rect.top) / rect.height;
  if (state.mirrored) nx = 1 - nx;
  teeth.selectAt({ x: nx * canvas.width, y: ny * canvas.height }, anchor);
});

setButtons(false);
if (!CameraManager.isSecureContext()) {
  hud.setBanner(
    'This page is not a secure context, so the camera is blocked. '
    + 'Use the HTTPS URL or localhost (see README).', 'error');
}

// Expose the pipeline for console poking / Step-3 experiments.
window.dentalAR = {
  camera, faceTracker, mouthTracker, anchor, overlay, state,
  // Step 3
  teeth, toothRenderer, debugRenderer,
};
