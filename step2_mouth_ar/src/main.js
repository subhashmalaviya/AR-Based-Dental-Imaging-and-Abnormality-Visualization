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
import { AR3DRenderer } from './ui/AR3DRenderer.js';
// --- Step 3 v2: recording + metadata -----------------------------------------
import { SessionRecorder, saveBlob } from './core/SessionRecorder.js';
import { MetadataLogger } from './core/MetadataLogger.js';
import { formatDuration } from './ui/HUD.js';

const APP_VERSION = '3.2.0';
const MODEL_URL = `${import.meta.env.BASE_URL}models/tooth_seg.onnx`;

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
const ar3d = new AR3DRenderer(ctx);

// --- recording ------------------------------------------------------------
const recorder = new SessionRecorder({
  getCameraStream: () => camera.stream,
  video,
  overlay: canvas,
  isMirrored: () => state.mirrored,
});
const metaLog = new MetadataLogger();
let lastRecording = null;   // { result, doc } of the most recent recording

const state = {
  running: false,
  showLandmarks: true,
  showMesh: false,
  // Step-2 logo quad: off by default — it sits exactly on the teeth this
  // step is about, and would be burned into annotated recordings.
  showOverlay: false,
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
overlay.setVisible(false);
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
  ar3d.setMirrored(mirror);
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
  // 3D proxies: one per tooth anchor, projected through each tooth's own
  // transform (not scaled 2D sprites).
  ar3d.render(teeth.anchors3D.list(), teeth.intrinsics, w, h);
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
  let headMatrix = null;   // MediaPipe's tracked 4x4 head pose — real 3D

  if (width && height) {
    const res = faceTracker.detect(video, nowMs);
    landmarkList = res.landmarks;
    headMatrix = res.matrix;
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
      video, landmarkList, mouth, anchor, nowMs / 1000, width, height, headMatrix);
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
    anchors3D: teeth.anchor3DInfo(),
    detectorName: teeth.detector.name,
    detectorIsLearned: teeth.detector.isLearnedModel,
    pose: readout ? { yaw: readout.yaw, pitch: readout.pitch, roll: readout.roll } : null,
    opening: readout?.mouthOpen,
    detectFps: teeth.detectFps,
    recording: { on: recorder.isRecording, mode: recorder.mode, ms: recorder.elapsedMs },
  });

  // Recording: composite the annotated view and log this frame's analysis.
  // Both use values already computed above; nothing is re-derived for them.
  if (recorder.isRecording) {
    if (recorder.mode === 'annotated') {
      recorder.compose((c, w, h) => drawBurnIn(c, w, h, fps, toothResult.stats));
    }
    if (metaLog.active) {
      metaLog.log({
        t_ms: recorder.now(),
        fps,
        face: !!landmarkList,
        mouth: !!mouth,
        opening: readout?.mouthOpen,
        reason: teeth.reason,
        stats: toothResult.stats,
        timing: teeth.timing,
        tracks: toothResult.tracks,
        anchor,
        anchors3D: teeth.anchors3D,
      });
    }
  }

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
  if (recorder.isRecording) await toggleRecording();
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
  document.getElementById('switchBtn').disabled = !running || recorder.isRecording;
  const rec = document.getElementById('recordBtn');
  if (rec) rec.disabled = !running || !recorder.isSupported;
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
bind('ar3dToggle', (e) => ar3d.setVisible(e.target.checked));
bind('ar3dModeSelect', (e) => ar3d.setMode(e.target.value));
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
for (const id of ['resetTeethBtn', 'resetTrackingBtn']) {
  document.getElementById(id)?.addEventListener('click', () => {
    teeth.reset();
    hud.setSelectedTooth(null);
  });
}

// ------------------------------------------------------- detector choice
const fmt = (v) => (v == null ? 'n/a' : v.toFixed(2));

function describeDetector(det) {
  if (!det.isLearnedModel) {
    return 'Hand-designed whiteness threshold + interdental split. Kept as the '
      + 'measured baseline; not a trained model.';
  }
  const i = det.info;
  if (!i) return 'Learned model loaded (no model card found next to it).';
  const ds = (i.training?.datasets ?? []).map((d) => `${d.name} (${d.license})`).join(' + ');
  const v = i.validation ?? {};
  return `${i.name} v${i.version} — ${Math.round(i.parameters / 1000)}k parameters, `
    + `${i.input.width}x${i.input.height} input, runs on-device (ONNX Runtime Web). `
    + `Trained on ${ds}. Held-out validation: teeth-mask IoU ${fmt(v.ep_teeth_iou)} on selfie `
    + `images, ${fmt(v.da_teeth_iou)} on intraoral photos; tooth-centre F1 ${fmt(v.da_center_f1)}.`;
}

async function selectDetector(key) {
  const sel = document.getElementById('detectorSelect');
  try {
    if (key === 'learned') hud.setBanner('Loading tooth model…', 'info');
    const det = await teeth.setDetector(key, key === 'learned' ? { modelUrl: MODEL_URL } : {});
    hud.setModelInfo(describeDetector(det));
    hud.setBanner('', 'info');
  } catch (err) {
    console.warn('[main] could not load detector', key, err);
    if (key !== 'classical') {
      await teeth.setDetector('classical');
      if (sel) sel.value = 'classical';
      hud.setModelInfo(describeDetector(teeth.detector));
      hud.setBanner(`Learned tooth model unavailable (${err.message}). `
        + 'Falling back to the classical detector.', 'error');
    }
  }
}
bind('detectorSelect', (e) => selectDetector(e.target.value));

// ------------------------------------------------------------- recording
const recBtn = document.getElementById('recordBtn');
const recFormat = document.getElementById('recFormatValue');
if (recFormat) {
  recFormat.textContent = recorder.isSupported
    ? `Format: ${recorder.format.mimeType} → .${recorder.format.ext}`
    : 'Recording is not supported in this browser (no MediaRecorder).';
}

/** Small text block burned into annotated recordings (unmirrored). */
function drawBurnIn(c, w, h, fps, stats) {
  const lines = [
    `Dental AR ${APP_VERSION}  ${new Date().toLocaleString()}`,
    `FPS ${fps.toFixed(1)}   teeth ${stats?.count ?? 0} (U${stats?.upper ?? 0}/L${stats?.lower ?? 0})`
      + `   conf ${stats?.count ? stats.avgConfidence.toFixed(2) : '—'}`,
    `tracking ${stats?.stabilityLabel ?? '—'}${stats?.stability != null ? ` ${(stats.stability * 100).toFixed(0)}%` : ''}`
      + `   detector ${teeth.detectorKey}   rec ${formatDuration(recorder.elapsedMs)}`,
  ];
  const fs = Math.max(11, Math.round(h / 48));
  c.save();
  c.font = `600 ${fs}px ui-monospace, Menlo, monospace`;
  const bw = Math.max(...lines.map((l) => c.measureText(l).width)) + 16;
  c.fillStyle = 'rgba(6,10,16,0.72)';
  c.fillRect(8, 8, bw, lines.length * (fs + 5) + 10);
  c.fillStyle = '#e6f6ff';
  lines.forEach((l, i) => c.fillText(l, 16, 8 + (i + 1) * (fs + 5)));
  c.restore();
}

// The indicator and timer must not depend on frames being processed: on a
// slow phone, or while the face is out of view, the loop can be sparse.
let recTick = null;
function refreshRecordingHud() {
  hud.setEvaluation({
    recording: { on: recorder.isRecording, mode: recorder.mode, ms: recorder.elapsedMs },
  });
}

function setRecordingUi(on) {
  clearInterval(recTick);
  recTick = on ? setInterval(refreshRecordingHud, 250) : null;
  refreshRecordingHud();
  if (!recBtn) return;
  recBtn.textContent = on ? '\u25A0 STOP RECORDING' : '\u25CF RECORD VIDEO';
  recBtn.classList.toggle('is-recording', on);
  for (const id of ['recordAnnotationsToggle', 'recordMetadataToggle', 'detectorSelect', 'switchBtn']) {
    const el = document.getElementById(id);
    if (el) el.disabled = on;
  }
}

async function toggleRecording() {
  if (!recorder.isRecording) {
    const annotated = !!document.getElementById('recordAnnotationsToggle')?.checked;
    try {
      await recorder.start({ annotated });
    } catch (err) {
      hud.setBanner(err.message || String(err), 'error');
      return;
    }
    if (document.getElementById('recordMetadataToggle')?.checked) {
      const i = teeth.detector.info;
      metaLog.begin({
        app: 'dental-ar', appVersion: APP_VERSION,
        startedAt: new Date().toISOString(),
        mode: recorder.mode,
        video: { mimeType: recorder.format.mimeType, width: video.videoWidth, height: video.videoHeight },
        mirrored: { preview: state.mirrored, annotatedVideo: annotated && state.mirrored, rawVideo: false },
        coordinates: 'bbox/center/contour: raw (unmirrored) video pixels; uv: mouth-local (1.0 = mouth width); '
          + 'pos3d_m: camera metres (depth estimated, see README Step 3b)',
        timebase: 't_ms is milliseconds since the MediaRecorder start event',
        detector: {
          key: teeth.detectorKey, name: teeth.detector.name, learned: teeth.detector.isLearnedModel,
          model: i ? { name: i.name, version: i.version, created: i.created } : null,
        },
        camera: camera.getSettings(),
        userAgent: navigator.userAgent,
      });
    }
    setRecordingUi(true);
    return;
  }
  if (recBtn) recBtn.disabled = true;
  const result = await recorder.stop();
  const doc = metaLog.active ? metaLog.end({
    recording: {
      mode: result.mode, duration_ms: Math.round(result.durationMs),
      mimeType: result.mimeType, bytes: result.blob.size,
      frames_composited: recorder.framesComposited,
    },
  }) : null;
  setRecordingUi(false);
  if (recBtn) recBtn.disabled = !state.running;
  showRecording(result, doc);
}

function showRecording(result, doc) {
  if (lastRecording?.result?.url) URL.revokeObjectURL(lastRecording.result.url);
  lastRecording = { result, doc };
  const box = document.getElementById('recPreview');
  const vid = document.getElementById('recVideo');
  const info = document.getElementById('recInfo');
  if (!box) return;
  box.hidden = false;
  vid.src = result.url;
  const mb = (result.blob.size / 1e6).toFixed(1);
  info.textContent = `${result.baseName}.${result.ext} — ${(result.durationMs / 1000).toFixed(1)} s, `
    + `${mb} MB, ${result.mode}${doc ? `, ${doc.frames.length} metadata frames` : ', no metadata'}`;
  document.getElementById('recSaveJsonBtn').disabled = !doc;
}

recBtn?.addEventListener('click', toggleRecording);
document.getElementById('recSaveVideoBtn')?.addEventListener('click', () => {
  const r = lastRecording?.result;
  if (r) saveBlob(r.blob, `${r.baseName}.${r.ext}`);
});
document.getElementById('recSaveJsonBtn')?.addEventListener('click', () => {
  const r = lastRecording;
  if (r?.doc) saveBlob(MetadataLogger.toBlob(r.doc), `${r.result.baseName}.json`);
});
document.getElementById('recShareBtn')?.addEventListener('click', async () => {
  const r = lastRecording;
  if (!r) return;
  const files = [new File([r.result.blob], `${r.result.baseName}.${r.result.ext}`, { type: r.result.blob.type })];
  if (r.doc) files.push(new File([MetadataLogger.toBlob(r.doc)], `${r.result.baseName}.json`, { type: 'application/json' }));
  if (navigator.canShare?.({ files })) {
    try { await navigator.share({ files, title: r.result.baseName }); return; } catch (e) {
      if (e?.name === 'AbortError') return;
    }
  }
  for (const f of files) await saveBlob(f, f.name);
});
document.getElementById('recDiscardBtn')?.addEventListener('click', () => {
  if (lastRecording?.result?.url) URL.revokeObjectURL(lastRecording.result.url);
  lastRecording = null;
  const box = document.getElementById('recPreview');
  if (box) box.hidden = true;
  document.getElementById('recVideo').removeAttribute('src');
});

// Capture the current raw frame + its detections, for ground-truth annotation.
document.getElementById('captureGtBtn')?.addEventListener('click', async () => {
  if (!state.running || !video.videoWidth) return;
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  const png = await new Promise((r) => c.toBlob(r, 'image/png'));
  const one = new MetadataLogger();
  one.begin({ app: 'dental-ar', appVersion: APP_VERSION, startedAt: new Date().toISOString(),
    mode: 'still', video: { mimeType: 'image/png', width: c.width, height: c.height },
    mirrored: { preview: state.mirrored, rawVideo: false },
    detector: { key: teeth.detectorKey, name: teeth.detector.name, learned: teeth.detector.isLearnedModel } });
  one.log({ t_ms: 0, fps: camera.fps, face: true, mouth: true, stats: teeth.tracker.stats(),
    timing: teeth.timing, tracks: teeth.tracker.visibleTracks(), anchor, anchors3D: teeth.anchors3D });
  const base = `dental_frame_${new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').slice(0, 15)}`;
  await saveBlob(png, `${base}.png`);
  await saveBlob(MetadataLogger.toBlob(one.end()), `${base}.json`);
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
selectDetector(document.getElementById('detectorSelect')?.value ?? 'learned');
if (!CameraManager.isSecureContext()) {
  hud.setBanner(
    'This page is not a secure context, so the camera is blocked. '
    + 'Use the HTTPS URL or localhost (see README).', 'error');
}

// Expose the pipeline for console poking / Step-3 experiments.
window.dentalAR = {
  camera, faceTracker, mouthTracker, anchor, overlay, state,
  // Step 3
  teeth, toothRenderer, debugRenderer, ar3d,
  recorder, metaLog,
  getLastRecording: () => lastRecording,
};
