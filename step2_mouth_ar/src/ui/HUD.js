/**
 * HUD.js — status readouts and control bindings.
 *
 * Kept free of tracking logic: it only reflects state handed to it by main.js,
 * so the tracking pipeline can be driven headlessly (as the tests do) without
 * any DOM present.
 */

export class HUD {
  constructor(root = document) {
    this.el = {
      faceStatus: root.getElementById('faceStatus'),
      mouthStatus: root.getElementById('mouthStatus'),
      anchorStatus: root.getElementById('anchorStatus'),
      fps: root.getElementById('fpsValue'),
      delegate: root.getElementById('delegateValue'),
      resolution: root.getElementById('resValue'),
      pose: root.getElementById('poseValue'),
      opening: root.getElementById('openingValue'),
      openingBar: root.getElementById('openingBar'),
      banner: root.getElementById('banner'),
      // --- Step 3 ---
      teethStatus: root.getElementById('teethStatus'),
      teethCount: root.getElementById('teethCountValue'),
      teethConf: root.getElementById('teethConfValue'),
      inferMs: root.getElementById('inferValue'),
      trackMs: root.getElementById('trackValue'),
      totalMs: root.getElementById('totalValue'),
      detector: root.getElementById('detectorValue'),
      selPanel: root.getElementById('selectedPanel'),
      selId: root.getElementById('selIdValue'),
      selConf: root.getElementById('selConfValue'),
      selTracking: root.getElementById('selTrackingValue'),
      selArch: root.getElementById('selArchValue'),
      // --- Step 3b: per-tooth 3D anchors ---
      anchorCount: root.getElementById('anchorCountValue'),
      anchorDist: root.getElementById('anchorDistValue'),
      depthSource: root.getElementById('depthSourceValue'),
      selPos: root.getElementById('selPosValue'),
      selRot: root.getElementById('selRotValue'),
      selVis: root.getElementById('selVisValue'),
      modelInfo: root.getElementById('modelInfoValue'),
      // --- research / evaluation panel ---
      evalTeeth: root.getElementById('evalTeethValue'),
      evalConf: root.getElementById('evalConfValue'),
      evalStability: root.getElementById('evalStabilityValue'),
      evalFps: root.getElementById('evalFpsValue'),
      evalDetFps: root.getElementById('evalDetFpsValue'),
      evalInfer: root.getElementById('evalInferValue'),
      evalTrack: root.getElementById('evalTrackValue'),
      evalRec: root.getElementById('evalRecValue'),
      evalDur: root.getElementById('evalDurValue'),
      recIndicator: root.getElementById('recIndicator'),
      recTimer: root.getElementById('recTimer'),
    };
    this._fpsSamples = [];
  }

  setSelectedTooth(track) {
    const p = this.el.selPanel;
    if (!p) return;
    if (!track) { p.hidden = true; return; }
    p.hidden = false;
    if (this.el.selId) this.el.selId.textContent = `T${track.id}`;
    const conf = track.smoothed?.confidence ?? track.confidence;
    if (this.el.selConf) this.el.selConf.textContent = conf.toFixed(2);
    if (this.el.selTracking) this.el.selTracking.textContent = track.status;
    if (this.el.selArch) this.el.selArch.textContent = track.arch;
    if (this.el.selVis) this.el.selVis.textContent = track.visibility ?? '—';
    const a = track.anchor3D ?? null;
    if (this.el.selPos) {
      const t = a?.getTransform();
      this.el.selPos.textContent = t
        ? `${(t.position.x * 100).toFixed(1)}, ${(t.position.y * 100).toFixed(1)}, ${(t.position.z * 100).toFixed(1)} cm`
        : '—';
    }
    if (this.el.selRot) {
      const t = a?.getTransform();
      this.el.selRot.textContent = t
        ? `${t.rotationInFace.rx.toFixed(0)}, ${t.rotationInFace.ry.toFixed(0)}, ${t.rotationInFace.rz.toFixed(0)} deg`
        : '—';
    }
  }

  setBanner(message, kind = 'info') {
    const b = this.el.banner;
    if (!b) return;
    if (!message) { b.hidden = true; b.textContent = ''; return; }
    b.hidden = false;
    b.textContent = message;
    b.className = `banner banner--${kind}`;
  }

  setPill(el, text, state) {
    if (!el) return;
    el.textContent = text;
    el.className = `pill pill--${state}`;
  }

  update({ faceDetected, mouthTracked, anchorValid, fps, delegate, resolution,
            pose, opening, coasting,
            teeth, toothTiming, toothReason, selectedTooth, anchors3D,
            detectorName, detectorIsLearned, detectFps, recording }) {
    this.setPill(this.el.faceStatus,
      faceDetected ? 'Face detected' : 'Face not detected',
      faceDetected ? 'ok' : 'bad');

    this.setPill(this.el.mouthStatus,
      mouthTracked ? 'Mouth tracked' : 'Mouth not tracked',
      mouthTracked ? 'ok' : 'bad');

    this.setPill(this.el.anchorStatus,
      anchorValid ? (coasting ? 'Anchor holding' : 'Anchor locked') : 'Anchor lost',
      anchorValid ? (coasting ? 'warn' : 'ok') : 'bad');

    if (this.el.fps) this.el.fps.textContent = fps ? fps.toFixed(1) : '—';
    if (this.el.delegate) this.el.delegate.textContent = delegate ?? '—';
    if (this.el.resolution) {
      this.el.resolution.textContent = resolution
        ? `${resolution.width}x${resolution.height}` : '—';
    }
    if (this.el.pose) {
      this.el.pose.textContent = pose
        ? `yaw ${pose.yaw.toFixed(0)}°  pitch ${pose.pitch.toFixed(0)}°  roll ${pose.roll.toFixed(0)}°`
        : '—';
    }
    if (this.el.opening) {
      this.el.opening.textContent = opening != null ? opening.toFixed(2) : '—';
    }
    if (this.el.openingBar) {
      const pct = Math.max(0, Math.min(1, (opening ?? 0) / 0.6)) * 100;
      this.el.openingBar.style.width = `${pct}%`;
    }

    // ------------------------------------------------------------ Step 3
    if (teeth) {
      const n = teeth.count;
      const label = n > 0
        ? `Teeth: ${n}`
        : `Teeth: 0${toothReason ? ` (${toothReason})` : ''}`;
      this.setPill(this.el.teethStatus, label,
        n > 0 ? (teeth.status === 'stable' ? 'ok' : 'warn') : 'bad');
      if (this.el.teethCount) this.el.teethCount.textContent = String(n);
      if (this.el.teethConf) {
        this.el.teethConf.textContent = n ? teeth.avgConfidence.toFixed(2) : '—';
      }
    }
    if (toothTiming) {
      if (this.el.inferMs) this.el.inferMs.textContent = `${toothTiming.detect.toFixed(1)} ms`;
      if (this.el.trackMs) this.el.trackMs.textContent = `${toothTiming.track.toFixed(1)} ms`;
      if (this.el.totalMs) this.el.totalMs.textContent = `${toothTiming.total.toFixed(1)} ms`;
    }
    if (this.el.detector && detectorName) {
      // Say plainly what the detector is: the classical method must never read
      // as an AI result, and the learned one says what it was trained on.
      this.el.detector.textContent = detectorIsLearned
        ? `${detectorName} — trained model, on-device`
        : `${detectorName} — not a neural network`;
    }
    this.setEvaluation({ teeth, toothTiming, fps, detectFps, recording });
    this.setAnchors3D(anchors3D ?? null);
    this.setSelectedTooth(selectedTooth ?? null);
  }

  /** Research / evaluation panel — every value is a live measurement. */
  setEvaluation({ teeth, toothTiming, fps, detectFps, recording }) {
    const set = (el, v) => { if (el) el.textContent = v; };
    if (fps === undefined && teeth === undefined) {
      // recording-only refresh (timer tick): leave the live stats untouched
      const on = !!recording?.on;
      set(this.el.evalRec, on ? `ON (${recording.mode})` : 'OFF');
      set(this.el.evalDur, formatDuration(recording?.ms ?? 0));
      if (this.el.recIndicator) this.el.recIndicator.hidden = !on;
      set(this.el.recTimer, formatDuration(recording?.ms ?? 0));
      return;
    }
    if (teeth) {
      set(this.el.evalTeeth, teeth.count
        ? `${teeth.count}  (upper ${teeth.upper ?? 0}, lower ${teeth.lower ?? 0})` : '0');
      set(this.el.evalConf, teeth.count ? teeth.avgConfidence.toFixed(2) : '—');
      set(this.el.evalStability, teeth.stability == null ? '—'
        : `${teeth.stabilityLabel} (${(teeth.stability * 100).toFixed(0)}%)`);
    }
    set(this.el.evalFps, fps ? fps.toFixed(1) : '—');
    set(this.el.evalDetFps, detectFps ? detectFps.toFixed(1) : '—');
    if (toothTiming) {
      set(this.el.evalInfer, `${toothTiming.detect.toFixed(1)} ms`);
      set(this.el.evalTrack, `${toothTiming.track.toFixed(1)} ms`);
    }
    const on = !!recording?.on;
    const dur = formatDuration(recording?.ms ?? 0);
    set(this.el.evalRec, on ? `ON (${recording.mode})` : 'OFF');
    set(this.el.evalDur, dur);
    if (this.el.recIndicator) this.el.recIndicator.hidden = !on;
    set(this.el.recTimer, dur);
  }

  setModelInfo(text) { if (this.el.modelInfo) this.el.modelInfo.textContent = text; }

  /** Persistent (not cleared by setBanner) — a fallback must never go unnoticed. */
  setDetectorWarning(text) {
    const el = document.getElementById('detectorWarning');
    if (!el) return;
    el.hidden = !text;
    el.textContent = text ?? '';
  }

  /**
   * 3D anchor read-out. Distance and depth source are shown together on
   * purpose: the number is only as good as where it came from, and the user
   * should be able to see which of the two estimators is running.
   */
  setAnchors3D(info) {
    const n = info?.count ?? 0;
    if (this.el.anchorCount) this.el.anchorCount.textContent = String(n);
    if (this.el.anchorDist) {
      this.el.anchorDist.textContent = info?.mouthDepthM
        ? `~${(info.mouthDepthM * 100).toFixed(1)} cm (estimated)` : '—';
    }
    if (this.el.depthSource) {
      this.el.depthSource.textContent = info?.depthSource
        ? (info.depthSource === 'mediapipe-metric-head-model'
          ? 'MediaPipe metric head model'
          : 'apparent mouth width (assumed 50 mm)')
        : '—';
    }
  }
}

export function formatDuration(ms) {
  const t = Math.floor(ms / 1000);
  const m = Math.floor(t / 60), sec = t % 60;
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
