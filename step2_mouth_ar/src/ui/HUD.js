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
            teeth, toothTiming, toothReason, selectedTooth,
            detectorName, detectorIsLearned }) {
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
      // Say plainly what the detector is. This is a classical CV method, and
      // the UI must not let it read as an AI result.
      this.el.detector.textContent = detectorIsLearned
        ? detectorName : `${detectorName} — not a neural network`;
    }
    this.setSelectedTooth(selectedTooth ?? null);
  }
}
