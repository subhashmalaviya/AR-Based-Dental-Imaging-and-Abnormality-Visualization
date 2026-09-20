/**
 * HUD.js — status readouts and control bindings.
 *
 * Kept free of tracking logic: it only reflects state handed to it by main.js,
 * so the tracking pipeline can be driven headlessly (as the tests do) without
 * any DOM present.
 */

import { DentalReferenceModel, DEFAULT_HALF_ARCH, TOOTH_TYPES } from '../core/DentalReferenceModel.js';

export class HUD {
  constructor(root = document) {
    this.root = root;
    this.dentalModel = new DentalReferenceModel();

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
      // --- scale / size readout ---
      scaleSource: root.getElementById('scaleSourceValue'),
      scalePxMm: root.getElementById('scalePxMmValue'),
      toothWidthMm: root.getElementById('toothWidthMmValue'),
      toothHeightMm: root.getElementById('toothHeightMmValue'),
      // --- live particular dimension error ---
      dimErrWidth:  root.getElementById('dimErrWidthValue'),
      dimErrHeight: root.getElementById('dimErrHeightValue'),
      dimMapeWidth:  root.getElementById('dimMapeWidthValue'),
      dimMapeHeight: root.getElementById('dimMapeHeightValue'),
      particularRows: root.getElementById('particularErrorRows'),
    };
    this._fpsSamples = [];
    this._initDentalArchInputs();
  }

  _initDentalArchInputs() {
    const root = this.root;
    const tabUpper = root.getElementById('tabUpperArch');
    const tabLower = root.getElementById('tabLowerArch');
    const upperCont = root.getElementById('upperArchContainer');
    const lowerCont = root.getElementById('lowerArchContainer');

    tabUpper?.addEventListener('click', () => {
      tabUpper.classList.add('active');
      tabLower?.classList.remove('active');
      if (upperCont) upperCont.hidden = false;
      if (lowerCont) lowerCont.hidden = true;
    });

    tabLower?.addEventListener('click', () => {
      tabLower.classList.add('active');
      tabUpper?.classList.remove('active');
      if (lowerCont) lowerCont.hidden = false;
      if (upperCont) upperCont.hidden = true;
    });

    const readInputs = () => {
      const readVal = (id, fallback) => {
        const el = root.getElementById(id);
        const v = parseFloat(el?.value);
        return Number.isFinite(v) && v > 0 ? v : fallback;
      };

      const dims = { upper: {}, lower: {} };
      TOOTH_TYPES.forEach((t) => {
        dims.upper[t.key] = {
          width:  readVal(`dim_upper_${t.key}_w`, DEFAULT_HALF_ARCH.upper[t.key].width),
          height: readVal(`dim_upper_${t.key}_h`, DEFAULT_HALF_ARCH.upper[t.key].height),
        };
        dims.lower[t.key] = {
          width:  readVal(`dim_lower_${t.key}_w`, DEFAULT_HALF_ARCH.lower[t.key].width),
          height: readVal(`dim_lower_${t.key}_h`, DEFAULT_HALF_ARCH.lower[t.key].height),
        };
      });

      this.dentalModel.setDimensions(dims);
    };

    // Attach input listeners
    TOOTH_TYPES.forEach((t) => {
      ['upper', 'lower'].forEach((arch) => {
        ['w', 'h'].forEach((dim) => {
          const el = root.getElementById(`dim_${arch}_${t.key}_${dim}`);
          el?.addEventListener('input', readInputs);
        });
      });
    });

    // Reset Defaults button
    root.getElementById('resetArchDefaultsBtn')?.addEventListener('click', () => {
      this.dentalModel.resetDefaults();
      TOOTH_TYPES.forEach((t) => {
        const uW = root.getElementById(`dim_upper_${t.key}_w`);
        const uH = root.getElementById(`dim_upper_${t.key}_h`);
        const lW = root.getElementById(`dim_lower_${t.key}_w`);
        const lH = root.getElementById(`dim_lower_${t.key}_h`);
        if (uW) uW.value = DEFAULT_HALF_ARCH.upper[t.key].width;
        if (uH) uH.value = DEFAULT_HALF_ARCH.upper[t.key].height;
        if (lW) lW.value = DEFAULT_HALF_ARCH.lower[t.key].width;
        if (lH) lH.value = DEFAULT_HALF_ARCH.lower[t.key].height;
      });
    });
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
            detectorName, detectorIsLearned, detectFps, recording,
            scaleInfo, toothSizesMm, toothDimError }) {
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
    this.setScaleInfo(scaleInfo ?? null, toothSizesMm ?? null, toothDimError ?? null);
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

  /** Low-light warning: add light rather than trust a dark detection. */
  setLighting(l) {
    const pill = document.getElementById('lightStatus');
    const val = document.getElementById('evalLightValue');
    if (pill) {
      pill.hidden = !l?.low;
      pill.textContent = 'Low light — add light in front of the face';
    }
    if (val) val.textContent = l?.p97 == null ? '—' : `${l.low ? 'LOW' : 'OK'} (${Math.round(l.p97)}/255)`;
  }

  setModelInfo(text) { if (this.el.modelInfo) this.el.modelInfo.textContent = text; }

  /**
   * Returns { widthMm, heightMm } from the patient dimension input form,
   * or null if the clinician hasn't filled both fields yet.
   * Values must be positive numbers to be accepted.
   */
  getDentalReferenceModel() {
    return this.dentalModel;
  }

  /** Persistent (not cleared by setBanner) — a fallback must never go unnoticed. */
  setDetectorWarning(text) {
    const el = document.getElementById('detectorWarning');
    if (!el) return;
    el.hidden = !text;
    el.textContent = text ?? '';
  }

  /**
   * Scale reference and estimated tooth sizes in mm.
   * scaleInfo comes from IrisScaler.info(); toothSizesMm is the per-track
   * array computed in main.js from bbox corners projected through the anchor.
   * particularError comes from DentalReferenceModel.computeParticularErrors().
   */
  setScaleInfo(scaleInfo, toothSizesMm, particularError) {
    const set = (el, v) => { if (el) el.textContent = v; };
    const pct = (v) => `${(v * 100).toFixed(1)}%`;

    // Scale source label.
    if (scaleInfo) {
      const src = scaleInfo.source === 'iris'
        ? `Iris (${scaleInfo.pixelsPerMm.toFixed(1)} px/mm)`
        : `Mouth width (${scaleInfo.pixelsPerMm.toFixed(1)} px/mm)`;
      set(this.el.scaleSource, src);
      set(this.el.scalePxMm, scaleInfo.pixelsPerMm.toFixed(2));
    } else {
      set(this.el.scaleSource, '—');
      set(this.el.scalePxMm, '—');
    }

    // Mean estimated tooth width / height across all visible teeth.
    if (toothSizesMm && toothSizesMm.length) {
      const meanW = toothSizesMm.reduce((s, t) => s + t.wMm, 0) / toothSizesMm.length;
      const meanH = toothSizesMm.reduce((s, t) => s + t.hMm, 0) / toothSizesMm.length;
      set(this.el.toothWidthMm,  `~${meanW.toFixed(1)} mm`);
      set(this.el.toothHeightMm, `~${meanH.toFixed(1)} mm`);
    } else {
      set(this.el.toothWidthMm, '—');
      set(this.el.toothHeightMm, '—');
    }

    // Particular tooth error metrics
    if (particularError && particularError.overall) {
      set(this.el.dimErrWidth,   `${particularError.overall.widthMAE.toFixed(2)} mm (MAE)`);
      set(this.el.dimErrHeight,  `${particularError.overall.heightMAE.toFixed(2)} mm (MAE)`);
      set(this.el.dimMapeWidth,  pct(particularError.overall.widthMAPE));
      set(this.el.dimMapeHeight, pct(particularError.overall.heightMAPE));
    } else {
      set(this.el.dimErrWidth,  '—');
      set(this.el.dimErrHeight, '—');
      set(this.el.dimMapeWidth,  '—');
      set(this.el.dimMapeHeight, '—');
    }

    // Populate per-tooth live error breakdown table
    if (this.el.particularRows) {
      if (particularError?.teeth && particularError.teeth.length) {
        const rowsHtml = particularError.teeth.map((t) => {
          const wSign = t.measuredW >= t.refW ? '+' : '-';
          const hSign = t.measuredH >= t.refH ? '+' : '-';
          return `<tr>
            <td><span class="tooth-type-tag tooth-type-tag--${t.group}">${t.label} (T${t.id})</span></td>
            <td>${t.measuredW.toFixed(1)} × ${t.measuredH.toFixed(1)} mm</td>
            <td>${t.refW.toFixed(1)} × ${t.refH.toFixed(1)} mm</td>
            <td>${wSign}${t.wAbs.toFixed(1)} / ${hSign}${t.hAbs.toFixed(1)} mm (${(t.wRel * 100).toFixed(0)}%)</td>
          </tr>`;
        }).join('');
        this.el.particularRows.innerHTML = rowsHtml;
      } else {
        this.el.particularRows.innerHTML = '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:8px">No teeth detected yet</td></tr>';
      }
    }
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
