/**
 * IrisScaler.js — pixel-to-millimetre scale estimation from biological reference landmarks.
 *
 * ==========================================================================
 * WHAT THIS MODULE DOES AND WHY IT IS NEEDED
 * ==========================================================================
 * A single RGB camera cannot observe absolute metric sizes directly. To convert
 * the bounding-box dimension errors reported by metrics.js from pixels into
 * millimetres, we need to know how many pixels correspond to 1 mm at the plane
 * of the teeth. We derive this from two biological references that MediaPipe
 * already tracks, requiring nothing extra from the user:
 *
 *   1. IRIS DIAMETER (primary, most accurate)
 *      The human iris horizontal diameter is remarkably consistent across adult
 *      populations: 11.7 +/- 0.5 mm (Bekerman et al. 2014). MediaPipe Face Mesh
 *      (with iris refinement enabled) tracks four iris-edge landmarks per eye.
 *      The horizontal distance between them, measured in pixels, directly gives
 *      pixels-per-mm at the eye plane, which is essentially the same as the
 *      tooth plane over a mouth-sized depth range.
 *
 *   2. INTER-COMMISSURE MOUTH WIDTH (fallback)
 *      Average adult inter-commissure distance ~50 mm (Farkas 1994). This is
 *      the same prior ToothPoseEstimator already uses. Useful when iris
 *      landmarks are unavailable (e.g. iris refinement not enabled).
 *
 * ==========================================================================
 * ACCURACY AND LIMITATIONS
 * ==========================================================================
 * - Iris diameter varies by +/-0.5 mm (~4%), so metric errors derived here
 *   carry roughly +/-4% absolute uncertainty. At 500 mm camera distance and a
 *   typical tooth width of 8.5 mm, a 4% error means +/-0.34 mm -- acceptable
 *   for a clinical tracking tool, though not as accurate as a depth sensor.
 *
 * - The scale is computed at the eye/mouth plane. Teeth sit a few mm behind
 *   that plane, introducing a small perspective error (< 1% at normal working
 *   distances of 300-600 mm). The arch-depth correction in ToothPoseEstimator
 *   accounts for this when 3D metrics are needed.
 *
 * - For precise per-patient measurement, replace with a known reference
 *   (e.g. patient-supplied crown dimensions or a calibration card at mouth
 *   depth). The API is unchanged.
 *
 * ==========================================================================
 * USAGE
 * ==========================================================================
 *   import { IrisScaler } from './IrisScaler.js';
 *
 *   const scaler = new IrisScaler();
 *
 *   // Call once per frame with the raw MediaPipe landmark array:
 *   scaler.update(landmarks, frameWidth, frameHeight);
 *
 *   if (scaler.isReady()) {
 *     const pxPerMm = scaler.pixelsPerMm;       // scale at this frame
 *     const mmWidth = somePixelWidth / pxPerMm; // convert a pixel dimension
 *     const info    = scaler.info();             // provenance + numeric details
 *   }
 *
 *   // In formatSummary, pass the scaler's pixelsPerMm value:
 *   //   formatSummary('Teeth', aggregated, scaler.pixelsPerMm);
 *
 * ==========================================================================
 * MEDIAPIPE LANDMARK INDICES (iris refinement)
 * ==========================================================================
 * When Face Landmarker is created with the default .task bundle (includes iris
 * tracking), landmarks 468-477 are the iris points:
 *
 *   468  left-iris  centre
 *   469  left-iris  right edge
 *   470  left-iris  top edge
 *   471  left-iris  left edge
 *   472  left-iris  bottom edge
 *   473  right-iris centre
 *   474  right-iris right edge
 *   475  right-iris top edge
 *   476  right-iris left edge
 *   477  right-iris bottom edge
 *
 * Horizontal diameter = distance between left-edge and right-edge landmarks.
 */

/** Average adult iris horizontal diameter in mm (Bekerman et al. 2014). */
export const IRIS_DIAMETER_MM = 11.7;

/** Average adult inter-commissure width in mm (Farkas 1994). */
export const MOUTH_WIDTH_MM = 50.0;

/** MediaPipe 478-landmark iris indices (iris refinement enabled). */
export const IRIS_LANDMARKS = {
  // Left eye iris points (MediaPipe canonical landmarks 468..472)
  LEFT_CENTER:      468,
  LEFT_RIGHT_EDGE:  469,
  LEFT_TOP_EDGE:    470,
  LEFT_LEFT_EDGE:   471,
  LEFT_BOTTOM_EDGE: 472,

  // Right eye iris points (MediaPipe canonical landmarks 473..477)
  RIGHT_CENTER:     473,
  RIGHT_RIGHT_EDGE: 474,
  RIGHT_TOP_EDGE:   475,
  RIGHT_LEFT_EDGE:  476,
  RIGHT_BOTTOM_EDGE:477,
};

/** Mouth-corner indices (same as FaceLandmarkIndices.js). */
const MOUTH_CORNER_LEFT  = 61;
const MOUTH_CORNER_RIGHT = 291;

// ---------------------------------------------------------------------------

export class IrisScaler {
  /**
   * @param {object}  [opts]
   * @param {number}  [opts.irisDiameterMm=11.7]  assumed iris diameter
   * @param {number}  [opts.mouthWidthMm=50]       assumed inter-commissure width
   * @param {number}  [opts.smoothingAlpha=0.2]    EMA weight (0 = no smoothing)
   * @param {boolean} [opts.preferIris=true]       use iris over mouth when both available
   */
  constructor({
    irisDiameterMm  = IRIS_DIAMETER_MM,
    mouthWidthMm    = MOUTH_WIDTH_MM,
    smoothingAlpha  = 0.2,
    preferIris      = true,
  } = {}) {
    this.irisDiameterMm = irisDiameterMm;
    this.mouthWidthMm   = mouthWidthMm;
    this.smoothingAlpha = smoothingAlpha;
    this.preferIris     = preferIris;

    /** Smoothed pixels-per-mm estimate (null until first valid update). */
    this.pixelsPerMm = null;

    this._source         = null;  // 'iris' | 'mouth_width' | null
    this._rawPixelsPerMm = null;  // unsmoothed value from last frame
    this._leftIrisPx     = null;  // left  iris pixel diameter, last frame
    this._rightIrisPx    = null;  // right iris pixel diameter, last frame
    this._mouthWidthPx   = null;  // pixel inter-commissure width, last frame

    /** Visual geometry for on-canvas AR overlays */
    this.irises          = null;  // { left, right } or null
    this.mouthRef        = null;  // { left, right, widthPx, refMm } or null
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Update the scale estimate from this frame's MediaPipe landmarks.
   *
   * @param {Array}  landmarks   Raw landmark array (478 elements, {x,y,z} in 0-1).
   * @param {number} frameWidth  Video frame width in pixels.
   * @param {number} frameHeight Video frame height in pixels.
   * @returns {boolean} true if a valid estimate was obtained this frame.
   */
  update(landmarks, frameWidth, frameHeight) {
    if (!landmarks || !landmarks.length) return false;

    const px = (lm) => ({ x: lm.x * frameWidth, y: lm.y * frameHeight });

    this.irises = null;
    this.mouthRef = null;

    // --- 1. Iris scale (primary) ------------------------------------------
    let irisPpMm = null;
    if (landmarks.length >= 478) {
      const buildIris = (centerIdx, leftEdgeIdx, rightEdgeIdx, topEdgeIdx, bottomEdgeIdx, side) => {
        const c = px(landmarks[centerIdx]);
        const l = px(landmarks[leftEdgeIdx]);
        const r = px(landmarks[rightEdgeIdx]);
        const t = px(landmarks[topEdgeIdx]);
        const b = px(landmarks[bottomEdgeIdx]);

        const diamH = Math.hypot(r.x - l.x, r.y - l.y);
        const diamV = Math.hypot(b.x - t.x, b.y - t.y);
        const diam = (diamH > 0 && diamV > 0) ? (diamH + diamV) / 2 : (diamH || diamV);
        if (diam <= 4) return null;

        const radius = diam / 2;
        return {
          side,
          center: c,
          radius,
          diamPx: diam,
          diamH,
          diamV,
          box: {
            x: c.x - radius,
            y: c.y - radius,
            width: diam,
            height: diam,
          },
          points: { center: c, left: l, right: r, top: t, bottom: b },
        };
      };

      const leftIris = buildIris(
        IRIS_LANDMARKS.LEFT_CENTER,
        IRIS_LANDMARKS.LEFT_LEFT_EDGE,
        IRIS_LANDMARKS.LEFT_RIGHT_EDGE,
        IRIS_LANDMARKS.LEFT_TOP_EDGE,
        IRIS_LANDMARKS.LEFT_BOTTOM_EDGE,
        'left',
      );
      const rightIris = buildIris(
        IRIS_LANDMARKS.RIGHT_CENTER,
        IRIS_LANDMARKS.RIGHT_LEFT_EDGE,
        IRIS_LANDMARKS.RIGHT_RIGHT_EDGE,
        IRIS_LANDMARKS.RIGHT_TOP_EDGE,
        IRIS_LANDMARKS.RIGHT_BOTTOM_EDGE,
        'right',
      );

      this._leftIrisPx  = leftIris?.diamPx  ?? null;
      this._rightIrisPx = rightIris?.diamPx ?? null;

      if (leftIris || rightIris) {
        this.irises = { left: leftIris, right: rightIris };
        const valid = [this._leftIrisPx, this._rightIrisPx].filter(Boolean);
        const meanDiamPx = valid.reduce((s, v) => s + v, 0) / valid.length;
        irisPpMm = meanDiamPx / this.irisDiameterMm;
      }
    }

    // --- 2. Mouth-width scale (fallback) ----------------------------------
    let mouthPpMm = null;
    const ml = landmarks[MOUTH_CORNER_LEFT];
    const mr = landmarks[MOUTH_CORNER_RIGHT];
    if (ml && mr) {
      const mlPx = px(ml);
      const mrPx = px(mr);
      const mwPx = Math.hypot(mrPx.x - mlPx.x, mrPx.y - mlPx.y);
      this._mouthWidthPx = mwPx > 10 ? mwPx : null;
      if (this._mouthWidthPx) {
        mouthPpMm = this._mouthWidthPx / this.mouthWidthMm;
        this.mouthRef = {
          left: mlPx,
          right: mrPx,
          widthPx: mwPx,
          refMm: this.mouthWidthMm,
        };
      }
    }

    // --- 3. Choose the best available estimate ----------------------------
    let raw;
    if (this.preferIris) {
      raw          = irisPpMm  ?? mouthPpMm;
      this._source = irisPpMm  ? 'iris' : (mouthPpMm ? 'mouth_width' : null);
    } else {
      raw          = mouthPpMm ?? irisPpMm;
      this._source = mouthPpMm ? 'mouth_width' : (irisPpMm ? 'iris' : null);
    }

    if (raw == null || !Number.isFinite(raw) || raw <= 0) return false;
    this._rawPixelsPerMm = raw;

    // --- 4. EMA smoothing -------------------------------------------------
    if (this.pixelsPerMm == null || this.smoothingAlpha <= 0) {
      this.pixelsPerMm = raw;
    } else {
      this.pixelsPerMm =
        this.pixelsPerMm * (1 - this.smoothingAlpha) +
        raw              *      this.smoothingAlpha;
    }
    return true;
  }

  /** Returns true once at least one valid estimate has been obtained. */
  isReady() {
    return this.pixelsPerMm != null && this.pixelsPerMm > 0;
  }

  /**
   * Convert a pixel distance to millimetres.
   * Returns null if not yet ready.
   *
   * @param  {number} pixels
   * @returns {number|null}
   */
  toMm(pixels) {
    return this.isReady() ? pixels / this.pixelsPerMm : null;
  }

  /**
   * Compact status object, suitable for the HUD or session metadata logging.
   *
   * @returns {object}
   */
  info() {
    return {
      pixelsPerMm:      this.pixelsPerMm,
      rawPixelsPerMm:   this._rawPixelsPerMm,
      source:           this._source,
      leftIrisDiamPx:   this._leftIrisPx,
      rightIrisDiamPx:  this._rightIrisPx,
      mouthWidthPx:     this._mouthWidthPx,
      mmPerPixel:       this.isReady() ? 1 / this.pixelsPerMm : null,
    };
  }

  /** Reset to initial state (call when the subject changes). */
  reset() {
    this.pixelsPerMm     = null;
    this._source         = null;
    this._rawPixelsPerMm = null;
    this._leftIrisPx     = null;
    this._rightIrisPx    = null;
    this._mouthWidthPx   = null;
    this.irises          = null;
    this.mouthRef        = null;
  }
}
