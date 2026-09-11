/**
 * DebugRenderer.js — diagnostic views for the Step-3 tooth pipeline.
 *
 * The point of this module is auditability: when tooth detection looks
 * unconvincing on a live camera, you need to see *why*. A tooth overlay drawn
 * on the face tells you nothing about whether the segmenter found any enamel
 * at all, or whether the mouth was simply too dark for there to be anything to
 * find. So this renders the segmenter's actual intermediate stages —
 * rectified ROI, whiteness-threshold mask, per-arch masks, per-tooth
 * contours — blown up large enough to read on a phone.
 *
 * Nothing here feeds back into tracking; it is purely a view.
 */

const ARCH_UPPER = [124, 240, 176];   // green
const ARCH_LOWER = [255, 178, 122];   // orange
const CANDIDATE = [90, 150, 255];     // blue
// learned-model view
const TEETH_P = [80, 230, 255];      // cyan   — P(teeth)
const BOUNDARY = [255, 60, 220];     // magenta — interdental boundary
const CENTER = [255, 230, 60];       // yellow — tooth-centre peaks

/**
 * Run `fn` with the mirror undone.
 *
 * The preview canvas carries `transform: scaleX(-1)` for the front camera, so
 * anything drawn into it comes out reversed — which renders every debug panel
 * and text label backwards and effectively illegible. Flipping the coordinate
 * system a second time cancels the CSS flip, so inside `fn` you draw using the
 * coordinates you actually want on screen and the glyphs read correctly.
 */
export function withUnmirrored(ctx, canvasW, mirrored, fn) {
  if (!mirrored) return fn();
  ctx.save();
  ctx.translate(canvasW, 0);
  ctx.scale(-1, 1);
  fn();
  ctx.restore();
}

export class DebugRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.show = { roi: false, rectified: false, masks: false, stats: false };
    this.mirrored = false;
    this._pip = document.createElement('canvas');
    this._mask = document.createElement('canvas');
  }

  setShow(partial) { Object.assign(this.show, partial); }
  setMirrored(m) { this.mirrored = !!m; }

  /** Outline of the mouth ROI, drawn in mouth-local space so it tracks. */
  drawROI(roi, anchor) {
    if (!this.show.roi || !roi?.bounds || !anchor?.isValid()) return;
    const { u0, u1, v0, v1 } = roi.bounds;
    const pts = [
      { x: u0, y: v0 }, { x: u1, y: v0 }, { x: u1, y: v1 }, { x: u0, y: v1 },
    ].map((p) => anchor.localToScreen({ x: p.x, y: p.y, z: 0 })).filter(Boolean);
    if (pts.length < 4) return;

    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = 'rgba(120, 220, 255, 0.9)';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  /**
   * The segmenter's own view, as a picture-in-picture panel: the rectified
   * mouth, then the same region with the threshold/arch masks painted over it.
   *
   * This is the single most useful debug view — if the right-hand panel is
   * black, the segmenter found no enamel, and the problem is lighting or mouth
   * opening, not the tracker.
   */
  drawSegmenterView(roiImage, dbg, tracks, roi, canvasW, canvasH) {
    if (!this.show.rectified || !roiImage) return;
    withUnmirrored(this.ctx, canvasW, this.mirrored,
      () => this._segmenterView(roiImage, dbg, tracks, roi, canvasW, canvasH));
  }

  _segmenterView(roiImage, dbg, tracks, roi, canvasW, canvasH) {
    const ctx = this.ctx;
    const W = roiImage.width, H = roiImage.height;

    // left: rectified ROI as the camera sees it
    this._pip.width = W; this._pip.height = H;
    this._pip.getContext('2d').putImageData(roiImage, 0, 0);

    // right: same ROI with the masks painted on
    this._mask.width = W; this._mask.height = H;
    const mctx = this._mask.getContext('2d');
    const overlay = mctx.createImageData(W, H);
    const src = roiImage.data;
    const o = overlay.data;
    for (let i = 0; i < W * H; i++) {
      // dim the underlying image so the masks read clearly
      let r = src[i * 4] * 0.35, g = src[i * 4 + 1] * 0.35, b = src[i * 4 + 2] * 0.35;
      if (dbg?.maps) {
        // learned model: teeth probability (cyan), interdental boundary
        // (magenta), tooth-centre peaks (yellow) — exactly what it predicted
        const pt = dbg.maps.teeth[i], pb = dbg.maps.boundary[i], pc = dbg.maps.center[i];
        r += TEETH_P[0] * 0.6 * pt; g += TEETH_P[1] * 0.6 * pt; b += TEETH_P[2] * 0.6 * pt;
        r += BOUNDARY[0] * pb; g += BOUNDARY[1] * pb; b += BOUNDARY[2] * pb;
        if (pc > 0.3) { r += CENTER[0] * pc; g += CENTER[1] * pc; b += CENTER[2] * pc; }
      } else if (dbg) {
        if (dbg.candidate?.[i]) { r += CANDIDATE[0] * 0.30; g += CANDIDATE[1] * 0.30; b += CANDIDATE[2] * 0.30; }
        if (dbg.upper?.[i]) { r += ARCH_UPPER[0] * 0.55; g += ARCH_UPPER[1] * 0.55; b += ARCH_UPPER[2] * 0.55; }
        if (dbg.lower?.[i]) { r += ARCH_LOWER[0] * 0.55; g += ARCH_LOWER[1] * 0.55; b += ARCH_LOWER[2] * 0.55; }
      }
      o[i * 4] = Math.min(255, r);
      o[i * 4 + 1] = Math.min(255, g);
      o[i * 4 + 2] = Math.min(255, b);
      o[i * 4 + 3] = 255;
    }
    mctx.putImageData(overlay, 0, 0);

    // Scale the pair up as large as sensibly fits — small teeth are the whole
    // problem, so the debug panel must not be small too.
    const target = Math.min(canvasW * 0.46, 520);
    const scale = target / W;
    const dw = W * scale, dh = H * scale;
    const pad = 10;
    const x0 = pad, y0 = canvasH - dh - pad;

    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 0.97;
    ctx.drawImage(this._pip, x0, y0, dw, dh);
    ctx.drawImage(this._mask, x0 + dw + 6, y0, dw, dh);
    ctx.globalAlpha = 1;

    // per-tooth contours drawn into the right-hand (mask) panel
    if (tracks?.length && roi) {
      ctx.save();
      ctx.translate(x0 + dw + 6, y0);
      ctx.scale(scale, scale);
      ctx.lineWidth = 1.2 / scale;
      for (const t of tracks) {
        const s = t.smoothed ?? t;
        const pts = (s.contour ?? [])
          .map((p) => roi.localToRoi(p.u, p.v)).filter(Boolean);
        if (pts.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.closePath();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.65)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(x0, y0, dw, dh);
    ctx.strokeRect(x0 + dw + 6, y0, dw, dh);

    ctx.font = '600 11px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fillText('rectified ROI (what the detector sees)', x0, y0 - 5);
    ctx.fillText(dbg?.maps ? 'model output + tooth instances' : 'threshold + arches + tooth contours',
      x0 + dw + 6, y0 - 5);

    // legend
    const legend = dbg?.maps ? [
      ['P(teeth) — learned mask', TEETH_P],
      ['interdental boundary (predicted)', BOUNDARY],
      ['tooth-centre peaks (predicted)', CENTER],
    ] : [
      ['candidate (whiteness > threshold)', CANDIDATE],
      ['upper arch', ARCH_UPPER],
      ['lower arch', ARCH_LOWER],
    ];
    let ly = y0 + dh + 14;
    ctx.font = '500 10px system-ui, sans-serif';
    for (const [label, col] of legend) {
      ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.fillRect(x0, ly - 7, 9, 9);
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fillText(label, x0 + 14, ly);
      ly += 13;
    }
    ctx.restore();
  }

  /** Numeric readout of the segmenter's internal state. */
  drawStats(dbg, stats, timing, detectFps, canvasW) {
    if (!this.show.stats) return;
    withUnmirrored(this.ctx, canvasW, this.mirrored,
      () => this._stats(dbg, stats, timing, detectFps, canvasW));
  }

  _stats(dbg, stats, timing, detectFps, canvasW) {
    const ctx = this.ctx;
    const lines = [
      `teeth detected   ${stats?.count ?? 0}`,
      `avg confidence   ${stats?.count ? stats.avgConfidence.toFixed(3) : '—'}`,
      `tracking         ${stats?.status ?? '—'}  (${stats?.stable ?? 0} stable)`,
      `inference        ${timing?.detect?.toFixed(2) ?? '—'} ms`,
      `detection FPS    ${detectFps ? detectFps.toFixed(1) : '—'}`,
      ...(dbg?.maps ? [
        `model            ${dbg.inferenceMs?.toFixed(1) ?? '—'} ms`,
        `decode           ${dbg.decodeMs?.toFixed(1) ?? '—'} ms`,
        `instances        ${dbg.toothCount ?? '—'}`,
        `stability        ${stats?.stability != null ? (stats.stability * 100).toFixed(0) + '%' : '—'}`,
      ] : [
        `aperture px      ${dbg?.aperturePx ?? '—'}`,
        `whiteness thr    ${dbg?.threshold?.toFixed(0) ?? '—'}`,
        `candidate px     ${dbg?.candidatePx ?? '—'}`,
        `arch px          ${dbg?.archPx ?? '—'}`,
      ]),
    ];
    const w = 210, h = lines.length * 15 + 14;
    const x = canvasW - w - 10, y = 10;

    ctx.save();
    ctx.fillStyle = 'rgba(6,10,16,0.80)';
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();
    ctx.font = '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#cfe8ff';
    lines.forEach((l, i) => ctx.fillText(l, x + 10, y + 20 + i * 15));
    ctx.restore();
  }
}
