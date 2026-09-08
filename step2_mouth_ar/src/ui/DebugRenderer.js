/**
 * DebugRenderer.js — diagnostic views for the Step-3 pipeline.
 *
 * Draws the mouth ROI quad (so you can see the region actually being searched)
 * and, optionally, a picture-in-picture of the rectified ROI with the raw
 * segmentation mask. Being able to see the rectified view is what makes the
 * segmentation debuggable on a phone, where no console is at hand.
 */

export class DebugRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.show = { roi: false, rectified: false };
    this._pip = document.createElement('canvas');
  }

  setShow(partial) { Object.assign(this.show, partial); }

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
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(120, 220, 255, 0.95)';
    ctx.fillText('mouth ROI', pts[0].x + 4, pts[0].y - 5);
    ctx.restore();
  }

  /** Picture-in-picture of the rectified ROI, bottom-left of the canvas. */
  drawRectifiedPiP(imageData, canvasW, canvasH) {
    if (!this.show.rectified || !imageData) return;
    const ctx = this.ctx;
    const w = imageData.width, h = imageData.height;
    this._pip.width = w;
    this._pip.height = h;
    this._pip.getContext('2d').putImageData(imageData, 0, 0);

    const scale = Math.min(canvasW * 0.28 / w, canvasH * 0.28 / h);
    const dw = w * scale, dh = h * scale;
    const x = 12, y = canvasH - dh - 12;

    ctx.save();
    ctx.globalAlpha = 0.95;
    // Undo any mirroring so the PiP reads the same way the maths sees it.
    ctx.drawImage(this._pip, x, y, dw, dh);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.4;
    ctx.strokeRect(x, y, dw, dh);
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fillText('rectified ROI', x + 4, y - 4);
    ctx.restore();
  }
}
