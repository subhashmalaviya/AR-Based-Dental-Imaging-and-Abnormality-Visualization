/**
 * AROverlayController.js — draws AR content attached to the mouth anchor.
 *
 * The content is defined **once, in mouth-local coordinates** (see
 * MouthARAnchor for the contract) and re-projected every frame. Nothing is
 * positioned in screen space, which is what makes it stay attached to the
 * mouth as the head moves, approaches, recedes or rotates.
 *
 * Images are drawn as a genuine warped quad, not a scaled-and-rotated sprite:
 * the four corners are projected independently and the image is rasterised as
 * two affine-mapped triangles. That keeps the perspective foreshortening of a
 * turned head, which `drawImage` with a rotation/scale transform cannot do.
 */

const clamp01 = (v) => Math.max(0, Math.min(1, v));

export class AROverlayController {
  constructor(ctx) {
    this.ctx = ctx;
    this.image = null;
    this.mode = 'image';        // 'image' | 'rect'
    this.opacity = 0.85;
    this.visible = true;

    // Placement of the overlay quad in mouth-local units (mouth widths).
    this.placement = {
      cx: 0.0,      // centred on the mouth
      cy: 0.0,
      cz: 0.02,     // a hair in front of the lip plane so it reads as "on" it
      width: 1.5,   // 1.5x the mouth width
      followOpening: true,
    };
  }

  setImage(img) { this.image = img; }
  setMode(mode) { this.mode = mode; }
  setOpacity(o) { this.opacity = clamp01(o); }
  setVisible(v) { this.visible = !!v; }
  setPlacement(p) { Object.assign(this.placement, p); }

  /** The overlay quad's four corners in mouth-local coordinates. */
  localQuad(mouthOpen = 0) {
    const { cx, cy, cz, width, followOpening } = this.placement;
    const aspect = this.image && this.image.height
      ? this.image.width / this.image.height
      : 1;
    const w = width;
    const h = w / aspect;
    // Drift the quad down a touch as the jaw drops, so it keeps covering the
    // mouth region rather than riding up the upper lip.
    const yOff = followOpening ? mouthOpen * 0.25 : 0;
    const x0 = cx - w / 2, x1 = cx + w / 2;
    const y0 = cy - h / 2 + yOff, y1 = cy + h / 2 + yOff;
    return [
      { x: x0, y: y0, z: cz },   // top-left
      { x: x1, y: y0, z: cz },   // top-right
      { x: x1, y: y1, z: cz },   // bottom-right
      { x: x0, y: y1, z: cz },   // bottom-left
    ];
  }

  render(anchor) {
    if (!this.visible) return;
    const pose = anchor.getPose();
    if (!pose) return;

    const quad = this.localQuad(pose.mouthOpen ?? 0);
    const pts = quad.map((p) => anchor.localToScreen(p));
    if (pts.some((p) => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) return;

    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = this.opacity;

    if (this.mode === 'image' && this.image && this.image.complete) {
      this._drawWarpedImage(pts);
    } else {
      this._drawRect(pts);
    }

    ctx.restore();
    this._drawOutline(pts);
  }

  _drawRect(pts) {
    const ctx = this.ctx;
    const g = ctx.createLinearGradient(pts[0].x, pts[0].y, pts[2].x, pts[2].y);
    g.addColorStop(0, 'rgba(0, 229, 255, 0.55)');
    g.addColorStop(1, 'rgba(124, 77, 255, 0.55)');
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();
  }

  /**
   * Texture-map the image onto the projected quad as two affine triangles.
   *
   * Canvas2D has no perspective transform, but an affine map per triangle is
   * an accurate piecewise approximation over a region this small, and unlike a
   * single rotate+scale it reproduces the shear/foreshortening of a turned
   * head.
   */
  _drawWarpedImage(pts) {
    const img = this.image;
    const W = img.width, H = img.height;
    const uv = [{ u: 0, v: 0 }, { u: W, v: 0 }, { u: W, v: H }, { u: 0, v: H }];
    this._triangle(pts[0], pts[1], pts[2], uv[0], uv[1], uv[2]);
    this._triangle(pts[0], pts[2], pts[3], uv[0], uv[2], uv[3]);
  }

  _triangle(p0, p1, p2, t0, t1, t2) {
    const ctx = this.ctx;
    const denom = (t1.u - t0.u) * (t2.v - t0.v) - (t2.u - t0.u) * (t1.v - t0.v);
    if (Math.abs(denom) < 1e-9) return;

    const a = ((p1.x - p0.x) * (t2.v - t0.v) - (p2.x - p0.x) * (t1.v - t0.v)) / denom;
    const c = ((p2.x - p0.x) * (t1.u - t0.u) - (p1.x - p0.x) * (t2.u - t0.u)) / denom;
    const b = ((p1.y - p0.y) * (t2.v - t0.v) - (p2.y - p0.y) * (t1.v - t0.v)) / denom;
    const d = ((p2.y - p0.y) * (t1.u - t0.u) - (p1.y - p0.y) * (t2.u - t0.u)) / denom;
    const e = p0.x - a * t0.u - c * t0.v;
    const f = p0.y - b * t0.u - d * t0.v;

    ctx.save();
    ctx.beginPath();
    // Expand the clip a fraction of a pixel outwards from the centroid,
    // otherwise antialiasing leaves a visible hairline along the shared edge.
    const gx = (p0.x + p1.x + p2.x) / 3, gy = (p0.y + p1.y + p2.y) / 3;
    const grow = (p) => {
      const dx = p.x - gx, dy = p.y - gy;
      const len = Math.hypot(dx, dy) || 1;
      return { x: p.x + (dx / len) * 0.6, y: p.y + (dy / len) * 0.6 };
    };
    const q0 = grow(p0), q1 = grow(p1), q2 = grow(p2);
    ctx.moveTo(q0.x, q0.y);
    ctx.lineTo(q1.x, q1.y);
    ctx.lineTo(q2.x, q2.y);
    ctx.closePath();
    ctx.clip();
    ctx.transform(a, b, c, d, e, f);
    ctx.drawImage(this.image, 0, 0);
    ctx.restore();
  }

  _drawOutline(pts) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
}
