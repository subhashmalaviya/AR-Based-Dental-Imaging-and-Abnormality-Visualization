/**
 * ToothOverlayRenderer.js — AR visualisation of tracked teeth.
 *
 * Every point is drawn through `anchor.localToScreen()`, i.e. from mouth-local
 * coordinates. Nothing is in screen space, so the contours ride the real teeth
 * as the head translates, scales and rolls — the same discipline as the Step-1
 * cup overlay and the Step-2 mouth quad.
 */

const PALETTE = [
  '#7cf6b0', '#6fd2ff', '#ffd166', '#ff9ecd', '#c4a7ff',
  '#9be36b', '#ffb27a', '#8fe3d4', '#f5a3a3', '#b0c8ff',
];

const colorFor = (id) => PALETTE[id % PALETTE.length];

export class ToothOverlayRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.show = {
      contours: true,
      ids: true,
      confidence: true,
      boxes: false,
      centers: true,
    };
    this.mirrored = false;
  }

  setMirrored(m) { this.mirrored = !!m; }

  setShow(partial) { Object.assign(this.show, partial); }

  /**
   * @param {ToothTrack[]} tracks
   * @param {MouthARAnchor} anchor
   * @param {number|null} selectedId
   */
  render(tracks, anchor, selectedId = null) {
    if (!anchor?.isValid() || !tracks?.length) return;
    const ctx = this.ctx;

    for (const t of tracks) {
      const s = t.smoothed ?? t;
      const color = colorFor(t.id);
      const selected = t.id === selectedId;

      const pts = (s.contour ?? [])
        .map((p) => anchor.localToScreen({ x: p.u, y: p.v, z: 0 }))
        .filter(Boolean);

      ctx.save();

      if (this.show.contours && pts.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.closePath();
        ctx.fillStyle = selected ? 'rgba(255,255,255,0.42)' : `${color}33`;
        ctx.fill();
        ctx.strokeStyle = selected ? '#ffffff' : color;
        ctx.lineWidth = selected ? 3 : 1.8;
        ctx.lineJoin = 'round';
        // Partially visible teeth (cut by the lip / frame edge) are dashed.
        if (t.visibility === 'partial') ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (this.show.boxes) {
        const b = s.box;
        const corners = [
          { x: b.u, y: b.v }, { x: b.u + b.w, y: b.v },
          { x: b.u + b.w, y: b.v + b.h }, { x: b.u, y: b.v + b.h },
        ].map((p) => anchor.localToScreen({ x: p.x, y: p.y, z: 0 }));
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
        ctx.closePath();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = `${color}cc`;
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const c = anchor.localToScreen({ x: s.center.u, y: s.center.v, z: 0 });
      if (c && this.show.centers) {
        ctx.beginPath();
        ctx.arc(c.x, c.y, selected ? 4 : 2.6, 0, Math.PI * 2);
        ctx.fillStyle = selected ? '#ffffff' : color;
        ctx.fill();
      }

      ctx.restore();
    }

    // Labels are drawn in a second pass, after every contour, so no contour
    // can paint over a label. They are also *staggered* into rows: at a normal
    // camera distance the mouth is only ~100 px wide, so per-tooth labels at a
    // fixed offset collapse into an unreadable pile (which is exactly what
    // made the live overlay look like it had detected nothing).
    if (this.show.ids || this.show.confidence) {
      this._drawLabels(tracks, anchor, selectedId);
    }
  }

  _drawLabels(tracks, anchor, selectedId) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Order along the arch so the stagger alternates between neighbours.
    const ordered = [...tracks].sort((a, b) => {
      if (a.arch !== b.arch) return a.arch === 'upper' ? -1 : 1;
      return (a.smoothed ?? a).center.u - (b.smoothed ?? b).center.u;
    });

    ordered.forEach((t, i) => {
      const s = t.smoothed ?? t;
      const selected = t.id === selectedId;
      const c = anchor.localToScreen({ x: s.center.u, y: s.center.v, z: 0 });
      if (!c) return;

      const parts = [];
      if (this.show.ids) parts.push(`T${t.id}`);
      if (this.show.confidence) parts.push(s.confidence.toFixed(2));
      const label = parts.join(' ');

      // Two rows per arch, alternating, pushed clear of the lips.
      const rung = (i % 2) ? 0.30 : 0.19;
      const dir = t.arch === 'upper' ? -1 : 1;
      const anchorPt = anchor.localToScreen({
        x: s.center.u, y: s.center.v + dir * rung, z: 0,
      });
      const lx = anchorPt?.x ?? c.x;
      const ly = anchorPt?.y ?? c.y;

      // leader line back to the tooth, so a staggered label is unambiguous
      ctx.beginPath();
      ctx.strokeStyle = selected ? 'rgba(255,255,255,0.9)' : `${colorFor(t.id)}99`;
      ctx.lineWidth = 1;
      ctx.moveTo(c.x, c.y);
      ctx.lineTo(lx, ly);
      ctx.stroke();

      const w = ctx.measureText(label).width + 8;
      // Confidence is colour-coded: a weak detection should *look* weak rather
      // than being presented with the same authority as a strong one.
      const fg = selected ? '#ffffff'
        : (s.confidence >= 0.6 ? colorFor(t.id)
          : s.confidence >= 0.35 ? '#ffd166' : '#ff8a8a');

      // The label's *position* must stay in the mirrored face space, but its
      // glyphs must not be mirrored, so the flip is undone locally about the
      // label's own origin. Without this every ID reads backwards on the front
      // camera, which is most of why the overlay looked like noise.
      ctx.save();
      ctx.translate(lx, ly);
      if (this.mirrored) ctx.scale(-1, 1);
      ctx.fillStyle = 'rgba(6,10,16,0.82)';
      ctx.beginPath();
      ctx.roundRect(-w / 2, -8, w, 16, 5);
      ctx.fill();
      ctx.fillStyle = fg;
      ctx.fillText(label, 0, 0);
      ctx.restore();
    });
    ctx.restore();
  }
}
