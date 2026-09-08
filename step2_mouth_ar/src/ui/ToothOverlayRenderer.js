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
  }

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
        ctx.stroke();
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

      if (c && (this.show.ids || this.show.confidence)) {
        const parts = [];
        if (this.show.ids) parts.push(`T${t.id}`);
        if (this.show.confidence) parts.push(s.confidence.toFixed(2));
        const label = parts.join('  ');

        // Labels are placed at a fixed offset *in mouth-local units*, so they
        // scale with the face instead of colliding when the user leans back.
        const up = anchor.localToScreen({
          x: s.center.u,
          y: s.center.v + (t.arch === 'upper' ? -0.085 : 0.085),
          z: 0,
        });
        const lx = up?.x ?? c.x;
        const ly = up?.y ?? c.y;

        ctx.font = '600 11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const w = ctx.measureText(label).width + 8;
        ctx.fillStyle = 'rgba(6,10,16,0.72)';
        ctx.beginPath();
        ctx.roundRect(lx - w / 2, ly - 8, w, 16, 5);
        ctx.fill();
        ctx.fillStyle = selected ? '#ffffff' : color;
        ctx.fillText(label, lx, ly);
      }

      ctx.restore();
    }
  }
}
