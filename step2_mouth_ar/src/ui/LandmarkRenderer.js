/**
 * LandmarkRenderer.js — debug visualisation of the tracking state.
 *
 * Draws the lip outline, the named mouth landmarks, the mouth bounding box and
 * the anchor's coordinate axes. Purely diagnostic: nothing here feeds back into
 * tracking, so all of it can be toggled off without changing behaviour.
 */
import {
  LIPS_INNER_EDGES, LIPS_OUTER_EDGES,
} from '../landmarks/FaceLandmarkIndices.js';

const COLORS = {
  outer: '#00e5ff',
  inner: '#ffe14d',
  corner: '#ff3b6b',
  lipCentre: '#7cff6b',
  bbox: 'rgba(255,255,255,0.55)',
  axisX: '#ff4d4d',
  axisY: '#4dff88',
  axisZ: '#4d9bff',
  anchor: '#ffffff',
};

export class LandmarkRenderer {
  constructor(ctx) { this.ctx = ctx; }

  /** Full face mesh as a faint point cloud (off by default; it is dense). */
  drawAllLandmarks(landmarks, w, h, alpha = 0.35) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = `rgba(160, 220, 255, ${alpha})`;
    for (const p of landmarks) {
      ctx.fillRect(p.x * w - 0.75, p.y * h - 0.75, 1.5, 1.5);
    }
    ctx.restore();
  }

  drawMouth(mouth, landmarks, w, h) {
    if (!mouth) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    const edges = (list, color, width) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      for (const [a, b] of list) {
        ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      }
      ctx.stroke();
    };
    edges(LIPS_OUTER_EDGES, COLORS.outer, 2.2);
    edges(LIPS_INNER_EDGES, COLORS.inner, 1.6);

    const dot = (p, color, r) => {
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
    };
    dot(mouth.corners.left, COLORS.corner, 4.5);
    dot(mouth.corners.right, COLORS.corner, 4.5);
    dot(mouth.lips.upperInner, COLORS.lipCentre, 3.2);
    dot(mouth.lips.lowerInner, COLORS.lipCentre, 3.2);

    const bb = mouth.boundingBox;
    ctx.strokeStyle = COLORS.bbox;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([6, 5]);
    ctx.strokeRect(bb.x, bb.y, bb.width, bb.height);
    ctx.setLineDash([]);
    ctx.restore();
  }

  /**
   * The anchor's own axes, drawn through localToScreen so what you see is
   * literally the coordinate system Step 3 will place content in.
   */
  drawAnchor(anchor) {
    const pose = anchor.getPose();
    if (!pose) return;
    const ctx = this.ctx;
    const O = anchor.localToScreen({ x: 0, y: 0, z: 0 });
    const axes = [
      [{ x: 0.5, y: 0, z: 0 }, COLORS.axisX, 'X'],
      [{ x: 0, y: 0.5, z: 0 }, COLORS.axisY, 'Y'],
      [{ x: 0, y: 0, z: 0.5 }, COLORS.axisZ, 'Z'],
    ];

    ctx.save();
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    for (const [vec, color, label] of axes) {
      const P = anchor.localToScreen(vec);
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.moveTo(O.x, O.y);
      ctx.lineTo(P.x, P.y);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = '600 13px system-ui, sans-serif';
      ctx.fillText(label, P.x + 4, P.y - 4);
    }
    ctx.beginPath();
    ctx.fillStyle = COLORS.anchor;
    ctx.arc(O.x, O.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }
}
