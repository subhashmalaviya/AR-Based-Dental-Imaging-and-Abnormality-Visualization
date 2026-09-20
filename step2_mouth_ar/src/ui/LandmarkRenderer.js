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

  /**
   * Draw visual bounding boxes, circular reticles, and scale badges over the
   * tracked irises (or inter-commissure mouth ruler if falling back).
   *
   * @param {import('../core/IrisScaler.js').IrisScaler} scaler
   * @param {boolean} mirrored
   */
  drawIrisScale(scaler, mirrored = false) {
    if (!scaler || !scaler.isReady()) return;
    const ctx = this.ctx;
    ctx.save();

    if (scaler._source === 'iris' && scaler.irises) {
      const list = [scaler.irises.left, scaler.irises.right].filter(Boolean);
      list.forEach((iris, idx) => {
        const { box, center, radius, diamPx } = iris;

        // 1. Semi-transparent bounding box fill
        ctx.fillStyle = 'rgba(0, 245, 212, 0.08)';
        ctx.fillRect(box.x, box.y, box.width, box.height);

        // 2. Corner brackets for a precision medical AR feel
        const cornerLen = Math.max(5, radius * 0.4);
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 1.8;
        ctx.lineCap = 'square';

        // Top-left
        ctx.beginPath();
        ctx.moveTo(box.x, box.y + cornerLen);
        ctx.lineTo(box.x, box.y);
        ctx.lineTo(box.x + cornerLen, box.y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(box.x + box.width - cornerLen, box.y);
        ctx.lineTo(box.x + box.width, box.y);
        ctx.lineTo(box.x + box.width, box.y + cornerLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(box.x, box.y + box.height - cornerLen);
        ctx.lineTo(box.x, box.y + box.height);
        ctx.lineTo(box.x + cornerLen, box.y + box.height);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(box.x + box.width - cornerLen, box.y + box.height);
        ctx.lineTo(box.x + box.width, box.y + box.height);
        ctx.lineTo(box.x + box.width, box.y + box.height - cornerLen);
        ctx.stroke();

        // 3. Circular iris boundary reticle
        ctx.beginPath();
        ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 245, 212, 0.7)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 2]);
        ctx.stroke();
        ctx.setLineDash([]);

        // 4. Center crosshair
        const ch = Math.max(3, radius * 0.25);
        ctx.beginPath();
        ctx.moveTo(center.x - ch, center.y);
        ctx.lineTo(center.x + ch, center.y);
        ctx.moveTo(center.x, center.y - ch);
        ctx.lineTo(center.x, center.y + ch);
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // 5. Un-mirrored label badges
        const labelText = `Iris ⌀${scaler.irisDiameterMm}mm (${diamPx.toFixed(1)}px)`;
        const scaleText = idx === 0 ? `Scale: ${scaler.pixelsPerMm.toFixed(2)} px/mm` : null;

        ctx.save();
        ctx.translate(center.x, box.y - 8);
        if (mirrored) ctx.scale(-1, 1);

        ctx.font = '600 11px system-ui, -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = 'rgba(10, 25, 30, 0.85)';
        ctx.fillRect(-textWidth / 2 - 5, -16, textWidth + 10, 16);
        ctx.strokeStyle = 'rgba(0, 245, 212, 0.6)';
        ctx.lineWidth = 1;
        ctx.strokeRect(-textWidth / 2 - 5, -16, textWidth + 10, 16);

        ctx.fillStyle = '#00f5d4';
        ctx.fillText(labelText, 0, -3);

        if (scaleText) {
          const sWidth = ctx.measureText(scaleText).width;
          ctx.fillStyle = 'rgba(10, 25, 30, 0.85)';
          ctx.fillRect(-sWidth / 2 - 5, -34, sWidth + 10, 16);
          ctx.strokeStyle = 'rgba(255, 225, 77, 0.6)';
          ctx.lineWidth = 1;
          ctx.strokeRect(-sWidth / 2 - 5, -34, sWidth + 10, 16);
          ctx.fillStyle = '#ffe14d';
          ctx.fillText(scaleText, 0, -21);
        }

        ctx.restore();
      });
    } else if (scaler._source === 'mouth_width' && scaler.mouthRef) {
      // Fallback visualization: ruler across mouth corners
      const { left, right, widthPx, refMm } = scaler.mouthRef;
      ctx.beginPath();
      ctx.moveTo(left.x, left.y);
      ctx.lineTo(right.x, right.y);
      ctx.strokeStyle = 'rgba(255, 225, 77, 0.8)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      const midX = (left.x + right.x) / 2;
      const midY = (left.y + right.y) / 2;
      const label = `Mouth Scale Ref: ${refMm}mm (${widthPx.toFixed(1)}px) → ${scaler.pixelsPerMm.toFixed(2)} px/mm`;

      ctx.save();
      ctx.translate(midX, midY + 18);
      if (mirrored) ctx.scale(-1, 1);
      ctx.font = '600 11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(10, 25, 30, 0.85)';
      ctx.fillRect(-tw / 2 - 6, -16, tw + 12, 18);
      ctx.strokeStyle = 'rgba(255, 225, 77, 0.6)';
      ctx.lineWidth = 1;
      ctx.strokeRect(-tw / 2 - 6, -16, tw + 12, 18);
      ctx.fillStyle = '#ffe14d';
      ctx.fillText(label, 0, -2);
      ctx.restore();
    }

    ctx.restore();
  }
}
