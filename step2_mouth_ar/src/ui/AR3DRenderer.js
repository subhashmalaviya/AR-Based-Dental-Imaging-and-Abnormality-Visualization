/**
 * AR3DRenderer.js — draws 3D debug proxies attached to each tooth anchor.
 *
 * Every vertex is defined in **tooth-local metres**, pushed through that
 * tooth's own 4x4 transform into camera space, and projected with a pinhole
 * camera. Nothing is a scaled 2D sprite: a box drawn here foreshortens and
 * shows a different set of faces as the head turns, because it is genuinely
 * being projected rather than resized.
 *
 * Proxies are placeholders for real anatomy. Step 4 replaces the box with a
 * tooth mesh, root/nerve geometry or a lesion volume, attached to the same
 * anchor via `Tooth3DAnchor.attach()`; this renderer's job does not change.
 */
import { withUnmirrored } from './DebugRenderer.js';

const PALETTE = [
  '#7cf6b0', '#6fd2ff', '#ffd166', '#ff9ecd', '#c4a7ff',
  '#9be36b', '#ffb27a', '#8fe3d4', '#f5a3a3', '#b0c8ff',
];
const colorFor = (id) => PALETTE[id % PALETTE.length];

// Unit cube corners and edges, in tooth-local space (scaled per tooth).
const CUBE = [
  [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
  [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
];
const CUBE_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
// The outward (labial) face — highlighted so you can see the tooth's facing
// change as the head rotates, which is the visible proof of 3D orientation.
const CUBE_FRONT = [4, 5, 6, 7];

export class AR3DRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.mode = 'box';        // 'box' | 'axes' | 'sphere'
    this.visible = false;
    this.mirrored = false;
    this.showLabels = true;
  }

  setMode(m) { this.mode = m; }
  setVisible(v) { this.visible = !!v; }
  setMirrored(m) { this.mirrored = !!m; }

  /**
   * @param {Tooth3DAnchor[]} anchors
   * @param {object} intr camera intrinsics
   * @param {number} canvasW @param {number} canvasH
   */
  render(anchors, intr, canvasW, canvasH) {
    if (!this.visible || !anchors?.length || !intr) return;
    const ctx = this.ctx;

    // Painter's algorithm: draw far teeth first so nearer ones occlude them.
    // Sorting by real camera-space depth is only meaningful because these are
    // actual 3D positions.
    const sorted = [...anchors].sort((a, b) => b.position.z - a.position.z);

    for (const a of sorted) {
      const color = colorFor(a.id);
      if (this.mode === 'axes') this._axes(a, intr, color);
      else if (this.mode === 'sphere') this._sphere(a, intr, color);
      else this._box(a, intr, color);
    }

    if (this.showLabels) {
      withUnmirrored(ctx, canvasW, this.mirrored, () => {
        ctx.save();
        ctx.font = '600 10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        for (const a of sorted) {
          const o = a.toScreen({ x: 0, y: 0, z: 0 }, intr);
          if (!o) continue;
          // un-mirror the x coordinate to match the flipped context
          const x = this.mirrored ? canvasW - o.x : o.x;
          const zc = (a.position.z * 100).toFixed(1);
          ctx.fillStyle = 'rgba(6,10,16,0.75)';
          const label = `z=${zc}cm`;
          const w = ctx.measureText(label).width + 6;
          ctx.fillRect(x - w / 2, o.y + 10, w, 12);
          ctx.fillStyle = colorFor(a.id);
          ctx.fillText(label, x, o.y + 19);
        }
        ctx.restore();
      });
    }
  }

  _project(anchor, local, intr) {
    return anchor.toScreen(local, intr);
  }

  _box(a, intr, color) {
    const ctx = this.ctx;
    const s = a.scale;
    const pts = CUBE.map(([x, y, z]) =>
      this._project(a, { x: x * s.x, y: y * s.y, z: z * s.z }, intr));
    if (pts.some((p) => !p)) return;

    ctx.save();
    // filled outward face, so orientation is legible at a glance
    ctx.beginPath();
    ctx.moveTo(pts[CUBE_FRONT[0]].x, pts[CUBE_FRONT[0]].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[CUBE_FRONT[i]].x, pts[CUBE_FRONT[i]].y);
    ctx.closePath();
    ctx.fillStyle = `${color}44`;
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (const [i, j] of CUBE_EDGES) {
      ctx.moveTo(pts[i].x, pts[i].y);
      ctx.lineTo(pts[j].x, pts[j].y);
    }
    ctx.stroke();
    ctx.restore();
  }

  _axes(a, intr, color) {
    const ctx = this.ctx;
    const len = Math.max(a.scale.x, a.scale.y) * 0.9;
    const o = this._project(a, { x: 0, y: 0, z: 0 }, intr);
    if (!o) return;
    const axes = [
      [{ x: len, y: 0, z: 0 }, '#ff5d5d'],
      [{ x: 0, y: len, z: 0 }, '#5dff9b'],
      [{ x: 0, y: 0, z: len }, '#5db4ff'],
    ];
    ctx.save();
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    for (const [vec, c] of axes) {
      const p = this._project(a, vec, intr);
      if (!p) continue;
      ctx.beginPath();
      ctx.strokeStyle = c;
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(o.x, o.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  _sphere(a, intr, color) {
    const ctx = this.ctx;
    const r = Math.min(a.scale.x, a.scale.y) * 0.45;
    const o = this._project(a, { x: 0, y: 0, z: 0 }, intr);
    const edge = this._project(a, { x: r, y: 0, z: 0 }, intr);
    if (!o || !edge) return;
    const rPx = Math.max(2, Math.hypot(edge.x - o.x, edge.y - o.y));
    ctx.save();
    // Radius shrinks with distance because it is derived from a projected
    // 3D offset, not from a fixed pixel size.
    const g = ctx.createRadialGradient(o.x - rPx * 0.3, o.y - rPx * 0.3, rPx * 0.1, o.x, o.y, rPx);
    g.addColorStop(0, `${color}dd`);
    g.addColorStop(1, `${color}22`);
    ctx.beginPath();
    ctx.arc(o.x, o.y, rPx, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();
  }
}
