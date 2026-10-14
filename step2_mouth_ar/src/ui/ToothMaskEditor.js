import { instanceStats, paintLabelStroke } from '../eval/toothInstances.js';

export function toothColor(id) {
  const h = (id * 137.508) % 360 / 60, c = 190, x = c * (1 - Math.abs(h % 2 - 1));
  const rgb = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(h)];
  return rgb.map(v => Math.round(v + 55));
}

export function drawInstances(canvas, background, labels, prefix = 'T', opacity = 0.55) {
  canvas.width = background.width; canvas.height = background.height;
  const ctx = canvas.getContext('2d'); ctx.drawImage(background, 0, 0);
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height), colors = new Map();
  for (let i = 0; i < labels.length; i++) if (labels[i] > 0) {
    const color = colors.get(labels[i]) ?? toothColor(labels[i]); colors.set(labels[i], color);
    for (let k = 0; k < 3; k++) image.data[4 * i + k] = (1 - opacity) * image.data[4 * i + k] + opacity * color[k];
  }
  ctx.putImageData(image, 0, 0);
  ctx.font = `bold ${Math.max(10, Math.round(canvas.width / 65))}px system-ui`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineWidth = 3;
  for (const tooth of instanceStats(labels, canvas.width)) {
    ctx.strokeStyle = '#111'; ctx.strokeText(`${prefix}${tooth.id}`, tooth.x, tooth.y);
    ctx.fillStyle = '#fff'; ctx.fillText(`${prefix}${tooth.id}`, tooth.x, tooth.y);
  }
}

export class ToothMaskEditor {
  constructor(root, onChange) {
    this.el = id => root.querySelector(`#${id}`); this.onChange = onChange;
    this.canvas = this.el('toothEditCanvas'); this.history = []; this.points = [];
    this.el('toothEditSelect').addEventListener('change', e => { this.selected = Number(e.target.value); });
    this.el('toothEditNew').addEventListener('click', () => {
      if (instanceStats(this.labels, this.width).length >= 256) return;
      this.selected = this.nextId++; this.points = []; this.refresh(); this.draw();
    });
    this.el('toothEditDelete').addEventListener('click', () => {
      if (!this.labels) return; this.saveUndo();
      for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === this.selected) this.labels[i] = 0;
      this.points = [];
      this.changed();
    });
    this.el('toothEditUndo').addEventListener('click', () => {
      if (!this.history.length) return;
      this.labels = this.history.pop(); this.points = []; this.changed();
    });
    this.el('toothEditZoom').addEventListener('input', e => { this.canvas.style.width = `${Number(e.target.value) * 100}%`; });
    for (const id of ['toothEditOpacity', 'toothEditBackground']) this.el(id).addEventListener('input', () => this.draw());
    this.el('toothEditTool').addEventListener('change', () => { this.points = []; this.draw(); });
    this.el('toothEditCancel').addEventListener('click', () => { this.points = []; this.draw(); });
    this.el('toothEditFill').addEventListener('click', () => this.fillPolygon());
    this.canvas.addEventListener('pointerdown', e => {
      if (!this.labels || this.disabled) return;
      const p = this.point(e), tool = this.el('toothEditTool').value;
      if (tool === 'pick') {
        const id = this.labels[Math.floor(p[1]) * this.width + Math.floor(p[0])];
        if (id > 0) { this.selected = id; this.refresh(); } return;
      }
      if (tool === 'polygon') { this.points.push(p); this.draw(); return; }
      this.saveUndo(); this.last = p; this.drawing = true;
      this.canvas.setPointerCapture(e.pointerId); this.stroke(p);
    });
    this.canvas.addEventListener('pointermove', e => { if (this.drawing) this.stroke(this.point(e)); });
    const end = () => { if (this.drawing) { this.drawing = false; this.changed(); } };
    this.canvas.addEventListener('pointerup', end); this.canvas.addEventListener('pointercancel', end);
  }

  setData(labels, clean, marked) {
    Object.assign(this, { labels, clean, marked, width: clean.width, height: clean.height,
      history: [], points: [], drawing: false });
    const ids = instanceStats(labels, this.width).map(s => s.id);
    this.selected = ids[0] ?? 1; this.nextId = ids.reduce((max, id) => Math.max(max, id), 1) + 1;
    this.refresh(); this.draw();
  }

  point(e) {
    const r = this.canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(this.width - 1, (e.clientX - r.left) * this.width / r.width)),
      Math.max(0, Math.min(this.height - 1, (e.clientY - r.top) * this.height / r.height))];
  }

  saveUndo() {
    this.history.push(this.labels.slice());
    const max = Math.max(1, Math.min(20, Math.floor(64_000_000 / this.labels.byteLength)));
    while (this.history.length > max) this.history.shift();
  }

  stroke(p) {
    const id = this.el('toothEditTool').value === 'erase' ? 0 : this.selected;
    const radius = Math.max(1, Math.min(60, Number(this.el('toothEditBrush').value) || 1));
    paintLabelStroke(this.labels, this.width, this.height, this.last, p, radius, id);
    this.last = p; this.draw();
  }

  fillPolygon() {
    if (this.disabled || this.points.length < 3 || !this.labels) return;
    const c = document.createElement('canvas'); c.width = this.width; c.height = this.height;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.beginPath();
    this.points.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](...p)); ctx.closePath(); ctx.fill();
    const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
    this.saveUndo();
    for (let i = 0; i < this.labels.length; i++) if (rgba[i * 4 + 3] >= 128) this.labels[i] = this.selected;
    this.points = []; this.changed();
  }

  changed() { this.refresh(); this.draw(); this.onChange(this.labels); }

  refresh() {
    const teeth = instanceStats(this.labels, this.width), select = this.el('toothEditSelect');
    select.replaceChildren();
    const ids = teeth.map(t => t.id); if (!ids.includes(this.selected)) ids.push(this.selected);
    for (const id of ids.sort((a, b) => a - b)) {
      const option = document.createElement('option'); option.value = id; option.textContent = `Tooth T${id}${teeth.some(t => t.id === id) ? '' : ' (empty — paint to add)'}`;
      select.append(option);
    }
    select.value = this.selected;
    this.el('toothEditCount').textContent = `${teeth.length} ground-truth teeth. IDs are annotation labels, not anatomical tooth numbers.`;
    this.el('toothEditUndo').disabled = this.disabled || !this.history.length;
  }

  draw() {
    if (!this.labels) return;
    const choice = this.el('toothEditBackground').value;
    let background = choice === 'marked' ? this.marked : this.clean;
    if (choice === 'mask') {
      background = document.createElement('canvas'); background.width = this.width; background.height = this.height;
      const ctx = background.getContext('2d'); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, this.width, this.height);
    }
    drawInstances(this.canvas, background, this.labels, 'T', Number(this.el('toothEditOpacity').value));
    if (this.points.length) {
      const ctx = this.canvas.getContext('2d'); ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); this.points.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](...p)); ctx.stroke();
      for (const p of this.points) { ctx.beginPath(); ctx.arc(...p, 2, 0, 2 * Math.PI); ctx.fill(); }
    }
  }
}
