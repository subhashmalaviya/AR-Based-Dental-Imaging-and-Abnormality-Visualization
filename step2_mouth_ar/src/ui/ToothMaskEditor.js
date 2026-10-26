import { traceLabelContours, contourPath, validateVectorContours, rasterizeVector, flattenContour } from '../eval/vectorBoundary.js';
import { freehandBorder } from '../eval/freehandBorder.js';
import { enhancePixels } from './photoEnhancement.js';
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

// Draw inside the model's boundary pixels in the same crop grid as the editable
// labels. Keep touching model instances separate; never move or dilate labels.
export function drawPredictionOutline(canvas, labels) {
  if (!labels) return;
  const { width: w, height: h } = canvas;
  if (labels.length !== w * h) throw new Error('Prediction dimensions must match the editor');
  const ctx = canvas.getContext('2d'), image = ctx.getImageData(0, 0, w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, id = labels[i];
    if (id > 0 && (x === 0 || x === w - 1 || y === 0 || y === h - 1
      || labels[i - 1] !== id || labels[i + 1] !== id || labels[i - w] !== id || labels[i + w] !== id)) {
      image.data.set([255, 90, 54, 255], i * 4);
    }
  }
  ctx.putImageData(image, 0, 0);
}

export class ToothMaskEditor {
  constructor(root, onChange, onViewChange = () => {}) {
    this.el = id => root.querySelector(`#${id}`); this.onChange = onChange;
    this.svg = typeof document.createElementNS === 'function' ? this.el('toothEditVectors') : null;
    this.vectors = []; this.canvas = this.el('toothEditCanvas'); this.history = []; this.redoHistory = []; this.points = [];
    this.el('toothEditSelect').addEventListener('change', e => { this.handle=null;this.cancelFreehand();this.selected = Number(e.target.value); this.draw(); });
    this.el('toothEditNew').addEventListener('click', () => {
      if (instanceStats(this.labels, this.width).length >= 256) return;
      this.saveUndo(); this.selected = this.nextId++; this.points = []; this.refresh(); this.draw();
    });
    this.el('toothEditDelete').addEventListener('click', () => {
      if (!this.labels) return; this.saveUndo();
      for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === this.selected) this.labels[i] = 0;
      this.vectors = this.vectors.filter(c => c.id !== this.selected); this.points = [];
      this.changed();
    });
    this.el('toothEditUndo').addEventListener('click', () => this.undo());
    this.el('toothEditRedo').addEventListener('click', () => this.redo());
    this.el('toothEditZoom').addEventListener('input', e => { (this.svg ? this.el('toothEditStage') : this.canvas).style.width = `${Number(e.target.value) * 100}%`; this.draw(); });
    for (const id of ['toothEditOpacity', 'toothEditBackground', 'toothEditGrayscale', 'toothEditContrast']) this.el(id).addEventListener('input', () => {
      this.draw(); onViewChange();
    });
    this.el('toothEditPrediction').addEventListener('change', () => this.draw());
    this.el('toothEditTool').addEventListener('change', () => { this.handle=null; this.cancelFreehand(); this.points = []; this.draw(); });
    this.el('toothEditCancel').addEventListener('click', () => { this.handle=null; this.cancelFreehand(); this.points = []; this.draw(); });
    this.el('toothEditFill').addEventListener('click', () => this.fillPolygon());
    this.el('toothEditVectorize').addEventListener('click', () => {
      if (!this.labels || this.disabled) return;
      const loops=this.getContours().filter(c=>c.id===this.selected);
      if (loops.length!==1) {
        this.el('toothEditPredictionNote').textContent='Select a tooth with one closed border, or draw a new smooth border. Disconnected regions and holes stay as pixel borders.';
        return;
      }
      const points=flattenContour(loops[0]), lengths=points.map((p,i)=>Math.hypot(p[0]-points[(i+1)%points.length][0],p[1]-points[(i+1)%points.length][1]));
      const total=lengths.reduce((a,b)=>a+b,0),count=Math.max(8,Math.min(64,Math.ceil(total/8))),knots=[];
      for(let k=0;k<count;k++) {
        let distance=k*total/count, i=0;
        while(i<lengths.length-1&&distance>lengths[i]) distance-=lengths[i++];
        const t=lengths[i]?distance/lengths[i]:0,a=points[i],b=points[(i+1)%points.length];
        knots.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);
      }
      this.saveUndo();this.applyVector({id:this.selected,smooth:true,points:knots});
      this.el('toothEditTool').value='vector-edit';this.changed();
    });
    this.canvas.addEventListener('pointerdown', e => {
      if (!this.labels || this.disabled || this.freehand != null || this.handle || this.drawing || (e.button != null && e.button !== 0)) return;
      const p = this.point(e), tool = this.el('toothEditTool').value;
      if (tool === 'pick') {
        const id = this.labels[Math.floor(p[1]) * this.width + Math.floor(p[0])];
        if (id > 0) { this.selected = id; this.refresh(); this.draw(); } return;
      }
      if (tool === 'curve') {
        e.preventDefault(); this.canvas.focus?.({preventScroll:true}); this.freehand = e.pointerId; this.points = [p];
        this.canvas.setPointerCapture(e.pointerId); this.drawVectors(); return;
      }
      if (tool === 'polygon') { this.points.push(p); this.draw(); return; }
      if (tool === 'vector-edit') {
        const scale = this.canvas.getBoundingClientRect().width / this.width;
        let best = 10 / scale;
        for (const c of this.vectors.filter(c => c.id === this.selected)) c.points.forEach((q,i) => {
          const d = Math.hypot(p[0]-q[0],p[1]-q[1]);
          if (d < best) { best=d; this.handle={c:structuredClone(c),i,pointer:e.pointerId}; }
        });
        if (this.handle) { e.preventDefault(); this.canvas.focus?.({preventScroll:true}); this.canvas.setPointerCapture(e.pointerId); }
        return;
      }
      this.saveUndo(); this.last = p; this.drawing = true;
      this.canvas.setPointerCapture(e.pointerId); this.stroke(p);
    });
    this.canvas.addEventListener('pointermove', e => {
      if (this.freehand != null) {
        if (e.pointerId !== this.freehand) return;
        const samples=e.getCoalescedEvents?.();
        for (const sample of samples?.length ? samples : [e]) this.appendFreehand(this.point(sample));
        this.drawVectors();
      }
      else if (this.handle && e.pointerId===this.handle.pointer) { this.handle.c.points[this.handle.i] = this.point(e); this.drawVectors(); }
      else if (this.drawing) this.stroke(this.point(e));
    });
    const end = e => {
      if (this.freehand != null) {
        if (e.pointerId !== this.freehand) return;
        this.appendFreehand(this.point(e));
        const curve=freehandBorder(this.points,this.selected);
        this.cancelFreehand();
        if(curve){this.saveUndo();this.applyVector(curve);this.changed();}
        else this.draw();
        return;
      }
      if (this.handle) {
        if(e.pointerId!==this.handle.pointer)return;
        const {c,i}=this.handle;c.points[i]=this.point(e);
        const previous=this.vectors.find(v=>v.id===c.id)?.points[i];
        this.handle=null;
        if(!previous || previous.some((v,k)=>v!==c.points[i][k])){this.saveUndo();this.applyVector(c);this.changed();}
        else this.draw();
      } if (this.drawing) { this.drawing = false; this.changed(); } };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', e => {
      if(this.handle){this.handle=null;this.draw();}
      else if(this.freehand != null){this.cancelFreehand();this.draw();} else end(e);
    });
    this.canvas.addEventListener('lostpointercapture', () => { if(this.handle){this.handle=null;this.draw();} if(this.freehand != null){this.cancelFreehand();this.draw();} });
    this.canvas.tabIndex=0;
    this.canvas.addEventListener('keydown', e => { if(e.key==='Escape'){this.handle=null;this.cancelFreehand();this.draw();} });
  }

  appendFreehand(p) {
    const last=this.points.at(-1);
    if(this.points.length<20000 && (!last || Math.hypot(p[0]-last[0],p[1]-last[1])>=.25)) this.points.push(p);
  }

  cancelFreehand() {
    const pointer=this.freehand; this.freehand=null;this.points=[];
    if(pointer!=null && this.canvas.hasPointerCapture?.(pointer)) this.canvas.releasePointerCapture(pointer);
  }

  setData(labels, clean, marked) {
    this.cancelFreehand();
    Object.assign(this, { labels, clean, marked, width: clean.width, height: clean.height,
      history: [], redoHistory: [], vectors: [], contourCache: null, points: [], drawing: false, handle: null, prediction: null, blackBackground: null, enhancedView: null });
    const ids = instanceStats(labels, this.width).map(s => s.id);
    this.selected = ids[0] ?? 1; this.nextId = ids.reduce((max, id) => Math.max(max, id), 1) + 1;
    this.refresh(); this.draw();
  }

  point(e) {
    const r = this.canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(this.width - 1, (e.clientX - r.left) * this.width / r.width)),
      Math.max(0, Math.min(this.height - 1, (e.clientY - r.top) * this.height / r.height))];
  }

  snapshot() {
    return {labels:this.labels.slice(),vectors:structuredClone(this.vectors??[]),selected:this.selected,nextId:this.nextId};
  }

  restoreState(state) {
    this.cancelFreehand();this.handle=null;this.drawing=false;
    Object.assign(this,state);this.changed();
  }

  undo() {
    if(this.disabled || !this.history.length || this.drawing || this.handle || this.freehand!=null)return;
    (this.redoHistory??=[]).push(this.snapshot());this.restoreState(this.history.pop());
  }

  redo() {
    if(this.disabled || !this.redoHistory?.length || this.drawing || this.handle || this.freehand!=null)return;
    this.history.push(this.snapshot());this.restoreState(this.redoHistory.pop());
  }

  saveUndo() {
    this.redoHistory=[];
    this.history.push(this.snapshot());
    const max = Math.max(1, Math.min(20, Math.floor(64_000_000 / this.labels.byteLength)));
    while (this.history.length > max) this.history.shift();
  }

  stroke(p) {
    const id = this.el('toothEditTool').value === 'erase' ? 0 : this.selected;
    const radius = Math.max(1, Math.min(60, Number(this.el('toothEditBrush').value) || 1));
    this.vectors = []; this.contourCache = null;
    paintLabelStroke(this.labels, this.width, this.height, this.last, p, radius, id);
    this.last = p; this.draw();
  }

  fillPolygon() {
    if (this.disabled || this.points.length < 3 || !this.labels) return;
    if (this.el('toothEditTool').value === 'curve') {
      const curve = { id: this.selected, smooth: true, points: this.points.map(p => p.slice()) };
      this.saveUndo(); this.applyVector(curve); this.points = []; this.changed(); return;
    }
    const c = document.createElement('canvas'); c.width = this.width; c.height = this.height;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.beginPath();
    this.points.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](...p)); ctx.closePath(); ctx.fill();
    const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
    this.saveUndo(); this.vectors = [];
    for (let i = 0; i < this.labels.length; i++) if (rgba[i * 4 + 3] >= 128) this.labels[i] = this.selected;
    this.points = []; this.changed();
  }

  changed() { this.contourCache = null; this.refresh(); this.draw(); this.onChange(this.labels); }

  getContours() {
    if (!this.contourCache) this.contourCache = traceLabelContours(this.labels, this.width, this.height);
    const ids = new Set(this.vectors.map(c => c.id));
    return [...this.contourCache.filter(c => !ids.has(c.id)), ...this.vectors];
  }

  restoreVectors(contours) {
    this.vectors = validateVectorContours(contours, this.width, this.height, this.labels);
    // Reconstruct mask pixels from the saved curves, maintaining one geometry.
    for (const c of [...this.vectors]) this.applyVector(c);
    this.contourCache = null; this.draw();
  }

  applyVector(curve) {
    const touched = rasterizeVector(this.labels, this.width, this.height, curve);
    this.vectors=this.vectors.filter(v=>v.id!==curve.id&&!touched.has(v.id));
    if (this.labels.includes(curve.id)) this.vectors.push(curve);
    this.contourCache=null;
  }

  drawVectors() {
    if (!this.svg) return;
    this.svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`); this.svg.replaceChildren();
    const add = (name, attrs) => {
      const e=document.createElementNS('http://www.w3.org/2000/svg',name);
      for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,String(v));
      this.svg.append(e);
    };
    const path=(c,color)=>add('path',{d:contourPath(c),fill:'none',stroke:color,'stroke-width':1,'vector-effect':'non-scaling-stroke'});
    for(const saved of this.getContours()) { const c=this.handle?.c.id===saved.id ? this.handle.c : saved; path(c, `rgb(${toothColor(c.id).join(',')})`); }
    if(this.el('toothEditPrediction').checked && this.prediction) for(const c of this.predictionContours) path(c,'#ff5a36');
    if(this.freehand != null && this.points.length>1) add('polyline',{points:this.points.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#fff','stroke-width':1,'vector-effect':'non-scaling-stroke','stroke-linecap':'round','stroke-linejoin':'round'});
    else if(this.points.length>=3) path({points:this.points,smooth:false},'#fff');
    const scale=this.canvas.getBoundingClientRect().width/this.width || 1;
    const handles=this.el('toothEditTool').value==='vector-edit'
      ? (this.handle ? [this.handle.c] : this.vectors.filter(c=>c.id===this.selected)).flatMap(c=>c.points) : this.freehand != null ? [] : this.points;
    for(const p of handles) add('circle',{cx:p[0],cy:p[1],r:3/scale,fill:'#fff',stroke:'#111','stroke-width':1,'vector-effect':'non-scaling-stroke'});
  }

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
    this.el('toothEditRedo').disabled = this.disabled || !this.redoHistory?.length;
  }

  setPrediction(labels) {
    if (labels !== this.prediction) this.predictionContours = labels ? traceLabelContours(labels, this.width, this.height) : [];
    this.prediction = labels;
    this.el('toothEditPredictionNote').textContent = labels
      ? 'Colored T regions are your editable ground truth. Orange outlines are the model prediction on the same pixels.'
      : 'Run Detect & compare to see the model outline directly over your editable ground truth.';
    this.draw();
  }

  getView() {
    const choice = this.el('toothEditBackground').value;
    let background = choice === 'marked' ? this.marked : this.clean;
    if (choice === 'mask') {
      if (!this.blackBackground) {
        this.blackBackground = document.createElement('canvas');
        this.blackBackground.width = this.width; this.blackBackground.height = this.height;
        const ctx = this.blackBackground.getContext('2d'); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, this.width, this.height);
      }
      background = this.blackBackground;
    }
    const grayscale = this.el('toothEditGrayscale').checked;
    const contrast = Number(this.el('toothEditContrast').value) || 1;
    this.el('toothEditContrastValue').textContent = `${contrast.toFixed(2)}×`;
    if (grayscale || contrast !== 1) {
      const cached = this.enhancedView;
      if (!cached || cached.source !== background || cached.grayscale !== grayscale || cached.contrast !== contrast) {
        const enhanced = document.createElement('canvas'); enhanced.width = this.width; enhanced.height = this.height;
        const ctx = enhanced.getContext('2d'); ctx.drawImage(background, 0, 0);
        const image = ctx.getImageData(0, 0, this.width, this.height);
        image.data.set(enhancePixels(image.data, grayscale, contrast)); ctx.putImageData(image, 0, 0);
        this.enhancedView = { source: background, grayscale, contrast, enhanced };
      }
      background = this.enhancedView.enhanced;
    }
    return { background, opacity: Number(this.el('toothEditOpacity').value) };
  }

  draw() {
    if (!this.labels) return;
    const { background, opacity } = this.getView();
    drawInstances(this.canvas, background, this.labels, 'T', opacity);
    if (!this.svg && this.el('toothEditPrediction').checked) drawPredictionOutline(this.canvas, this.prediction);
    this.drawVectors();
    if (!this.svg && this.points.length) {
      const ctx = this.canvas.getContext('2d'); ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); this.points.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](...p)); ctx.stroke();
      for (const p of this.points) { ctx.beginPath(); ctx.arc(...p, 2, 0, 2 * Math.PI); ctx.fill(); }
    }
  }
}
