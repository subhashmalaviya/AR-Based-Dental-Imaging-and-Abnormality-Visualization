/**
 * evalApp.js — offline ground-truth annotation and accuracy measurement.
 *
 * Loads a recording made by the app (video or captured PNG) together with its
 * per-frame metadata JSON, lets a person mark the teeth that are actually
 * visible, and scores the recorded detections against those marks with
 * src/eval/metrics.js. Everything stays in the browser.
 *
 * Coordinates: metadata and ground truth are both in RAW (unmirrored) video
 * pixels. An annotated recording of the front camera is stored mirrored, so
 * it is drawn flipped back here to line up with the metadata.
 */
import { evaluateFrame, aggregate } from './metrics.js';
import { saveBlob } from '../core/SessionRecorder.js';

const $ = (id) => document.getElementById(id);
const canvas = $('evCanvas');
const ctx = canvas.getContext('2d');
const video = document.createElement('video');
video.muted = true;
video.playsInline = true;
const still = new Image();

const S = {
  meta: null,           // parsed metadata document
  frames: [],           // meta.frames
  idx: 0,
  media: null,          // 'video' | 'image'
  mirroredSource: false,
  gt: {},               // frameIndex -> { teeth:[...], ignore:[...] }
  mode: 'point',
  jaw: 'upper',
  view: null,           // source-pixel rect shown on the canvas
  drag: null,
  history: [],
  sourceName: '',
};

function banner(msg, kind = 'info') {
  const b = $('banner');
  if (!msg) { b.hidden = true; return; }
  b.hidden = false; b.textContent = msg; b.className = `banner banner--${kind}`;
}

// ------------------------------------------------------------ loading
async function loadMeta(file) {
  const doc = JSON.parse(await file.text());
  if (!doc.frames || !doc.header) throw new Error('not a Dental AR metadata file');
  S.meta = doc;
  S.frames = doc.frames;
  S.mirroredSource = !!doc.header.mirrored?.annotatedVideo;
  $('evSlider').max = String(Math.max(0, S.frames.length - 1));
  S.idx = S.frames.findIndex((f) => f.mouth) >= 0 ? S.frames.findIndex((f) => f.mouth) : 0;
  $('evLoadInfo').textContent = `${file.name}: ${S.frames.length} frames, mode ${doc.header.mode}, `
    + `detector ${doc.header.detector?.name ?? '?'}`
    + (doc.header.mode === 'annotated' ? ' — annotated video: shown un-mirrored, burned-in text reads backwards.' : '');
  renderRecStats();
  await showFrame(S.idx);
}

async function loadMedia(file) {
  const url = URL.createObjectURL(file);
  S.sourceName = file.name;
  if (file.type.startsWith('image/')) {
    S.media = 'image';
    await new Promise((res, rej) => { still.onload = res; still.onerror = rej; still.src = url; });
  } else {
    S.media = 'video';
    video.src = url;
    await new Promise((res) => video.addEventListener('loadedmetadata', res, { once: true }));
    // MediaRecorder WebM files often lack a duration, which breaks seeking.
    // Seeking far past the end makes the browser scan and learn it.
    if (!Number.isFinite(video.duration)) {
      video.currentTime = 1e7;
      await new Promise((res) => video.addEventListener('durationchange', res, { once: true }));
      video.currentTime = 0;
    }
  }
  await showFrame(S.idx);
}

async function loadGt(file) {
  const doc = JSON.parse(await file.text());
  S.gt = {};
  for (const [k, v] of Object.entries(doc.frames ?? {})) S.gt[k] = v;
  update();
}

// ------------------------------------------------------------ frames
function seek(t) {
  return new Promise((res) => {
    if (Math.abs(video.currentTime - t) < 1e-3) { res(); return; }
    video.addEventListener('seeked', res, { once: true });
    video.currentTime = t;
  });
}

async function showFrame(i) {
  if (!S.frames.length) { draw(); return; }
  S.idx = Math.max(0, Math.min(S.frames.length - 1, i));
  $('evSlider').value = String(S.idx);
  const f = S.frames[S.idx];
  if (S.media === 'video' && f.t_ms != null) {
    const off = Number($('evOffset').value) || 0;
    await seek(Math.max(0, (f.t_ms + off) / 1000));
  }
  S.view = computeView();
  update();
}

function sourceSize() {
  if (S.media === 'video') return { w: video.videoWidth, h: video.videoHeight };
  if (S.media === 'image') return { w: still.naturalWidth, h: still.naturalHeight };
  const v = S.meta?.header?.video;
  return { w: v?.width ?? 1280, h: v?.height ?? 720 };
}

/** Zoom window: union of this frame's (and nearby frames') detections, padded. */
/** Expand a rect to the canvas aspect ratio, about its centre (no stretching). */
function fitAspect(r) {
  const ar = canvas.width / canvas.height;
  let { w, h } = r;
  if (w / h > ar) h = w / ar; else w = h * ar;
  return { x: r.x + r.w / 2 - w / 2, y: r.y + r.h / 2 - h / 2, w, h };
}

function computeView() {
  const { w, h } = sourceSize();
  const full = fitAspect({ x: 0, y: 0, w, h });
  if (!$('evZoom').checked) return full;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = Math.max(0, S.idx - 15); k < Math.min(S.frames.length, S.idx + 15); k++) {
    for (const t of S.frames[k].teeth) {
      if (!t.bbox) continue;
      x0 = Math.min(x0, t.bbox[0]); y0 = Math.min(y0, t.bbox[1]);
      x1 = Math.max(x1, t.bbox[0] + t.bbox[2]); y1 = Math.max(y1, t.bbox[1] + t.bbox[3]);
    }
  }
  for (const g of currentGt().teeth) {
    const p = g.point ?? [g.box[0] + g.box[2] / 2, g.box[1] + g.box[3] / 2];
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
  }
  if (!Number.isFinite(x0)) return full;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const vw = Math.max((x1 - x0) * 2.0, 160), vh = Math.max((y1 - y0) * 2.6, 120);
  return fitAspect({ x: cx - vw / 2, y: cy - vh / 2, w: vw, h: vh });
}

const key = () => String(S.idx);
function currentGt() {
  return S.gt[key()] ?? { teeth: [], ignore: [] };
}
function editGt(fn) {
  const g = structuredClone(currentGt());
  g.teeth ??= []; g.ignore ??= [];
  S.history.push({ k: key(), before: S.gt[key()] ? structuredClone(S.gt[key()]) : null });
  fn(g);
  S.gt[key()] = g;
  update();
}

// ------------------------------------------------------------ drawing
const toCanvas = (x, y) => {
  const v = S.view, sx = canvas.width / v.w, sy = canvas.height / v.h;
  return [(x - v.x) * sx, (y - v.y) * sy];
};
const toSource = (cx, cy) => {
  const v = S.view;
  return [v.x + (cx / canvas.width) * v.w, v.y + (cy / canvas.height) * v.h];
};

function draw() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!S.view) return;
  const src = S.media === 'video' ? video : S.media === 'image' ? still : null;
  if (src) {
    const { w } = sourceSize();
    const v = S.view;
    ctx.save();
    ctx.imageSmoothingQuality = 'high';
    if (S.mirroredSource) {
      // stored mirrored: flip back so it matches raw-pixel metadata
      ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
      ctx.drawImage(src, w - (v.x + v.w), v.y, v.w, v.h, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.drawImage(src, v.x, v.y, v.w, v.h, 0, 0, canvas.width, canvas.height);
    }
    ctx.restore();
  }
  const f = S.frames[S.idx];
  if (f && $('evShowDets').checked) {
    for (const t of f.teeth) {
      const col = t.jaw === 'lower' ? '#ffb27a' : '#7cf6b0';
      ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      if (t.contour?.length >= 3) {
        ctx.beginPath();
        t.contour.forEach(([x, y], k) => { const [a, b] = toCanvas(x, y); k ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
        ctx.closePath(); ctx.stroke();
      } else if (t.bbox) {
        const [a, b] = toCanvas(t.bbox[0], t.bbox[1]);
        const [c, d] = toCanvas(t.bbox[0] + t.bbox[2], t.bbox[1] + t.bbox[3]);
        ctx.strokeRect(a, b, c - a, d - b);
      }
      if (t.center) {
        const [a, b] = toCanvas(t.center[0], t.center[1]);
        ctx.fillStyle = col; ctx.font = '600 11px system-ui';
        ctx.fillText(`T${t.id} ${t.conf.toFixed(2)}`, a + 4, b - 4);
      }
    }
  }
  const g = currentGt();
  for (const r of g.ignore ?? []) {
    const [a, b] = toCanvas(r[0], r[1]); const [c, d] = toCanvas(r[0] + r[2], r[1] + r[3]);
    ctx.fillStyle = 'rgba(150,150,150,0.25)'; ctx.fillRect(a, b, c - a, d - b);
    ctx.strokeStyle = '#aaa'; ctx.setLineDash([5, 4]); ctx.strokeRect(a, b, c - a, d - b); ctx.setLineDash([]);
  }
  for (const t of g.teeth) {
    const col = t.jaw === 'lower' ? '#ff4d8d' : '#ff3b3b';
    if (t.box) {
      const [a, b] = toCanvas(t.box[0], t.box[1]); const [c, d] = toCanvas(t.box[0] + t.box[2], t.box[1] + t.box[3]);
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(a, b, c - a, d - b);
    } else {
      const [a, b] = toCanvas(t.point[0], t.point[1]);
      ctx.beginPath(); ctx.arc(a, b, 6, 0, Math.PI * 2);
      ctx.fillStyle = t.partial ? 'rgba(200,0,255,0.85)' : col; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  if (S.drag) {
    ctx.strokeStyle = S.mode === 'ignore' ? '#ccc' : '#ff3b3b';
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(S.drag.cx0, S.drag.cy0, S.drag.cx1 - S.drag.cx0, S.drag.cy1 - S.drag.cy0);
    ctx.setLineDash([]);
  }
}

// ------------------------------------------------------------ metrics
function detsOf(f) {
  return (f?.teeth ?? []).filter((t) => t.bbox).map((t) => ({
    box: { x: t.bbox[0], y: t.bbox[1], w: t.bbox[2], h: t.bbox[3] },
    center: t.center ? { x: t.center[0], y: t.center[1] } : undefined,
    polygon: t.contour?.length >= 3 ? t.contour.map(([x, y]) => ({ x, y })) : null,
    jaw: t.jaw, conf: t.conf,
  }));
}

function gtsOf(g, clearOnly) {
  const ignore = (g.ignore ?? []).map(([x, y, w, h]) => ({ x, y, w, h }));
  let teeth = g.teeth.map((t) => (t.box
    ? { box: { x: t.box[0], y: t.box[1], w: t.box[2], h: t.box[3] }, jaw: t.jaw, partial: !!t.partial }
    : { point: { x: t.point[0], y: t.point[1] }, jaw: t.jaw, partial: !!t.partial }));
  if (clearOnly) {
    for (const t of teeth.filter((q) => q.partial)) {
      const p = t.point ?? { x: t.box.x + t.box.w / 2, y: t.box.y + t.box.h / 2 };
      ignore.push(t.box ?? { x: p.x - 10, y: p.y - 12, w: 20, h: 24 });
    }
    teeth = teeth.filter((q) => !q.partial);
  }
  return { teeth, ignore };
}

const pct = (v) => (v == null ? '<span class="metric-na">n/a</span>' : `${(v * 100).toFixed(1)}%`);
const rows = (pairs) => pairs.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

function renderFrameTable() {
  const f = S.frames[S.idx];
  const g = S.gt[key()];
  const base = [
    ['time', f ? `${(f.t_ms / 1000).toFixed(2)} s` : '—'],
    ['detections', f ? `${f.n} (conf ${f.avg_conf ?? '—'})` : '—'],
    ['fps / detect', f ? `${f.fps} / ${f.timing?.detect_ms ?? '—'} ms` : '—'],
  ];
  if (!g?.teeth?.length) {
    $('evFrameTable').innerHTML = rows([...base, ['ground truth', '<span class="metric-na">not annotated</span>']]);
    return;
  }
  const { teeth, ignore } = gtsOf(g, $('evClearOnly').checked);
  const r = evaluateFrame(detsOf(f), teeth, { ignore });
  $('evFrameTable').innerHTML = rows([...base,
    ['GT teeth', teeth.length], ['matched (TP)', r.tp], ['missed (FN)', r.fn],
    ['false positives', r.fp], ['duplicates', r.duplicates], ['merges', r.merges]]);
}

function renderAgg() {
  const clear = $('evClearOnly').checked;
  const res = [];
  for (const [k, g] of Object.entries(S.gt)) {
    if (!g.teeth?.length) continue;
    const { teeth, ignore } = gtsOf(g, clear);
    res.push(evaluateFrame(detsOf(S.frames[Number(k)]), teeth, { ignore }));
  }
  const s = aggregate(res);
  if (!s) { $('evAggTable').innerHTML = rows([['annotated frames', '<span class="metric-na">none yet</span>']]); return; }
  $('evAggTable').innerHTML = rows([
    ['annotated frames', s.frames], ['GT teeth', s.gtTeeth], ['detections', s.detections],
    ['precision', pct(s.precision)], ['recall (detection rate)', pct(s.recall)], ['F1', pct(s.f1)],
    ['missed teeth', s.fn], ['false positives', s.fp], ['duplicates', s.duplicates], ['merges', s.merges],
    ['count MAE', s.countMAE.toFixed(2)], ['exact-count frames', pct(s.exactCountRate)],
    ['mean IoU (box GT)', s.meanIoU == null ? '<span class="metric-na">n/a</span>' : s.meanIoU.toFixed(3)],
    ['jaw accuracy', pct(s.jawAccuracy)],
  ]);
  S.lastSummary = s;
}

function renderRecStats() {
  const fr = S.frames;
  if (!fr.length) { $('evRecTable').innerHTML = ''; return; }
  const mouth = fr.filter((f) => f.mouth);
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const fps = mean(fr.map((f) => f.fps).filter((v) => v > 0));
  const det = mean(fr.filter((f) => f.timing?.detect_ms > 0).map((f) => f.timing.detect_ms));
  let jac = 0, n = 0, switches = 0;
  for (let i = 1; i < fr.length; i++) {
    const a = new Set(fr[i - 1].teeth.map((t) => t.id)), b = new Set(fr[i].teeth.map((t) => t.id));
    if (!a.size || !b.size) continue;
    let inter = 0; for (const id of a) if (b.has(id)) inter++;
    jac += inter / (a.size + b.size - inter); n++;
    switches += [...b].filter((id) => !a.has(id)).length;
  }
  const ids = new Set(fr.flatMap((f) => f.teeth.map((t) => t.id)));
  const meanTeeth = mean(mouth.map((f) => f.n));
  $('evRecTable').innerHTML = rows([
    ['frames (with mouth)', `${fr.length} (${mouth.length})`],
    ['mean camera FPS', fps == null ? '—' : fps.toFixed(1)],
    ['mean detection latency', det == null ? '—' : `${det.toFixed(1)} ms`],
    ['mean teeth / frame (mouth)', meanTeeth == null ? '—' : meanTeeth.toFixed(2)],
    ['ID continuity (Jaccard)', n ? pct(jac / n) : '—'],
    ['new IDs appearing', `${switches} over ${n} frame pairs`],
    ['unique tooth IDs', ids.size],
  ]);
  S.recStats = { frames: fr.length, framesWithMouth: mouth.length, meanFps: fps, meanDetectMs: det,
    meanTeeth, idContinuity: n ? jac / n : null, newIds: switches, uniqueIds: ids.size };
}

function update() {
  $('evFrameLabel').textContent = S.frames.length ? `${S.idx + 1} / ${S.frames.length}` : '—';
  draw();
  renderFrameTable();
  renderAgg();
}

// ------------------------------------------------------------ input
function canvasPoint(ev) {
  const r = canvas.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * canvas.width, ((ev.clientY - r.top) / r.height) * canvas.height];
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (ev) => {
  if (!S.frames.length) return;
  const [cx, cy] = canvasPoint(ev);
  if (ev.button === 2 || ev.altKey) { deleteNearest(cx, cy); return; }
  S.drag = { cx0: cx, cy0: cy, cx1: cx, cy1: cy };
  canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener('pointermove', (ev) => {
  if (!S.drag) return;
  [S.drag.cx1, S.drag.cy1] = canvasPoint(ev);
  draw();
});
canvas.addEventListener('pointerup', () => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  const [x0, y0] = toSource(Math.min(d.cx0, d.cx1), Math.min(d.cy0, d.cy1));
  const [x1, y1] = toSource(Math.max(d.cx0, d.cx1), Math.max(d.cy0, d.cy1));
  const r1 = (v) => Math.round(v * 10) / 10;
  const moved = Math.hypot(d.cx1 - d.cx0, d.cy1 - d.cy0) > 6;
  const partial = $('evPartial').checked;
  if (S.mode === 'ignore' && moved) {
    editGt((g) => g.ignore.push([r1(x0), r1(y0), r1(x1 - x0), r1(y1 - y0)]));
  } else if (S.mode === 'box' && moved) {
    editGt((g) => g.teeth.push({ box: [r1(x0), r1(y0), r1(x1 - x0), r1(y1 - y0)], jaw: S.jaw, partial }));
  } else if (!moved) {
    const [x, y] = toSource(d.cx0, d.cy0);
    editGt((g) => g.teeth.push({ point: [r1(x), r1(y)], jaw: S.jaw, partial }));
  } else {
    draw();
  }
});

function deleteNearest(cx, cy) {
  const g = currentGt();
  let best = -1, bd = 30, kind = 'teeth';
  g.teeth.forEach((t, i) => {
    const p = t.point ?? [t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2];
    const [a, b] = toCanvas(p[0], p[1]);
    const d = Math.hypot(a - cx, b - cy);
    if (d < bd) { bd = d; best = i; kind = 'teeth'; }
  });
  (g.ignore ?? []).forEach((r, i) => {
    const [a, b] = toCanvas(r[0] + r[2] / 2, r[1] + r[3] / 2);
    const d = Math.hypot(a - cx, b - cy);
    if (d < bd) { bd = d; best = i; kind = 'ignore'; }
  });
  if (best >= 0) editGt((q) => q[kind].splice(best, 1));
}

function setSeg(id, attr, value) {
  for (const b of $(id).querySelectorAll('button')) b.classList.toggle('on', b.dataset[attr] === value);
}
$('evMode').addEventListener('click', (e) => { const m = e.target.dataset.mode; if (m) { S.mode = m; setSeg('evMode', 'mode', m); } });
$('evJaw').addEventListener('click', (e) => { const j = e.target.dataset.jaw; if (j) { S.jaw = j; setSeg('evJaw', 'jaw', j); } });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') return;
  if (e.key === 'ArrowRight') showFrame(S.idx + 1);
  else if (e.key === 'ArrowLeft') showFrame(S.idx - 1);
  else if (e.key === 'u' || e.key === 'U') { S.jaw = 'upper'; setSeg('evJaw', 'jaw', 'upper'); }
  else if (e.key === 'l' || e.key === 'L') { S.jaw = 'lower'; setSeg('evJaw', 'jaw', 'lower'); }
  else if (e.key === 'p' || e.key === 'P') $('evPartial').checked = !$('evPartial').checked;
  else if (e.key === 'z' || e.key === 'Z') { $('evZoom').checked = !$('evZoom').checked; S.view = computeView(); update(); }
  else if (e.key === 'Backspace') {
    const h = S.history.pop();
    if (h) { if (h.before) S.gt[h.k] = h.before; else delete S.gt[h.k]; update(); }
    e.preventDefault();
  }
});

$('evSlider').addEventListener('input', (e) => showFrame(Number(e.target.value)));
$('evPrev').addEventListener('click', () => showFrame(S.idx - 1));
$('evNext').addEventListener('click', () => showFrame(S.idx + 1));
$('evNextMouth').addEventListener('click', () => {
  for (let k = S.idx + 1; k < S.frames.length; k++) if (S.frames[k].mouth && S.frames[k].n) return showFrame(k);
  return null;
});
for (const id of ['evZoom', 'evShowDets', 'evClearOnly']) {
  $(id).addEventListener('change', () => { S.view = computeView(); update(); });
}
$('evOffset').addEventListener('change', () => showFrame(S.idx));

const wrap = (fn) => async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  try { banner(''); await fn(f); } catch (err) { banner(`${f.name}: ${err.message}`, 'error'); }
};
$('evMetaFile').addEventListener('change', wrap(loadMeta));
$('evVideoFile').addEventListener('change', wrap(loadMedia));
$('evGtFile').addEventListener('change', wrap(loadGt));

$('evExportGt').addEventListener('click', () => {
  const doc = {
    schema: 'dental-ar-gt/1',
    source: S.sourceName,
    metadataStartedAt: S.meta?.header?.startedAt ?? null,
    space: 'raw (unmirrored) video pixels; frame keys are metadata frame indices',
    protocol: 'point = crown centre of each visible tooth; partial = cut/shadowed/uncertain; ignore = teeth visible but not separable',
    frames: S.gt,
  };
  saveBlob(new Blob([JSON.stringify(doc, null, 1)], { type: 'application/json' }),
    `${(S.sourceName || 'recording').replace(/\.[^.]+$/, '')}_gt.json`);
});
$('evExportReport').addEventListener('click', () => {
  renderAgg();
  const doc = {
    source: S.sourceName, generatedAt: new Date().toISOString(),
    clearTeethOnly: $('evClearOnly').checked,
    accuracy: S.lastSummary ?? null,
    recording: S.recStats ?? null,
    detector: S.meta?.header?.detector ?? null,
  };
  saveBlob(new Blob([JSON.stringify(doc, null, 1)], { type: 'application/json' }),
    `${(S.sourceName || 'recording').replace(/\.[^.]+$/, '')}_report.json`);
});

update();

// Read-only handle for automated tests of this page (no effect on behaviour).
window.__dentalEval = S;
