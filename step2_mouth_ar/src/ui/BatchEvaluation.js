import { readImageFile } from './imageInput.js';
import { evaluationDetector, inferCanvas } from '../eval/inferCanvas.js';
import { pairPatientFiles, parsePatientTruth, scoreBatchPrediction, summarizeBatch } from '../eval/batchEvaluation.js';

const pause = () => new Promise(resolve => setTimeout(resolve, 0));
const format = value => value == null ? '—' : value.toFixed(6);

export class BatchEvaluation {
  constructor({ root, getDetector, modelUrl }) {
    Object.assign(this, { root, getDetector, modelUrl }); this.rows = []; this.pairs = [];
    root.innerHTML = `<details><summary><strong>Batch patient evaluation → Excel report</strong></summary>
      <p>Select a main folder containing patient subfolders. Each clean image needs a JSON file with the same name in the same folder, for example Patient_1/IOP_Center_1.png and IOP_Center_1.json.</p>
      <label class="row row--stack">Main folder path (local project)<input data-b="path" type="text" placeholder="/home/subhash/iitd-sem-7/cod492/testvideophoto"></label>
      <div class="buttons buttons--wrap"><button class="btn" data-b="scan">Scan folder path</button>
      <label class="btn">Choose main folder<input data-b="folder" type="file" webkitdirectory multiple style="display:block;max-width:100%"></label></div>
      <p class="hint">Path entry works on this computer with the local server. Choose main folder also works on hosted pages. Files and model inference stay on this computer. Patient JSON with teeth contours and saved tooth-labels/1 JSON are supported.</p>
      <div class="buttons buttons--wrap"><button class="btn btn--primary" data-b="run" disabled>Run batch comparison</button><button class="btn" data-b="cancel" disabled>Stop after current image</button><button class="btn" data-b="download" disabled>Download Excel report</button></div>
      <p data-b="status" role="status"></p><progress data-b="progress" value="0" max="1" style="width:100%"></progress>
      <p data-b="summary"></p>
      <p class="hint">Error is the same normalized bidirectional boundary distance as the photo comparison: 0 = exact match; lower is better. Overall error is the mean of scored images. Missing boundaries and failed entries are reported separately. Uses the selected Detector, the full image (or an explicitly saved crop), and never a crop derived from GT.</p>
      <div style="overflow:auto;max-height:440px"><table><thead><tr><th>Patient</th><th>Image</th><th>Boundary error (0–1)</th><th>Status</th></tr></thead><tbody data-b="rows"></tbody></table></div>
      </details>`;
    this.el = key => root.querySelector(`[data-b="${key}"]`);
    this.el('scan').onclick = () => this.scan();
    this.el('folder').onchange = e => {
      const files = [...e.target.files];
      if (files.length > 5000) { this.status('Select at most 5000 files.'); return; }
      this.source = files[0]?.webkitRelativePath.split('/')[0] ?? 'Selected folder';
      this.setFiles(files.map(file => ({ path: file.webkitRelativePath || file.name, size: file.size, file })));
    };
    this.el('run').onclick = () => this.run();
    this.el('cancel').onclick = () => { this.cancelled = true; this.el('cancel').disabled = true; this.status('Stopping after the current image…'); };
    this.el('download').onclick = () => this.download();
  }
  status(text) { this.el('status').textContent = text; }
  lock(value) {
    this.busy = value;
    for (const key of ['scan', 'folder', 'path', 'run']) this.el(key).disabled = value;
    this.el('cancel').disabled = !value; this.el('download').disabled = value || !this.runInfo;
    if (!value) this.el('run').disabled = !this.pairs.some(p => !p.issue);
  }
  async request(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { 'X-Dental-Client': '1', ...options.headers } });
    if (!response.ok) { const detail = await response.json().catch(() => null); throw new Error(detail?.error ?? 'Folder access failed. Use Choose main folder on hosted pages.'); }
    if (response.headers.get('Content-Type')?.includes('text/html')) throw new Error('Folder paths require the local server. Use Choose main folder here.');
    return response;
  }
  async scan() {
    if (this.busy) return; this.lock(true); this.el('cancel').disabled = true;
    this.status('Scanning patient folders…');
    try {
      const folder = this.el('path').value.trim();
      const data = await (await this.request('/__batch/scan', { method: 'POST', body: JSON.stringify({ folder }), headers: { 'Content-Type': 'application/json' } })).json();
      this.source = folder; this.setFiles(data.files);
    } catch (e) { this.status(e.message); }
    finally { this.lock(false); }
  }
  setFiles(files) {
    this.pairs = pairPatientFiles(files); this.runInfo = null;
    this.rows = this.pairs.map(p => ({ patient: p.patient, image: p.image?.path ?? '', annotation: p.annotation?.path ?? '',
      status: p.issue ? 'unpaired' : 'pending', details: p.issue ?? '', error: null }));
    this.render(); this.el('download').disabled = !this.rows.length;
    this.el('run').disabled = !this.pairs.some(p => !p.issue);
    this.el('progress').max = Math.max(1, this.pairs.length); this.el('progress').value = 0;
    this.status(`${this.pairs.filter(p => !p.issue).length} image/JSON pairs found; ${this.pairs.filter(p => p.issue).length} unmatched or ambiguous entries. Ready to run the selected detector.`);
  }
  async file(entry) {
    if (entry.size > 80 * 1024 * 1024) throw new Error('File exceeds 80 MB');
    if (entry.file) return entry.file;
    const blob = await (await this.request(entry.url)).blob();
    return new File([blob], entry.path.split('/').at(-1));
  }
  async run() {
    if (this.busy || !this.pairs.length) return;
    this.cancelled = false; this.lock(true);
    const selection = structuredClone(this.getDetector()); let detector;
    this.runInfo = { timestamp: new Date().toISOString(), source: this.source, detector: selection.key,
      model: selection.key === 'learned' ? this.modelUrl : selection.key === 'classical' ? 'Classical CV' : JSON.stringify(selection.options), parameters: selection.params };
    this.rows = this.pairs.map(p => ({ patient: p.patient, image: p.image?.path ?? '', annotation: p.annotation?.path ?? '',
      status: p.issue ? 'unpaired' : 'pending', details: p.issue ?? '', error: null }));
    this.render(); this.el('progress').value = 0;
    try {
      this.status(`Loading ${selection.key} detector…`);
      detector = evaluationDetector(selection, this.modelUrl); await detector.init?.(); detector.setParams?.(selection.params);
      this.runInfo.parameters = { ...(detector.params ?? selection.params) };
      this.runInfo.inputSize = [detector.inputWidth ?? 192, detector.inputHeight ?? 144];
      for (let index = 0; index < this.pairs.length; index++) {
        if (this.cancelled) break;
        const pair = this.pairs[index], row = this.rows[index];
        if (!pair.issue) {
          this.status(`Evaluating ${index + 1}/${this.pairs.length}: ${pair.image.path}`);
          await pause(); let clean, input;
          try {
            const annotation = await this.file(pair.annotation);
            if (annotation.size > 40_000_000) throw new Error('JSON exceeds 40 MB');
            const doc = JSON.parse(await annotation.text());
            clean = await readImageFile(await this.file(pair.image));
            row.width = clean.width; row.height = clean.height;
            const truth = parsePatientTruth(doc, clean.width, clean.height, pair.image.path.split('/').at(-1));
            row.crop = truth.crop; row.patientId = truth.patientId;
            const [x, y, w, h] = truth.crop;
            input = clean;
            if (x || y || w !== clean.width || h !== clean.height) {
              input = Object.assign(document.createElement('canvas'), { width: w, height: h });
              input.getContext('2d').drawImage(clean, x, y, w, h, 0, 0, w, h);
            }
            const prediction = await inferCanvas(detector, input);
            Object.assign(row, scoreBatchPrediction(truth.contours, prediction.labels, w, h), { elapsedMs: prediction.elapsedMs });
          } catch (e) { row.status = 'failed'; row.details = e.message; }
          finally { if (input && input !== clean) input.width = input.height = 1; if (clean) clean.width = clean.height = 1; }
        }
        this.el('progress').value = index + 1; this.render(); await pause();
      }
    } catch (e) {
      for (const row of this.rows) if (row.status === 'pending') { row.status = 'failed'; row.details = `Detector initialization failed: ${e.message}`; }
    } finally {
      try { await detector?.dispose?.(); } catch { /* Evaluation results remain available. */ }
      this.runInfo.cancelled = this.cancelled;
      for (const row of this.rows) if (row.status === 'pending') row.status = 'cancelled';
      this.render(); this.lock(false);
      this.status(this.cancelled ? 'Stopped. Download the partial report; unprocessed entries are marked cancelled.' : 'Batch finished. Download the Excel report; check unavailable entries in the table.');
    }
  }
  render() {
    const summary = summarizeBatch(this.rows);
    this.el('summary').textContent = `Overall mean boundary error: ${format(summary.mean)} · ${summary.scored}/${summary.total} entries scored · ${summary.unavailable} without a score`;
    const body = this.el('rows'); body.replaceChildren();
    for (const row of this.rows) {
      const tr = document.createElement('tr');
      for (const text of [row.patient, row.image || row.annotation, format(row.error), [row.status, row.details].filter(Boolean).join(': ')]) {
        const td = document.createElement('td'); td.textContent = text; tr.append(td);
      }
      body.append(tr);
    }
  }
  async download() {
    if (this.busy || !this.rows.length) return;
    this.el('download').disabled = true;
    try {
      const { createBatchWorkbook } = await import('../eval/batchWorkbook.js');
      const run = this.runInfo ?? { timestamp: new Date().toISOString(), source: this.source, detector: 'Not run', model: 'Not run' };
      const workbook = await createBatchWorkbook(this.rows, run), bytes = await workbook.xlsx.writeBuffer();
      const url = URL.createObjectURL(new Blob([bytes], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
      const link = document.createElement('a'); link.href = url; link.download = `dental-batch-${run.timestamp.slice(0, 10)}.xlsx`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);
      this.status('Excel report downloaded. Summary contains the overall mean; Images contains every entry and its status.');
    } catch (e) { this.status(`Excel export failed: ${e.message}`); }
    finally { this.el('download').disabled = false; }
  }
}
