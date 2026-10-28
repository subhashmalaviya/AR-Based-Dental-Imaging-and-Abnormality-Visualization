import { IMAGE_ACCEPT } from './imageInput.js';

export async function phoneRequest(path, key, options = {}) {
  const response = await fetch(`/__phone/${path}`, { ...options, headers: {
    'X-Dental-Client': '1', ...(key ? { Authorization: `Bearer ${key}` } : {}), ...options.headers,
  } });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.error ?? 'Phone pairing needs the local server. Run npm run dev or npm run dev:https on your computer.');
  }
  return response;
}

// Shared by the desktop app (local/USB webcam) and the lightweight phone sender.
export class CaptureControls {
  constructor(root, onCapture) {
    this.root = root; this.onCapture = onCapture;
    root.innerHTML = `<div class="buttons buttons--wrap">
      <label class="btn">Take / choose photo<input data-c="photoFile" type="file" accept="${IMAGE_ACCEPT}" capture="environment"></label>
      <label class="btn">Record / choose video<input data-c="videoFile" type="file" accept="video/*" capture="environment"></label>
      </div><p class="hint">These buttons use your phone camera or file picker. For a continuous preview, open the camera below. Photos go to GT comparison; videos go to teeth detection.</p>
      <label class="row row--stack">Camera<select data-c="device"><option value="environment">Rear camera</option><option value="user">Front camera</option></select></label>
      <div class="buttons buttons--wrap"><button class="btn" data-c="refresh">Refresh cameras</button><button class="btn" data-c="open">Open camera</button><button class="btn" data-c="close" disabled>Close camera</button></div>
      <video data-c="preview" playsinline muted autoplay hidden style="width:100%;max-height:480px;object-fit:contain"></video>
      <div class="buttons buttons--wrap"><button class="btn" data-c="photo" disabled>Capture photo → input</button><button class="btn" data-c="record" disabled>Record video</button><button class="btn" data-c="stop" disabled>Stop recording → input</button></div>
      <p data-c="status" class="hint" role="status">Camera preview requires HTTPS or localhost. Native capture buttons are also available. Recordings are silent, limited to 2 minutes / 200 MB.</p>`;
    this.el = key => root.querySelector(`[data-c="${key}"]`); this.video = this.el('preview');
    for (const [id, kind] of [['photoFile', 'photo'], ['videoFile', 'video']]) this.el(id).addEventListener('change', async e => {
      const file = e.target.files[0]; e.target.value = ''; if (file) await this.deliver(file, kind);
    });
    this.el('refresh').onclick = () => this.refresh().catch(e => this.status(e.message));
    this.el('open').onclick = () => this.open(); this.el('close').onclick = () => this.close();
    this.el('photo').onclick = () => this.photo(); this.el('record').onclick = () => this.record();
    this.el('stop').onclick = () => this.stopRecording();
    window.addEventListener('pagehide', () => { this.disposed = true; this.close(); });
  }
  status(message) { this.el('status').textContent = message; }
  sync() {
    const active = !!this.stream, recording = !!this.recorder;
    for (const id of ['device', 'refresh', 'open', 'photoFile', 'videoFile']) this.el(id).disabled = !!this.busy || recording;
    this.el('photo').disabled = !active || !!this.busy || recording;
    this.el('record').disabled = !active || !!this.busy || recording || !globalThis.MediaRecorder;
    this.el('stop').disabled = !recording; this.el('close').disabled = !active || !!this.busy;
  }
  async deliver(file, kind) {
    this.busy = true; this.sync();
    try {
      if (!file.size || file.size > 200_000_000) throw new Error('Capture must be nonempty and below 200 MB.');
      this.status(`Opening ${file.name}…`);
      await this.onCapture(file, kind); this.status(`${file.name}: sent to input.`);
    } catch (e) { this.status(e.message); }
    finally { this.busy = false; this.sync(); }
  }
  async refresh() {
    if (!navigator.mediaDevices?.enumerateDevices) throw new Error('Camera preview needs HTTPS or localhost. Use the native capture buttons above.');
    const selected = this.el('device').value;
    this.el('device').replaceChildren(new Option('Rear camera', 'environment'), new Option('Front camera', 'user'));
    for (const [i, d] of (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput').entries())
      this.el('device').add(new Option(d.label || `Camera ${i + 1}`, `device:${d.deviceId}`));
    if ([...this.el('device').options].some(o => o.value === selected)) this.el('device').value = selected;
  }
  async open() {
    if (this.busy || this.recorder) return;
    this.close(); this.busy = true; this.sync();
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Use HTTPS or localhost for preview, or use the native capture buttons above.');
      const selected = this.el('device').value;
      const source = selected.startsWith('device:') ? { deviceId: { exact: selected.slice(7) } } : { facingMode: { ideal: selected } };
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...source, width: { ideal: 1920 }, height: { ideal: 1080 } } });
      this.video.srcObject = this.stream; this.video.hidden = false; await this.video.play();
      await this.refresh(); this.status('Camera ready. Capture a clean photo or record a video.');
    } catch (e) { this.close(); this.status(`Camera unavailable: ${e.message}`); }
    finally { this.busy = false; this.sync(); }
  }
  close() {
    this.stopRecording();
    this.stream?.getTracks().forEach(t => t.stop()); this.stream = null;
    this.video.srcObject = null; this.video.hidden = true; this.sync();
  }
  async photo() {
    if (this.busy || !this.stream || this.recorder) return;
    this.busy = true; this.sync();
    try {
      let blob;
      if (globalThis.ImageCapture) {
        try { blob = await new ImageCapture(this.stream.getVideoTracks()[0]).takePhoto(); } catch { /* Frame fallback below. */ }
      }
      if (!blob) {
        const c = document.createElement('canvas'); c.width = this.video.videoWidth; c.height = this.video.videoHeight;
        if (!c.width || !c.height) throw new Error('Wait for a camera frame.');
        c.getContext('2d').drawImage(this.video, 0, 0);
        blob = await new Promise(resolve => c.toBlob(resolve, 'image/png'));
      }
      if (!blob) throw new Error('Could not capture photo.');
      await this.deliver(new File([blob], `camera-${Date.now()}.${blob.type.includes('jpeg') ? 'jpg' : 'png'}`, { type: blob.type }), 'photo');
    } catch (e) { this.status(e.message); }
    finally { this.busy = false; this.sync(); }
  }
  record() {
    if (!this.stream || this.busy || this.recorder) return;
    try {
      if (!globalThis.MediaRecorder) throw new Error('Use Record / choose video above; browser recording is unavailable.');
      const mimeType = ['video/mp4', 'video/webm;codecs=vp8', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : {});
      const chunks = []; let bytes = 0, failed = false;
      recorder.ondataavailable = e => {
        bytes += e.data.size;
        if (bytes > 200_000_000) { failed = true; chunks.length = 0; this.status('Recording exceeded 200 MB. Record a shorter clip.'); this.stopRecording(); }
        else if (!failed && e.data.size) chunks.push(e.data);
      };
      recorder.onerror = () => { failed = true; this.status('Recording failed. Use the native video capture button.'); this.stopRecording(); };
      recorder.onstop = async () => {
        clearTimeout(this.recordTimer); this.recorder = null;
        if (!failed && !this.disposed) {
          const type = recorder.mimeType || chunks[0]?.type || 'video/webm';
          await this.deliver(new File(chunks, `camera-${Date.now()}.${type.includes('mp4') ? 'mp4' : 'webm'}`, { type }), 'video');
        }
        this.sync();
      };
      recorder.start(1000); this.recorder = recorder;
      this.recordTimer = setTimeout(() => this.stopRecording(), 120_000);
      this.status('Recording… Stop recording to send this clip directly to video input.'); this.sync();
    } catch (e) { this.status(e.message); }
  }
  stopRecording() { if (this.recorder?.state === 'recording') this.recorder.stop(); clearTimeout(this.recordTimer); }
}

export class PhoneCapture {
  constructor({ root, onCapture, canReceive }) {
    Object.assign(this, { root, onCapture, canReceive });
    root.innerHTML = `<details><summary><strong>Phone / USB camera → photo and video input</strong></summary>
      <div data-local-capture></div><h3>Connect a phone to this computer</h3>
      <p class="hint">On the same Wi-Fi or phone hotspot, create a link and open it on your phone. Captures transfer through this computer’s local server. Keep the pairing link private. Pairings expire after 30 minutes of inactivity.</p>
      <div class="buttons buttons--wrap"><button class="btn" data-p="pair">Create phone pairing link</button><button class="btn" data-p="disconnect" disabled>Disconnect phone</button></div>
      <div data-p="links"></div><p class="hint" data-p="status" role="status"></p>
      <div class="buttons buttons--wrap"><button class="btn" data-p="use" hidden>Use received capture</button><button class="btn" data-p="dismiss" hidden>Dismiss received capture</button></div>
      <details><summary>USB cable and hosting setup</summary><p class="hint">Android USB: enable USB debugging, connect the cable, run <code>adb reverse tcp:5173 tcp:5173</code>, and open the localhost pairing link on the phone (use <code>npm run dev</code>). For Wi-Fi, use a displayed LAN address; <code>npm run dev:https</code> enables the camera preview where the phone trusts the certificate. Native capture/file buttons are available if preview is blocked. An iPhone can use Wi-Fi or a hotspot. Plugging in a cable alone does not expose its camera to a browser. If your phone/driver exposes a USB webcam, select it under Camera. A static hosted site supports capture on that device; cross-device pairing requires this local server.</p></details>
      </details>`;
    this.el = key => root.querySelector(`[data-p="${key}"]`);
    this.local = new CaptureControls(root.querySelector('[data-local-capture]'), async (file, kind) => {
      if (this.pending) throw new Error('Use or dismiss the pending capture first.');
      if (!canReceive(kind)) { this.pending = { file, kind }; this.showPending(); return; }
      await onCapture(file, kind);
    });
    this.el('pair').onclick = () => this.pair();
    this.el('disconnect').onclick = () => this.disconnect();
    this.el('use').onclick = () => this.usePending();
    this.el('dismiss').onclick = () => this.dismiss();
    window.addEventListener('pagehide', () => { clearTimeout(this.timer); });
  }
  status(message) { this.el('status').textContent = message; }
  async pair() {
    this.el('pair').disabled = true;
    try {
      await this.disconnect();
      this.session = await (await phoneRequest('session', null, { method: 'POST' })).json();
      const { id, sender, origins } = this.session;
      for (const origin of origins) {
        const url = new URL('/phone.html', origin); url.hash = new URLSearchParams({ id, key: sender }).toString();
        const row = document.createElement('p'), link = document.createElement('a'), copy = document.createElement('button');
        link.href = url.href; link.textContent = `${origin}/phone.html — open on phone`; link.target = '_blank'; link.rel = 'noopener';
        copy.className = 'btn'; copy.textContent = 'Copy pairing link';
        copy.onclick = async () => { try { await navigator.clipboard.writeText(url.href); this.status('Pairing link copied. Open it on your phone.'); } catch { this.status('Copy the link address using your browser’s link menu.'); } };
        row.append(link, ' ', copy); this.el('links').append(row);
      }
      this.el('disconnect').disabled = false; this.status('Waiting for phone captures. Use a LAN address for Wi-Fi or localhost with Android USB forwarding.');
      this.poll();
    } catch (e) { this.status(e.message); }
    finally { this.el('pair').disabled = false; }
  }
  async disconnect() {
    clearTimeout(this.timer); const s = this.session; this.session = null;
    this.pending = null; this.showPending(); this.el('links').replaceChildren(); this.el('disconnect').disabled = true;
    if (s) { try { await phoneRequest(`${s.id}/close`, s.receiver, { method: 'DELETE' }); } catch { /* Expiry also clears data. */ } }
    this.status('Phone disconnected.');
  }
  async poll() {
    const s = this.session; if (!s) return;
    try {
      if (!this.pending) {
        const meta = await (await phoneRequest(`${s.id}/pending`, s.receiver)).json();
        if (this.session !== s) return;
        if (meta) {
          const blob = await (await phoneRequest(`${s.id}/media`, s.receiver)).blob();
          if (this.session !== s) return;
          this.pending = { file: new File([blob], meta.name, { type: meta.type }), kind: meta.kind, session: s };
          this.showPending();
          if (this.canReceive(meta.kind)) await this.usePending();
        }
      }
    } catch (e) { this.status(e.message); }
    finally { if (this.session === s) this.timer = setTimeout(() => this.poll(), 1500); }
  }
  showPending() {
    this.el('use').hidden = this.el('dismiss').hidden = !this.pending;
    if (this.pending) this.status(`${this.pending.file.name} received. Use received capture when ready to replace the current input, or dismiss it.`);
  }
  async usePending() {
    if (!this.pending || this.receiving) return;
    this.receiving = true; this.el('use').disabled = true; this.el('dismiss').disabled = true;
    try {
      await this.onCapture(this.pending.file, this.pending.kind);
      await this.dismiss(); this.status('Capture loaded into the corresponding input. Ready for your work.');
    } catch (e) { this.status(e.message); }
    finally { this.receiving = false; this.el('use').disabled = false; this.el('dismiss').disabled = false; }
  }
  async dismiss() {
    const p = this.pending;
    try {
      if (p?.session) await phoneRequest(`${p.session.id}/media`, p.session.receiver, { method: 'DELETE' });
      if (this.pending === p) this.pending = null;
      this.showPending();
    } catch (e) { this.status(e.message); }
  }
}
