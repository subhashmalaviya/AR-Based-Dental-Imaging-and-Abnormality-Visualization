import { CaptureControls, phoneRequest } from './ui/PhoneCapture.js';
const params = new URLSearchParams(location.hash.slice(1)), id = params.get('id'), key = params.get('key');
const status = document.getElementById('phoneStatus');
if (!/^[a-f0-9]{48}$/.test(id ?? '') || !/^[a-f0-9]{48}$/.test(key ?? '')) {
  status.textContent = 'Open a complete pairing link from Phone / USB camera in Dental AR on your computer.';
} else {
  const connection = document.createElement('p'); connection.setAttribute('role', 'status');
  status.before(connection);
  let heartbeatTimer, stopped = false, heartbeatRunning = false;
  const heartbeat = async () => {
    if (stopped || heartbeatRunning) return;
    heartbeatRunning = true;
    clearTimeout(heartbeatTimer);
    try {
      await phoneRequest(`${id}/hello`, key, { method: 'POST' });
      connection.textContent = 'Connected to Dental AR on your computer.';
    } catch (e) { connection.textContent = `Connection unavailable: ${e.message}`; }
    finally { heartbeatRunning = false; if (!stopped) heartbeatTimer = setTimeout(heartbeat, 10000); }
  };
  window.addEventListener('pagehide', () => { stopped = true; clearTimeout(heartbeatTimer); });
  window.addEventListener('pageshow', () => { stopped = false; heartbeat(); });
  heartbeat();
  status.textContent = 'Capture a photo or video to send it to the computer.';
  new CaptureControls(document.getElementById('phoneSender'), async (file, kind) => {
    status.textContent = `Sending ${file.name}… Keep this page open.`;
    try {
      await phoneRequest(`${id}/upload`, key, { method: 'POST', body: file, headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-File-Name': encodeURIComponent(file.name), 'X-Media-Kind': kind,
      } });
      status.textContent = 'Received by computer. If another input is open there, press Use received capture on the computer.';
    } catch (e) { status.textContent = e.message; throw e; }
  });
}
