import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { pathToFileURL } from 'node:url';

const exec = promisify(execFile);
const adb = async args => (await exec('adb', args, { timeout: 8000, maxBuffer: 128 * 1024 })).stdout;

export function parseDevices(output) {
  return output.split(/\r?\n/).map(line => line.trim().split(/\s+/))
    .filter(parts => parts.length >= 2 && !['List', '*'].includes(parts[0]))
    .map(([serial, state]) => ({ serial, state }));
}

// Keep the initially selected device: never silently switch to another phone.
export function createUsbKeeper({ serial, port = 5173, run = adb } = {}) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Port must be between 1 and 65535.');
  let selected = serial;
  const endpoint = `tcp:${port}`;
  return async () => {
    const devices = parseDevices(await run(['devices']));
    if (!selected) {
      if (devices.length > 1) return 'Multiple devices found. Restart with --serial DEVICE_ID.';
      selected = devices[0]?.serial;
    }
    const device = devices.find(d => d.serial === selected);
    if (!device) return 'Waiting for Android USB connection…';
    if (device.state !== 'device') return `Android ${device.state}. Unlock the phone and allow USB debugging for this computer.`;
    const rules = await run(['-s', selected, 'reverse', '--list']);
    const forwarded = rules.split(/\r?\n/).some(line => {
      const fields = line.trim().split(/\s+/);
      return fields[1] === endpoint && fields[2] === endpoint;
    });
    if (!forwarded) await run(['-s', selected, 'reverse', endpoint, endpoint]);
    return `USB ready (${selected}). Open the Android USB pairing QR in Chrome; localhost:${port} forwards to this computer.`;
  };
}

async function main() {
  const args = process.argv.slice(2), options = {};
  for (let i = 0; i < args.length; i++) {
    if (!['--serial', '--port'].includes(args[i]) || !args[i + 1])
      throw new Error('Usage: npm run phone:android -- [--serial DEVICE_ID] [--port 5173]');
    options[args[i].slice(2)] = args[++i];
  }
  if (options.port != null) options.port = Number(options.port);
  const check = createUsbKeeper(options);
  let stopped = false, previous = '', timer;
  const stop = () => { stopped = true; clearTimeout(timer); };
  process.once('SIGINT', stop); process.once('SIGTERM', stop);
  console.log('Keeping Android USB forwarding active. Keep npm run dev running separately. Ctrl+C stops monitoring.');
  const tick = async () => {
    let status;
    try { status = await check(); }
    catch (e) { status = e.code === 'ENOENT' ? 'ADB not found. Install Android platform-tools, then restart this command.' : `USB unavailable: ${e.message.trim()}`; }
    if (status !== previous) { console.log(status); previous = status; }
    if (!stopped) timer = setTimeout(tick, 2000);
  };
  await tick();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(e => { console.error(e.message); process.exitCode = 1; });
}
