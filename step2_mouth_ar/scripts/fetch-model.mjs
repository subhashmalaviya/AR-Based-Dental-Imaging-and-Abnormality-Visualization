// Downloads the MediaPipe Face Landmarker model into public/models/, so a
// fresh clone can run the app without committing a 3.7 MB binary to git.
// Cross-platform (plain Node https, no curl/wget dependency) and idempotent —
// safe to run on every `npm install` via the "postinstall" script.
import { createWriteStream, existsSync, mkdirSync, statSync, unlinkSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import https from 'node:https';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const MIN_EXPECTED_BYTES = 1_000_000; // the real file is ~3.7 MB; guards against a truncated/error download

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dest = resolve(root, 'public/models/face_landmarker.task');

function download(url, dest, redirectsLeft = 5) {
  return new Promise((resolvePromise, reject) => {
    https.get(url, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        res.resume();
        if (redirectsLeft <= 0) return reject(new Error('too many redirects'));
        return resolvePromise(download(res.headers.location, dest, redirectsLeft - 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} fetching ${url}`));
      }
      const file = createWriteStream(dest);
      res.pipe(file);
      file.on('finish', () => file.close(() => resolvePromise()));
      file.on('error', reject);
    }).on('error', reject);
  });
}

async function main() {
  if (existsSync(dest) && statSync(dest).size > MIN_EXPECTED_BYTES) {
    console.log(`[fetch-model] already present (${(statSync(dest).size / 1e6).toFixed(1)} MB) — skipping`);
    return;
  }
  mkdirSync(dirname(dest), { recursive: true });
  console.log('[fetch-model] downloading face_landmarker.task ...');
  try {
    await download(MODEL_URL, dest);
    const size = statSync(dest).size;
    if (size < MIN_EXPECTED_BYTES) throw new Error(`downloaded file looks truncated (${size} bytes)`);
    console.log(`[fetch-model] done: ${dest} (${(size / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    if (existsSync(dest)) unlinkSync(dest);
    console.error(`[fetch-model] FAILED: ${err.message}`);
    console.error('[fetch-model] The app cannot detect faces without this file.');
    console.error(`[fetch-model] Retry with: npm run fetch:model`);
    console.error(`[fetch-model] Or download manually from:\n  ${MODEL_URL}\n  -> ${dest}`);
    process.exitCode = 1;
  }
}

main();
