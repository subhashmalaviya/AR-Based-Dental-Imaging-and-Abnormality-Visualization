// Copies the MediaPipe WASM runtime out of node_modules into public/wasm so the
// app is fully self-contained: no CDN at runtime. This matters for the native
// Capacitor builds, where a WKWebView/WebView may have no network at all.
import { cp, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const dst = resolve(root, 'public/wasm');
await mkdir(dst, { recursive: true });
await cp(src, dst, { recursive: true });
console.log(`vendored MediaPipe wasm -> public/wasm`);
