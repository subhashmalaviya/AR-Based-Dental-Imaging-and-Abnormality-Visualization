// Copies ONNX Runtime Web's WebAssembly runtime into public/ort so the tooth
// model loads from a fixed, same-origin path in every setup — Vite dev server,
// production build, Vercel and the Capacitor WebView. (Under `vite dev` the
// package's own relative wasm URL resolved to index.html, and the app fell
// back to the classical detector without the model ever running.)
import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, 'node_modules/onnxruntime-web/dist');
const dst = resolve(root, 'public/ort');
await mkdir(dst, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  await copyFile(resolve(src, f), resolve(dst, f));
}
console.log('vendored ONNX Runtime wasm -> public/ort');
