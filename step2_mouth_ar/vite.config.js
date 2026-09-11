import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// `npm run dev`       -> http://<lan-ip>:5173  (camera works on localhost only)
// `npm run dev:https` -> https://<lan-ip>:5173 (camera works on a phone)
// Cross-origin isolation (COOP/COEP) unlocks SharedArrayBuffer, which lets
// ONNX Runtime Web run the tooth model on several threads. Every resource the
// app loads is same-origin, so these headers cost nothing. Vercel sends the
// same headers (vercel.json). Set COI=0 to test without them.
const isolationHeaders = process.env.COI === '0' ? {} : {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig(({ mode }) => ({
  base: './',                       // relative paths: required by Capacitor
  plugins: mode === 'https' ? [basicSsl()] : [],
  server: { host: true, port: 5173, headers: isolationHeaders },
  preview: { host: true, port: 4173, headers: isolationHeaders },
  build: {
    outDir: 'dist',
    target: 'es2020',
    // The MediaPipe .wasm/.task assets live in public/ and are copied verbatim;
    // nothing here should try to inline them.
    assetsInlineLimit: 0,
    // Two pages: the live app, and the offline evaluation / annotation tool.
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        eval: fileURLToPath(new URL('./eval.html', import.meta.url)),
      },
    },
  },
}));
