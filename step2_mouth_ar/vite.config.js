import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// `npm run dev`       -> http://<lan-ip>:5173  (camera works on localhost only)
// `npm run dev:https` -> https://<lan-ip>:5173 (camera works on a phone)
export default defineConfig(({ mode }) => ({
  base: './',                       // relative paths: required by Capacitor
  plugins: mode === 'https' ? [basicSsl()] : [],
  server: { host: true, port: 5173 },
  preview: { host: true, port: 4173 },
  build: {
    outDir: 'dist',
    target: 'es2020',
    // The MediaPipe .wasm/.task assets live in public/ and are copied verbatim;
    // nothing here should try to inline them.
    assetsInlineLimit: 0,
  },
}));
