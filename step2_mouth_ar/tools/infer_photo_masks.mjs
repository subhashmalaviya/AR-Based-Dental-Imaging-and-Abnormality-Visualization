#!/usr/bin/env node
// Run the shipped detector and decoder on prepared, clean RGB mouth crops.
import fs from 'node:fs';
import path from 'node:path';
import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';

const [manifestPath, modelPath] = process.argv.slice(2);
if (!manifestPath || !modelPath) throw new Error('Usage: infer_photo_masks.mjs manifest.json model.onnx');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const dir = path.dirname(manifestPath);
const detector = new LearnedToothDetector({ modelUrl: path.resolve(modelPath) });
await detector.init();
try {
  const W = detector.inputWidth, H = detector.inputHeight;
  if (W !== manifest.width || H !== manifest.height) throw new Error('Model/crop dimensions differ');
  for (const crop of manifest.crops) {
    const rgb = fs.readFileSync(path.join(dir, `${crop.name}.rgb`));
    if (rgb.length !== W * H * 3) throw new Error('Wrong RGB byte count');
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      data.set(rgb.subarray(i * 3, i * 3 + 3), i * 4);
      data[i * 4 + 3] = 255;
    }
    const roi = { bounds: { u0: 0, u1: 1, v0: 0, v1: 1 },
      roiToLocal: (x, y) => ({ u: x / W, v: y / H }) };
    const detections = await detector.detectAsync({ data, width: W, height: H },
      new Uint8Array(W * H).fill(255), roi);
    const { maps, labels } = detector.lastDebug;
    fs.writeFileSync(path.join(dir, `${crop.name}_decoded.u8`),
      Uint8Array.from(labels, x => x > 0 ? 255 : 0));
    fs.writeFileSync(path.join(dir, `${crop.name}_semantic.u8`),
      Uint8Array.from(maps.teeth, x => x >= (detector.params.semThr ?? 0.5) ? 255 : 0));
    crop.detections = detections.length;
  }
  fs.writeFileSync(path.join(dir, 'inference.json'), JSON.stringify({
    model: path.resolve(modelPath), inputWidth: W, inputHeight: H,
    decoder: detector.params, crops: manifest.crops,
    scope: 'Detector only; manually fixed crops; no face landmarks, tracking or lip aperture gating',
  }, null, 2));
} finally {
  detector.dispose();
}
