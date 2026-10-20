#!/usr/bin/env node
// Generate annotation starting points, never supervised labels or accuracy scores.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { encodeLabels, resizeInstanceLabels, instanceStats } from '../src/eval/toothInstances.js';

const [manifestPath, modelPath] = process.argv.slice(2);
if (!modelPath) throw new Error('Usage: draft_upload_labels.mjs unreviewed-manifest.json model.onnx');
const root = path.dirname(path.resolve(manifestPath));
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
if (manifest.schema !== 'unreviewed-upload-session/1') throw new Error('Expected an unreviewed upload session');
const modelSha256 = createHash('sha256').update(fs.readFileSync(modelPath)).digest('hex');
const detector = new LearnedToothDetector({ modelUrl: path.resolve(modelPath) });
try {
  await detector.init();
  const W = detector.inputWidth, H = detector.inputHeight;
  if (manifest.input[0] !== W || manifest.input[1] !== H) throw new Error('Model input dimensions differ');
  for (const s of manifest.samples) {
    if (![s.width, s.height].every(n => Number.isInteger(n) && n > 0) || s.width * s.height > 4_000_000) {
      throw new Error('Invalid native crop dimensions');
    }
    const rgb = fs.readFileSync(path.resolve(root, s.rgb));
    if (rgb.length !== W * H * 3) throw new Error('Invalid RGB size');
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      data.set(rgb.subarray(i * 3, i * 3 + 3), i * 4); data[i * 4 + 3] = 255;
    }
    const roi = { bounds: { u0: 0, u1: 1, v0: 0, v1: 1 },
      roiToLocal: (x, y) => ({ u: x / W, v: y / H }) };
    await detector.detectAsync({ data, width: W, height: H }, new Uint8Array(W * H).fill(255), roi);
    const labels = resizeInstanceLabels(detector.lastDebug.labels, W, H, s.width, s.height);
    const doc = { schema: 'tooth-labels/1', annotationStatus: 'unreviewed_model_draft',
      cleanImage: path.basename(s.image), crop: [0, 0, s.width, s.height],
      width: s.width, height: s.height, labels: encodeLabels(labels),
      provenance: { modelSha256, decoder: detector.params,
        warning: 'Review every tooth, add missing regions and split merges. This is not ground truth.' } };
    // Exclusive writes prevent a rerun from silently replacing annotation work.
    fs.writeFileSync(path.join(root, `${s.id}_draft_labels.json`), JSON.stringify(doc), { flag: 'wx' });
    const mask = Buffer.alloc(labels.length);
    for (let i = 0; i < labels.length; i++) mask[i] = labels[i] > 0 ? 255 : 0;
    fs.writeFileSync(path.join(root, `${s.id}_draft_mask.pgm`),
      Buffer.concat([Buffer.from(`P5\n${s.width} ${s.height}\n255\n`), mask]), { flag: 'wx' });
    console.log(`${s.id}: ${instanceStats(labels, s.width).length} predicted regions (unreviewed)`);
  }
} finally { detector.dispose(); }
