#!/usr/bin/env node
// Baseline/candidate evaluation using the SAME native-label boundary metric as the editor.
import fs from 'node:fs';
import path from 'node:path';
import { LearnedToothDetector } from '../src/core/LearnedToothDetector.js';
import { decodeLabels, resizeInstanceLabels, compareInstances } from '../src/eval/toothInstances.js';
import { boundaryDistance, matchedBoundaryDistances } from '../src/eval/boundaryDistance.js';

const [manifestPath, modelPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error('Usage: evaluate_corrections.mjs manifest.json model.onnx report.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const detector = new LearnedToothDetector({ modelUrl: path.resolve(modelPath) });
await detector.init();
const samples = [];
try {
  const W = detector.inputWidth, H = detector.inputHeight;
  if (manifest.input[0] !== W || manifest.input[1] !== H) throw new Error('Prepared input dimensions differ from model');
  for (const sample of manifest.samples) {
    const rgb = fs.readFileSync(path.resolve(path.dirname(manifestPath), sample.rgb));
    if (rgb.length !== W * H * 3) throw new Error('Incorrect RGB size');
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) { data.set(rgb.subarray(3 * i, 3 * i + 3), 4 * i); data[4 * i + 3] = 255; }
    const doc = JSON.parse(fs.readFileSync(sample.labels, 'utf8'));
    const truth = decodeLabels(doc.labels, sample.width * sample.height);
    const roi = { bounds: { u0: 0, u1: 1, v0: 0, v1: 1 }, roiToLocal: (x, y) => ({ u: x / W, v: y / H }) };
    await detector.detectAsync({ data, width: W, height: H }, new Uint8Array(W * H).fill(255), roi);
    const prediction = resizeInstanceLabels(detector.lastDebug.labels, W, H, sample.width, sample.height);
    const instances = compareInstances(truth, prediction, sample.width);
    matchedBoundaryDistances(truth, prediction, sample.width, sample.height, instances.rows);
    const boundary = boundaryDistance(truth, prediction, sample.width, sample.height);
    samples.push({ id: sample.id, group: sample.group, split: sample.split,
      normalizedBoundaryError: boundary.normalizedAssd, boundary, instances,
      labelsSha256: sample.labelsSha256, imageSha256: sample.imageSha256 });
  }
  fs.writeFileSync(outputPath, JSON.stringify({ model: path.resolve(modelPath), decoder: detector.params, samples,
    scope: 'Fixed clean crops, no camera tracking. Missing teeth and unmatched instances remain explicit. A lower boundary score alone does not prove better recall or stability.' }, null, 2));
  for (const s of samples) console.log(JSON.stringify({ id: s.id, split: s.split, error: s.normalizedBoundaryError,
    groundTruth: s.instances.groundTruthCount, predicted: s.instances.predictionCount, matchedAt50: s.instances.matchedAt50 }));
} finally { detector.dispose(); }
