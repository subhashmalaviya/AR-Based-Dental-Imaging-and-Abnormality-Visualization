#!/usr/bin/env node
/**
 * analyze_recording.mjs — summarise recorded sessions (and score them against
 * ground truth made with eval.html) from the command line.
 *
 *   node tools/analyze_recording.mjs rec1.json rec2.json ...
 *   node tools/analyze_recording.mjs rec.json --gt rec_gt.json [--clear-only]
 *
 * Reads only the metadata JSON the app wrote next to each video; no video
 * decoding needed, no network.
 */
import fs from 'node:fs';
import path from 'node:path';
import { evaluateFrame, aggregate, formatSummary } from '../src/eval/metrics.js';

const argv = process.argv.slice(2);
const files = argv.filter((a, i) => !a.startsWith('--') && !(argv[i - 1] ?? '').startsWith('--gt'));
const gtPath = argv.includes('--gt') ? argv[argv.indexOf('--gt') + 1] : null;
const clearOnly = argv.includes('--clear-only');
if (!files.length) {
  console.error('usage: analyze_recording.mjs <metadata.json>... [--gt gt.json] [--clear-only]');
  process.exit(2);
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
const f2 = (v, d = 2) => (v == null ? 'n/a' : v.toFixed(d));

for (const file of files) {
  const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
  const fr = doc.frames ?? [];
  const mouth = fr.filter((f) => f.mouth);
  let jac = 0, n = 0;
  for (let i = 1; i < fr.length; i++) {
    const a = new Set(fr[i - 1].teeth.map((t) => t.id)), b = new Set(fr[i].teeth.map((t) => t.id));
    if (!a.size || !b.size) continue;
    let inter = 0;
    for (const id of a) if (b.has(id)) inter++;
    jac += inter / (a.size + b.size - inter);
    n++;
  }
  const confs = fr.flatMap((f) => f.teeth.map((t) => t.conf));
  console.log(`${path.basename(file)}  [${doc.header?.mode ?? '?'} | ${doc.header?.detector?.name ?? '?'}]`);
  console.log(`  frames ${fr.length} (mouth ${mouth.length})  duration ${f2((doc.summary?.duration_ms ?? 0) / 1000, 1)} s`
    + `  mean FPS ${f2(mean(fr.map((f) => f.fps).filter((v) => v > 0)), 1)}`);
  console.log(`  mean teeth/frame (mouth) ${f2(mean(mouth.map((f) => f.n)))}   mean confidence ${f2(mean(confs))}`
    + `   mean detect ${f2(mean(fr.filter((f) => f.timing?.detect_ms > 0).map((f) => f.timing.detect_ms)), 1)} ms`);
  console.log(`  ID continuity ${n ? (jac / n * 100).toFixed(1) + '%' : 'n/a'}   unique IDs ${new Set(fr.flatMap((f) => f.teeth.map((t) => t.id))).size}`);

  if (gtPath) {
    const gt = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
    const res = [];
    for (const [k, g] of Object.entries(gt.frames ?? {})) {
      const f = fr[Number(k)];
      if (!f || !g.teeth?.length) continue;
      const ignore = (g.ignore ?? []).map(([x, y, w, h]) => ({ x, y, w, h }));
      let teeth = g.teeth.map((t) => (t.box
        ? { box: { x: t.box[0], y: t.box[1], w: t.box[2], h: t.box[3] }, jaw: t.jaw, partial: !!t.partial }
        : { point: { x: t.point[0], y: t.point[1] }, jaw: t.jaw, partial: !!t.partial }));
      if (clearOnly) {
        teeth.filter((q) => q.partial).forEach((t) => {
          const p = t.point ?? { x: t.box.x + t.box.w / 2, y: t.box.y + t.box.h / 2 };
          ignore.push({ x: p.x - 10, y: p.y - 12, w: 20, h: 24 });
        });
        teeth = teeth.filter((q) => !q.partial);
      }
      const dets = f.teeth.filter((t) => t.bbox).map((t) => ({
        box: { x: t.bbox[0], y: t.bbox[1], w: t.bbox[2], h: t.bbox[3] },
        center: t.center ? { x: t.center[0], y: t.center[1] } : undefined,
        polygon: t.contour?.length >= 3 ? t.contour.map(([x, y]) => ({ x, y })) : null,
        jaw: t.jaw,
      }));
      res.push(evaluateFrame(dets, teeth, { ignore }));
    }
    console.log(formatSummary(`  accuracy vs ${path.basename(gtPath)}${clearOnly ? ' (clear teeth only)' : ''}`, aggregate(res))
      .split('\n').map((l) => `  ${l}`).join('\n'));
  }
  console.log('');
}
