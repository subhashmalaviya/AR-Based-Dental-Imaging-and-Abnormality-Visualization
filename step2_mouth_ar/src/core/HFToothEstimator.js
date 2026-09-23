/**
 * HFToothEstimator.js — Estimates tooth dimensions (mm) from an intraoral
 * photo using the Hugging Face Inference API.
 *
 * PRIMARY MODEL: sha000/deta-finetuned-teeth-v1
 * -------------------------------------------------------
 * DETA (DEtection TrAnsformer) fine-tuned on intraoral dental photos.
 * It is `endpoints_compatible` on the HF serverless Inference API and returns
 * standard [{score, label, box:{xmin,ymin,xmax,ymax}}] bounding boxes.
 *
 * FALLBACK: geometric estimation from live pipeline tracks
 * If the HF API call fails (network, bad token, model cold-start), the caller
 * can supply live `ToothPipeline` tracks + pixelsPerMm and get a geometric
 * estimate directly from those on-screen boxes.
 *
 * CALIBRATION
 * -----------
 * Pixel-to-mm conversion uses the iris-derived `pixelsPerMm` from IrisScaler.
 * The caller must pass the current value; estimation is refused if unavailable.
 *
 * ARCH & TOOTH-TYPE CLASSIFICATION
 * ----------------------------------
 * Upper arch = boxes whose centre-y is in the upper half of the frame.
 * Within each arch, teeth are sorted by horizontal offset from the arch
 * midline; the N closest are assigned positions 1-8 (CI → M3).
 */

/** @type {string} Default model endpoint — a real HF model that works via the Inference API. */
export const HF_MODEL_ID = 'sha000/deta-finetuned-teeth-v1';

/** Ordered tooth-type keys from midline outwards (positions 1–8). */
export const ARCH_TOOTH_KEYS = ['CI', 'LI', 'C', 'PM1', 'PM2', 'M1', 'M2', 'M3'];

/** Minimum confidence threshold. */
const MIN_SCORE = 0.25;

// =====================================================================
// PUBLIC API
// =====================================================================

/**
 * Estimates tooth dimensions by calling the Hugging Face Inference API.
 *
 * @param {Blob}   imageBlob    — JPEG/PNG blob of the captured frame.
 * @param {number} pixelsPerMm  — Current iris-calibrated scale (px/mm).
 * @param {string} hfToken      — HF read-access token (Bearer).
 * @param {object} [opts]
 * @param {string} [opts.modelId]  — Override the default HF model ID.
 * @param {number} [opts.minScore] — Override minimum detection confidence.
 * @returns {Promise<HFEstimateResult>}
 */
export async function estimateFromBlob(imageBlob, pixelsPerMm, hfToken, opts = {}) {
  if (!imageBlob) throw new Error('HFToothEstimator: imageBlob is required');
  if (!pixelsPerMm || pixelsPerMm <= 0) {
    throw new Error(
      'HFToothEstimator: pixelsPerMm must be a positive number. ' +
      'Wait for the iris scale to become available before estimating.'
    );
  }
  if (!hfToken?.trim()) throw new Error('HFToothEstimator: a Hugging Face API token is required');

  const modelId  = opts.modelId  ?? HF_MODEL_ID;
  const minScore = opts.minScore ?? MIN_SCORE;
  const urls = [
    `https://router.huggingface.co/hf-inference/models/${modelId}`,
    `https://api-inference.huggingface.co/models/${modelId}`,
  ];

  // ----------------------------------------------------------- API call
  let response;
  let lastErr;
  for (const url of urls) {
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${hfToken.trim()}`,
          'Content-Type': imageBlob.type || 'image/jpeg',
        },
        body: imageBlob,
      });
      if (response.ok || response.status === 401 || response.status === 403 || response.status === 503) {
        break;
      }
    } catch (err) {
      lastErr = err;
    }
  }

  if (!response) {
    throw new Error(`HFToothEstimator: network error — ${lastErr?.message || 'Failed to connect to Hugging Face API'}`);
  }

  if (!response.ok) {
    let msg = `HTTP ${response.status}`;
    try {
      const body = await response.text();
      const parsed = JSON.parse(body);
      // HF loading response: {"error":"Model ... is currently loading","estimated_time":20}
      if (parsed.estimated_time != null) {
        throw new Error(
          `HFToothEstimator: API error — Model is loading (retry in ~${Math.ceil(parsed.estimated_time)}s)`
        );
      }
      if (parsed.error) msg += ` — ${parsed.error}`;
    } catch (inner) {
      if (inner.message.includes('HFToothEstimator')) throw inner;
      // ignore JSON parse failure
    }
    throw new Error(`HFToothEstimator: API error — ${msg}`);
  }

  let detections;
  try {
    detections = await response.json();
  } catch (err) {
    throw new Error(`HFToothEstimator: could not parse API response — ${err.message}`);
  }

  if (!Array.isArray(detections)) {
    // Some models wrap result: { generated_text: ... } or { predictions: [...] }
    const wrapped = detections?.predictions ?? detections?.output ?? detections?.boxes;
    if (Array.isArray(wrapped)) {
      detections = wrapped;
    } else {
      throw new Error(
        `HFToothEstimator: unexpected API response shape — ` +
        `got ${JSON.stringify(detections).slice(0, 120)}`
      );
    }
  }

  return _buildResult(detections, pixelsPerMm, minScore);
}

/**
 * Estimates tooth dimensions directly from the **live pipeline tracks** already
 * on-screen — no API call needed.  Use as a fallback / instant preview.
 *
 * @param {object[]} tracks      — ToothPipeline track objects with .box {u,v,w,h}
 *                                 in mouth-local coords, plus .arch ('upper'|'lower').
 * @param {Function} localToScreen — anchor.localToScreen bound to the current frame.
 * @param {number}   pixelsPerMm
 * @returns {HFEstimateResult}
 */
export function estimateFromTracks(tracks, localToScreen, pixelsPerMm) {
  if (!pixelsPerMm || pixelsPerMm <= 0) throw new Error('pixelsPerMm required');
  if (!tracks?.length) return _emptyResult([]);

  // Convert each track's mouth-local box to pixel bounding box
  const detections = tracks.map((t) => {
    const s  = t.smoothed ?? t;
    const tl = localToScreen({ x: s.box.u,            y: s.box.v,            z: 0 });
    const br = localToScreen({ x: s.box.u + s.box.w,  y: s.box.v + s.box.h,  z: 0 });
    return {
      score: t.confidence ?? 0.8,
      label: 'tooth',
      box: { xmin: tl.x, ymin: tl.y, xmax: br.x, ymax: br.y },
      arch: t.arch ?? null,   // may already be classified
    };
  });

  return _buildResult(detections, pixelsPerMm, 0.0, true /* archPreClassified */);
}

// =====================================================================
// INTERNALS
// =====================================================================

/**
 * Core processing: filter, convert to mm, separate arches, assign tooth types.
 *
 * @param {object[]} rawDetections
 * @param {number}   pixelsPerMm
 * @param {number}   minScore
 * @param {boolean}  [archPreClassified] — if true, respect .arch field already set
 * @returns {HFEstimateResult}
 */
function _buildResult(rawDetections, pixelsPerMm, minScore, archPreClassified = false) {
  // Filter by confidence
  const filtered = rawDetections
    .filter((d) => (d.score ?? 1) >= minScore)
    .map((d) => {
      const box  = d.box ?? {};
      // Handle both absolute-pixel (xmax>1) and normalised [0-1] coordinates
      const scale = (box.xmax > 1 || box.ymax > 1) ? 1 : 1000; // normalised → upscale
      const xmin = (box.xmin ?? 0) * scale;
      const ymin = (box.ymin ?? 0) * scale;
      const xmax = (box.xmax ?? 0) * scale;
      const ymax = (box.ymax ?? 0) * scale;

      const w_px  = Math.abs(xmax - xmin);
      const h_px  = Math.abs(ymax - ymin);
      const cx    = (xmin + xmax) / 2;
      const cy    = (ymin + ymax) / 2;

      return {
        score:     d.score ?? 1,
        label:     d.label ?? 'tooth',
        box:       { xmin, ymin, xmax, ymax },
        cx, cy,
        width_px:  w_px,
        height_px: h_px,
        width_mm:  w_px / pixelsPerMm,
        height_mm: h_px / pixelsPerMm,
        arch:      archPreClassified ? (d.arch ?? null) : null,
        toothKey:  null,
      };
    })
    .filter((d) => d.width_px > 2 && d.height_px > 2);   // discard degenerate boxes

  if (!filtered.length) return _emptyResult([]);

  // Upper / lower separation (if not already set)
  if (!archPreClassified) {
    const imageH = _estimateImageHeight(filtered);
    for (const d of filtered) {
      d.arch = d.cy < imageH / 2 ? 'upper' : 'lower';
    }
  } else {
    // For pre-classified tracks that still have null arch, guess from y
    const imageH = _estimateImageHeight(filtered);
    for (const d of filtered) {
      if (!d.arch) d.arch = d.cy < imageH / 2 ? 'upper' : 'lower';
    }
  }

  const upper = _assignToothTypes(filtered.filter((d) => d.arch === 'upper'));
  const lower = _assignToothTypes(filtered.filter((d) => d.arch === 'lower'));

  return {
    upper,
    lower,
    raw:        filtered,
    count:      filtered.length,
    upperCount: filtered.filter((d) => d.arch === 'upper').length,
    lowerCount: filtered.filter((d) => d.arch === 'lower').length,
  };
}

/**
 * Sorts detected teeth by distance from arch midline (separated by left/right side)
 * and assigns tooth-type keys (CI, LI, C, PM1, PM2, M1, M2, M3).
 *
 * For each tooth type (e.g. CI), if detections exist on both left and right sides,
 * their measured dimensions are averaged to calculate the representative half-arch dimension.
 */
function _assignToothTypes(detections) {
  const result = {};
  if (!detections.length) return result;

  const cxVals  = detections.map((d) => d.cx).sort((a, b) => a - b);
  const midline = (cxVals[0] + cxVals[cxVals.length - 1]) / 2;

  const leftSide  = detections.filter((d) => d.cx < midline)
    .sort((a, b) => Math.abs(a.cx - midline) - Math.abs(b.cx - midline));
  const rightSide = detections.filter((d) => d.cx >= midline)
    .sort((a, b) => Math.abs(a.cx - midline) - Math.abs(b.cx - midline));

  const byKey = {};
  ARCH_TOOTH_KEYS.forEach((key, i) => {
    byKey[key] = [];
    if (leftSide[i]) {
      leftSide[i].toothKey = key;
      byKey[key].push(leftSide[i]);
    }
    if (rightSide[i]) {
      rightSide[i].toothKey = key;
      byKey[key].push(rightSide[i]);
    }
  });

  // If one side had no detections (e.g. lateral photo or offset mouth), sort all detections from midline
  const assignedCount = Object.values(byKey).reduce((acc, arr) => acc + arr.length, 0);
  if (assignedCount === 0 && detections.length > 0) {
    const sorted = [...detections].sort((a, b) => Math.abs(a.cx - midline) - Math.abs(b.cx - midline));
    sorted.forEach((d, i) => {
      if (i < ARCH_TOOTH_KEYS.length) {
        const key = ARCH_TOOTH_KEYS[i];
        d.toothKey = key;
        byKey[key] = [d];
      }
    });
  }

  ARCH_TOOTH_KEYS.forEach((key) => {
    const items = byKey[key];
    if (items && items.length > 0) {
      const avgW = items.reduce((s, d) => s + d.width_mm, 0) / items.length;
      const avgH = items.reduce((s, d) => s + d.height_mm, 0) / items.length;
      result[key] = { width: avgW, height: avgH };
    }
  });

  return result;
}

function _estimateImageHeight(detections) {
  if (!detections.length) return 480;
  const maxY = Math.max(...detections.map((d) => d.box?.ymax ?? 0));
  return maxY > 1 ? maxY * 1.4 : 1.0;
}

function _emptyResult(raw) {
  return { upper: {}, lower: {}, raw, count: 0, upperCount: 0, lowerCount: 0 };
}

// =====================================================================
// SNAPSHOT UTILITIES
// =====================================================================

/**
 * Captures the current video frame as a JPEG Blob (at half resolution).
 * @param {HTMLVideoElement} video
 * @param {number} [quality=0.88]
 * @returns {Promise<Blob>}
 */
export async function captureFrame(video, quality = 0.88) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) throw new Error('HFToothEstimator: video has no frame yet');

  const scale = 0.5;
  const oc  = document.createElement('canvas');
  oc.width  = Math.round(w * scale);
  oc.height = Math.round(h * scale);
  oc.getContext('2d').drawImage(video, 0, 0, oc.width, oc.height);

  return new Promise((resolve, reject) => {
    oc.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('canvas.toBlob failed'))),
      'image/jpeg',
      quality
    );
  });
}

/**
 * Draws a thumbnail preview of the snapshot onto a canvas element.
 * @param {Blob} blob
 * @param {HTMLCanvasElement} canvas
 */
export async function drawThumbnail(blob, canvas) {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.src = url;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  const ctx    = canvas.getContext('2d');
  const aspect = img.width / img.height;
  canvas.width  = canvas.offsetWidth || 180;
  canvas.height = Math.round(canvas.width / aspect);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);
}
