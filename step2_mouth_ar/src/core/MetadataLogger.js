/**
 * MetadataLogger.js — per-frame analysis record that accompanies a recording.
 *
 * Built entirely client-side from values the pipeline already computed for
 * the frame; nothing is re-derived and nothing leaves the device. The result
 * is a single JSON document saved next to the video with the same base name:
 *
 *   dental_tracking_20260911_101530.webm
 *   dental_tracking_20260911_101530.json
 *
 * Schema (version 1) — see README "Metadata format":
 *   header  { schema, app, startedAt, mode, video:{mimeType,width,height},
 *             mirrored:{preview,annotatedVideo,rawVideo}, coordinates,
 *             detector:{name,learned,model}, userAgent }
 *   frames  [{ i, t_ms, fps, face, mouth, opening, reason, n, avg_conf,
 *              stability, timing:{detect_ms,track_ms,total_ms},
 *              teeth:[{ id, jaw, conf, state, hits, bbox, center, uv,
 *                       contour?, pos3d_m?, depth_src? }] }]
 *   summary { frames, duration_ms, mean_fps, mean_teeth, unique_ids, ... }
 *
 * Coordinates: bbox/center/contour are in RAW (unmirrored) video pixels,
 * rounded to 0.1 px. uv is the mouth-local position (1.0 = mouth width).
 */
export const METADATA_SCHEMA = 'dental-ar-recording/1';

const r1 = (v) => Math.round(v * 10) / 10;
const r3 = (v) => Math.round(v * 1000) / 1000;

export class MetadataLogger {
  constructor({ includeContours = true, maxContourPoints = 16 } = {}) {
    this.includeContours = includeContours;
    this.maxContourPoints = maxContourPoints;
    this.reset();
  }

  reset() {
    this.header = null;
    this.frames = [];
    this.active = false;
  }

  begin(header) {
    this.reset();
    this.header = { schema: METADATA_SCHEMA, ...header };
    this.active = true;
  }

  /**
   * @param {object} f
   * @param {number} f.t_ms  ms since recording start
   * @param {Array} f.tracks visible ToothTracks (smoothed)
   * @param {object} f.anchor MouthARAnchor (for mouth-local -> pixel)
   */
  log({ t_ms, fps, face, mouth, opening, reason, stats, timing, tracks, anchor, anchors3D, light }) {
    if (!this.active) return;
    const teeth = [];
    for (const t of tracks ?? []) {
      const s = t.smoothed ?? t;
      const rec = {
        id: t.id,
        jaw: t.arch,
        conf: r3(s.confidence),
        state: t.status,
        hits: t.hits,
        uv: [r3(s.center.u), r3(s.center.v)],
      };
      if (anchor?.isValid?.()) {
        const c = anchor.localToScreen({ x: s.center.u, y: s.center.v, z: 0 });
        const corners = [
          [s.box.u, s.box.v], [s.box.u + s.box.w, s.box.v],
          [s.box.u, s.box.v + s.box.h], [s.box.u + s.box.w, s.box.v + s.box.h],
        ].map(([u, v]) => anchor.localToScreen({ x: u, y: v, z: 0 }));
        const xs = corners.map((p) => p.x), ys = corners.map((p) => p.y);
        const x0 = Math.min(...xs), y0 = Math.min(...ys);
        rec.center = [r1(c.x), r1(c.y)];
        rec.bbox = [r1(x0), r1(y0), r1(Math.max(...xs) - x0), r1(Math.max(...ys) - y0)];
        if (this.includeContours && s.contour?.length) {
          const step = Math.max(1, Math.ceil(s.contour.length / this.maxContourPoints));
          rec.contour = [];
          for (let k = 0; k < s.contour.length; k += step) {
            const p = anchor.localToScreen({ x: s.contour[k].u, y: s.contour[k].v, z: 0 });
            rec.contour.push([r1(p.x), r1(p.y)]);
          }
        }
      }
      const a3 = anchors3D?.get?.(t.id);
      if (a3) {
        const p = a3.position;
        rec.pos3d_m = [r3(p.x), r3(p.y), r3(p.z)];
        rec.depth_src = a3.pose?.depthSource;
      }
      teeth.push(rec);
    }
    this.frames.push({
      i: this.frames.length,
      t_ms: t_ms == null ? null : r1(t_ms),
      fps: r1(fps ?? 0),
      face: !!face,
      mouth: !!mouth,
      opening: opening == null ? null : r3(opening),
      reason: reason ?? null,
      n: teeth.length,
      avg_conf: stats?.count ? r3(stats.avgConfidence) : null,
      stability: stats?.status ?? null,
      light: light == null ? null : Math.round(light),
      timing: timing ? {
        detect_ms: r1(timing.detect), track_ms: r1(timing.track), total_ms: r1(timing.total),
      } : null,
      teeth,
    });
  }

  /** Close the log and compute a summary from the logged frames only. */
  end(extra = {}) {
    this.active = false;
    const fr = this.frames;
    const withMouth = fr.filter((f) => f.mouth);
    const ids = new Set();
    for (const f of fr) for (const t of f.teeth) ids.add(t.id);
    const mean = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);
    const last = fr[fr.length - 1];
    return {
      header: this.header,
      summary: {
        frames: fr.length,
        duration_ms: last?.t_ms ?? 0,
        frames_with_mouth: withMouth.length,
        mean_fps: mean(fr.map((f) => f.fps)),
        mean_teeth_when_mouth: mean(withMouth.map((f) => f.n)),
        unique_tooth_ids: ids.size,
        mean_detect_ms: mean(fr.filter((f) => f.timing).map((f) => f.timing.detect_ms)),
        ...extra,
      },
      frames: fr,
    };
  }

  static toBlob(doc) {
    return new Blob([JSON.stringify(doc)], { type: 'application/json' });
  }
}
