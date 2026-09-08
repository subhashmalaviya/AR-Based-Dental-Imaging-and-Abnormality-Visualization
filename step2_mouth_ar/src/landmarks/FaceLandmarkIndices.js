/**
 * FaceLandmarkIndices.js — named indices into MediaPipe's 478-point face mesh.
 *
 * MediaPipe Face Landmarker returns 468 mesh points (+10 iris points when
 * refinement is on). These constants are the canonical indices for the lip
 * region; keeping them in one file means Step 3 can add tooth/gum regions
 * without hunting magic numbers through the tracking code.
 *
 * Naming note: LEFT/RIGHT are **the subject's own** left and right, matching
 * the canonical model. On a mirrored selfie preview the subject's left appears
 * on the viewer's right; the anchor code never assumes a screen side, it only
 * needs the two corners to be consistently ordered so the mouth's +X axis has
 * a stable direction.
 */

// Two mouth corners (outer commissures) — the most stable lip landmarks, and
// the ones the anchor's width axis and scale are built from.
export const MOUTH_CORNER_LEFT = 61;
export const MOUTH_CORNER_RIGHT = 291;

// Lip centre points. "Outer" = vermillion border, "inner" = the aperture edge.
export const UPPER_LIP_OUTER = 0;    // cupid's bow / top of upper lip
export const UPPER_LIP_INNER = 13;   // inner edge of upper lip
export const LOWER_LIP_INNER = 14;   // inner edge of lower lip
export const LOWER_LIP_OUTER = 17;   // bottom of lower lip

// Outer lip ring, ordered around the mouth (used for the outline + bbox).
export const LIPS_OUTER_RING = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];

// Inner lip ring — encloses the mouth aperture; its area grows when the mouth
// opens, which is how the opening measure stays robust to head scale.
export const LIPS_INNER_RING = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
  308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
];

export const LIPS_ALL = [...new Set([...LIPS_OUTER_RING, ...LIPS_INNER_RING])];

// Connections for drawing the lip outline as line segments.
const ringToEdges = (ring) => ring.map((a, i) => [a, ring[(i + 1) % ring.length]]);
export const LIPS_OUTER_EDGES = ringToEdges(LIPS_OUTER_RING);
export const LIPS_INNER_EDGES = ringToEdges(LIPS_INNER_RING);

// A few extra face points used only for the face-level status/orientation HUD.
export const FACE_OVAL_SAMPLE = [10, 152, 234, 454]; // top, chin, right, left
export const NOSE_TIP = 1;
export const CHIN = 152;
export const FOREHEAD = 10;
