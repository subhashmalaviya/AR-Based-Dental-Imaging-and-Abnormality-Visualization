"""
roi_geometry.py -- the app's mouth-ROI rectification, reproduced in Python.

Training data, evaluation crops and the live app must all see the mouth the
same way, or a model trained here would be tested on a distribution it never
saw. This module is a line-for-line port of:

  src/core/MouthTracker.js   mouth frame: +X corner->corner, +Y upper->lower
                             lip (Gram-Schmidt, same as vec3.orthonormalBasis)
  src/core/MouthROI.js       bounds = inner-lip-ring extent + 16% padding,
                             affine resample to a fixed-size canvas, aperture
                             = inner ring polygon eroded off the lips

The only intentional difference: the app smooths the anchor with a one-euro
filter across frames; a single photograph has no history, so none is applied.
"""
import cv2
import numpy as np

CORNER_L, CORNER_R = 61, 291
UPPER_OUTER, LOWER_OUTER = 0, 17
UPPER_INNER, LOWER_INNER = 13, 14
INNER_RING = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
PAD = 0.16


def aperture_erosion_radius(roi_h):
    """Must match MouthROI.apertureMask(): radius, not kernel size."""
    return max(1, round(roi_h * 0.022))


def mouth_frame(landmarks, w, h):
    """landmarks: sequence with .x .y .z (normalised, MediaPipe) or (N,3) array."""
    arr = np.array([[p.x, p.y, p.z] for p in landmarks], np.float64) \
        if not isinstance(landmarks, np.ndarray) else landmarks.astype(np.float64)
    px = arr * np.array([w, h, w])
    xa = px[CORNER_R] - px[CORNER_L]
    width = float(np.linalg.norm(xa))
    if width < 1e-6:
        return None
    xa /= width
    up = px[LOWER_OUTER] - px[UPPER_OUTER]
    ya = up - xa * np.dot(up, xa)
    n = np.linalg.norm(ya)
    if n < 1e-6:
        return None
    ya /= n
    ring = px[INNER_RING]
    origin = ring.mean(axis=0)
    local = np.stack([(ring - origin) @ xa, (ring - origin) @ ya], axis=1) / width
    opening = float(np.linalg.norm(px[LOWER_INNER] - px[UPPER_INNER]) / width)
    return {"origin": origin, "xa": xa, "ya": ya, "width": width,
            "local": local, "opening": opening, "px": px}


def roi_bounds(fr, pad=PAD, pad_top=None, pad_bottom=None):
    """pad_top / pad_bottom: headroom above / below the inner lip ring
    (fraction of its height). MediaPipe's inner ring sits inside the real
    lip line when the mouth is wide open, so the upper row can fall outside a
    tight crop. Both default to `pad`."""
    u0, v0 = fr["local"].min(axis=0)
    u1, v1 = fr["local"].max(axis=0)
    du = (u1 - u0) * pad
    dt = (v1 - v0) * (pad if pad_top is None else pad_top)
    db = (v1 - v0) * (pad if pad_bottom is None else pad_bottom)
    return (u0 - du, u1 + du, v0 - dt, v1 + db)


def roi_maps(fr, bounds, out_w, out_h):
    """Sampling maps (ROI pixel -> source pixel), pixel-centre aligned like drawImage."""
    u0, u1, v0, v1 = bounds
    us = u0 + (np.arange(out_w) + 0.5) / out_w * (u1 - u0)
    vs = v0 + (np.arange(out_h) + 0.5) / out_h * (v1 - v0)
    U, V = np.meshgrid(us, vs)
    o, xa, ya, s = fr["origin"], fr["xa"], fr["ya"], fr["width"]
    X = o[0] + s * (U * xa[0] + V * ya[0]) - 0.5
    Y = o[1] + s * (U * xa[1] + V * ya[1]) - 0.5
    return X.astype(np.float32), Y.astype(np.float32)


def rectify(img, fr, bounds, out_w, out_h, interp=cv2.INTER_LINEAR):
    X, Y = roi_maps(fr, bounds, out_w, out_h)
    return cv2.remap(img, X, Y, interp, borderMode=cv2.BORDER_REPLICATE)


def aperture_mask(fr, bounds, out_w, out_h):
    u0, u1, v0, v1 = bounds
    pts = np.stack([(fr["local"][:, 0] - u0) / (u1 - u0) * out_w,
                    (fr["local"][:, 1] - v0) / (v1 - v0) * out_h], axis=1)
    m = np.zeros((out_h, out_w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    r = aperture_erosion_radius(out_h)
    return cv2.erode(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))


def roi_to_frame(fr, bounds, out_w, out_h, xy):
    """ROI pixel coords (N,2) -> source frame pixel coords (N,2)."""
    u0, u1, v0, v1 = bounds
    xy = np.asarray(xy, np.float64)
    U = u0 + xy[:, 0] / out_w * (u1 - u0)
    V = v0 + xy[:, 1] / out_h * (v1 - v0)
    o, xa, ya, s = fr["origin"], fr["xa"], fr["ya"], fr["width"]
    return np.stack([o[0] + s * (U * xa[0] + V * ya[0]),
                     o[1] + s * (U * xa[1] + V * ya[1])], axis=1)


def make_landmarker(model_path, video=False):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    mode = mp_vision.RunningMode.VIDEO if video else mp_vision.RunningMode.IMAGE
    lmk = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode, num_faces=1,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4))

    def run(bgr, ts_ms=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = lmk.detect_for_video(img, int(ts_ms)) if video else lmk.detect(img)
        return res.face_landmarks[0] if res.face_landmarks else None
    return run
