#!/usr/bin/env python3
"""Local stdio adapter: use the user's IOP Mask R-CNN/checkpoints, never downloads."""
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ['DENTAL_IOP_ROOT']).resolve()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'app' / 'backend')]
# Keep stdout exclusively for the JSON protocol, including during model imports.
with contextlib.redirect_stdout(sys.stderr):
    import cv2
    import numpy as np
    import torch
    from config import Settings
    from pipeline import Pipeline
    from iop_compass.data.imaging import target_size
    from iop_compass.roi.base import full_image_result
    from iop_compass.segmentation.postprocessing import postprocess_instances
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    settings = Settings.from_env()
    problems = settings.missing(('mask-rcnn',))
    if problems:
        raise RuntimeError('; '.join(problems))
    pipeline = Pipeline(settings, ('mask-rcnn',))
    pipeline.warm_up()


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


metadata = {
    'sha256': sha(settings.maskrcnn_weights),
    'targets': ['teeth'],
    'card': {'name': 'IOP-Compass Mask R-CNN', 'architecture': 'Mask R-CNN ResNet50-FPN',
             'modelPath': str(settings.maskrcnn_weights), 'classifierPath': str(settings.classifier_weights),
             'classifierSha256': sha(settings.classifier_weights), 'modality': 'rgb-intraoral'},
    'parameters': {'device': pipeline.device, 'postprocessing': pipeline.post_params.to_dict(),
                   'inferenceLongSide': settings.inference_long_side,
                   'scoreThreshold': float(os.environ.get('IOPC_SCORE_THRESHOLD', .35)),
                   'maskThreshold': pipeline._maskrcnn.config.mask_threshold,
                   'overlapResolution': 'highest confidence first; nonoverlapping native instance labels',
                   'view': 'ResNet18 single-image argmax; no five-view constrained assignment'}
}


def infer(req):
    w, h = req['width'], req['height']
    if type(w) is not int or type(h) is not int or min(w, h) < 1 or w * h > 32_000_000:
        raise ValueError('IOP input must be a native crop up to 32 megapixels')
    raw = base64.b64decode(req['rgb'], validate=True)
    if len(raw) != w * h * 3:
        raise ValueError('Incorrect RGB buffer size')
    bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)[:, :, ::-1].copy()
    view = pipeline.classify({'sample': bgr})[0]
    ww, hh = target_size(w, h, settings.inference_long_side)
    work = bgr if (ww, hh) == (w, h) else cv2.resize(bgr, (ww, hh), interpolation=cv2.INTER_AREA)
    masks, fdis, scores = pipeline._forward(work, view.view, 'mask-rcnn')
    roi = full_image_result((hh, ww), strategy='R0_full')
    masks, fdis, scores, stats = postprocess_instances(masks, fdis, scores,
        roi_mask=roi.roi_mask, exclusion_mask=roi.exclusion_mask,
        params=pipeline.post_params, view_label=view.view)
    labels = np.zeros((h, w), dtype=np.int32)
    instances = []
    for i in sorted(range(len(masks)), key=lambda i: -scores[i]):
        mask = masks[i]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        available = mask & (labels == 0)
        if not available.any():
            continue
        label = len(instances) + 1
        labels[available] = label
        instances.append({'id': label, 'fdi': fdis[i], 'confidence': float(scores[i])})
    flat = labels.ravel()
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    return {'width': w, 'height': h, 'labels': np.column_stack((flat[starts], lengths)).tolist(),
            'instances': instances, 'view': view.view}


for line in sys.stdin:
    try:
        req = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            result = metadata if req.get('action') == 'info' else infer(req) if req.get('action') == 'infer' else None
            if result is None:
                raise ValueError('Unknown IOP operation')
        print(json.dumps({'result': result}), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
