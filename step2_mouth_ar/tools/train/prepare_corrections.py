#!/usr/bin/env python3
"""Convert reviewed editor labels + clean photos into instance-supervised crops.

Manifest: {"samples":[{"id":"...", "group":"person/session", "split":"train|valid|test",
                       "image":"clean.png", "labels":"editable-tooth-labels.json"}]}
Paths are relative to the manifest. Assign all frames/crops from a person/session
to one group and split. No automatic train/test split of related images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read_sample(record, root):
    image_path, labels_path = root / record['image'], root / record['labels']
    doc = json.loads(labels_path.read_text())
    if doc.get('annotationStatus', '').startswith('unreviewed'):
        raise ValueError('Review and correct model drafts in the editor before importing as ground truth')
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read clean image: {image_path}')
    if doc.get('schema') != 'tooth-labels/1' or doc.get('cleanImage') != image_path.name:
        raise ValueError('Labels must name the same clean image and use tooth-labels/1')
    crop = doc.get('crop', [])
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError('Invalid crop')
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) < 1 or w * h > 4_000_000 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError('Crop is outside the clean image or exceeds 4 megapixels')
    if doc.get('width') != w or doc.get('height') != h:
        raise ValueError('Label and crop dimensions differ')
    labels = np.zeros(w * h, np.int32)
    offset = 0
    for run in doc['labels']:
        if len(run) != 2 or any(type(v) is not int for v in run):
            raise ValueError('Invalid label run')
        tooth, count = run
        if tooth < 0 or tooth > 100000 or count < 1 or offset + count > labels.size:
            raise ValueError('Invalid label run')
        labels[offset:offset + count] = tooth
        offset += count
    if offset != labels.size:
        raise ValueError('Incomplete labels')
    ids = np.unique(labels[labels > 0])
    if len(ids) > 256:
        raise ValueError('More than 256 tooth instances')
    # Compact IDs, preserving touching teeth and disconnected pieces of one ID.
    compact = np.zeros_like(labels)
    for new_id, old_id in enumerate(ids, 1):
        compact[labels == old_id] = new_id
    small_labels = cv2.resize(compact.reshape(h, w), (160, 120), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    small_rgb = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)
    return small_rgb, small_labels, {
        **record, 'image': str(image_path.resolve()), 'labels': str(labels_path.resolve()),
        'width': w, 'height': h, 'crop': crop,
        'imageSha256': hashlib.sha256(image.tobytes()).hexdigest(),
        'labelsSha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        'nativeTeeth': len(ids), 'modelResolutionTeeth': int(len(np.unique(small_labels[small_labels > 0]))),
    }


def prepare(manifest_path, out):
    manifest_path, out = Path(manifest_path), Path(out)
    records = json.loads(manifest_path.read_text())['samples']
    if not records:
        raise ValueError('No corrected samples')
    groups, hashes, ids, prepared = {}, {}, set(), []
    for record in records:
        sid, group, split = record['id'], record['group'], record['split']
        if not isinstance(sid, str) or not sid or not all(c.isascii() and (c.isalnum() or c in '-_') for c in sid) or sid in ids:
            raise ValueError('Sample IDs must be unique safe filenames')
        if not isinstance(group, str) or not group or split not in ('train', 'valid', 'test'):
            raise ValueError('Every sample needs a person/session group and train/valid/test split')
        if groups.setdefault(group, split) != split:
            raise ValueError('Data leakage: same group crosses splits')
        rgb, inst, meta = read_sample(record, manifest_path.parent)
        if hashes.setdefault(meta['imageSha256'], split) != split:
            raise ValueError('Data leakage: copies/crops of the same image cross splits')
        ids.add(sid); prepared.append((rgb, inst, meta))
    out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        chosen = [(rgb, inst, meta) for rgb, inst, meta in prepared if meta['split'] == split]
        np.savez_compressed(out / f'corrections_{split}.npz',
            img=np.stack([r[0] for r in chosen]) if chosen else np.empty((0, 120, 160, 3), np.uint8),
            inst=np.stack([r[1] for r in chosen]) if chosen else np.empty((0, 120, 160), np.int32))
    for rgb, _, meta in prepared:
        meta['rgb'] = f"{meta['id']}.rgb"
        rgb.tofile(out / meta['rgb'])
    result = {'input': [160, 120], 'samples': [r[2] for r in prepared],
              'counts': {s: sum(r[2]['split'] == s for r in prepared) for s in ('train', 'valid', 'test')},
              'scope': 'Reviewed instance labels, not anatomical tooth types. Empty masks are valid negatives. Keep test samples held out.'}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.out)['counts']))
