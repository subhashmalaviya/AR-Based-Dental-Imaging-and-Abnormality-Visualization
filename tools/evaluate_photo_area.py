#!/usr/bin/env python3
"""Compare the shipped tooth detector to photo_area.jpg's hand-drawn regions.

Run from any directory: python3 tools/evaluate_photo_area.py
Requires numpy, opencv-python and the app's installed Node dependencies.
This extraction profile is specific to the supplied 880x450 image pair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Fixed mouth rectangles selected by inspecting the clean photo, not predictions.
# Coordinates are x, y, width, height in the original 880x450 image.
CROPS = {"left": (105, 52, 280, 341), "right": (495, 55, 312, 340)}
CARDS = {"left": (11, 13, 421, 396), "right": (447, 13, 421, 396)}


def region_mask(shape, rectangles):
    mask = np.zeros(shape, np.uint8)
    for x, y, w, h in rectangles:
        mask[y:y+h, x:x+w] = 255
    return mask


def extract_ground_truth(annotated):
    """Fill colored outlines, then extend interiors one pixel toward stroke centers.

    No model output or existing GT JSON is used. The colored strokes themselves
    are not the foreground target. Local closing repairs JPEG/color-mask gaps.
    """
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colors = (((h > 80) & (h < 110) & (s > 55)) |
              ((h > 130) & (h < 165) & (s > 60)) |
              ((h > 22) & (h < 40) & (s > 60) & (v > 140)))
    # Exclude turquoise card edges, title text and other non-annotation colors.
    allowed = region_mask(h.shape, [(110, 55, 282, 342), (495, 55, 315, 342)]) > 0
    strokes = (colors & allowed).astype(np.uint8) * 255
    repaired = strokes.copy()
    # Visible purple/yellow boundary junction has a gap: explicit, auditable repair.
    cv2.line(repaired, (716, 325), (726, 325), 255, 2)
    closed = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(255 - closed)
    ids = [i for i, stat in enumerate(stats) if 40 < stat[cv2.CC_STAT_AREA] < 10000]
    interior = np.isin(labels, ids).astype(np.uint8) * 255
    gt = cv2.dilate(interior, np.ones((3, 3), np.uint8))
    return gt, strokes, len(ids)


def score(gt, pred, valid):
    g, p, m = gt > 0, pred > 0, valid > 0
    tp = int(np.count_nonzero(m & g & p))
    fp = int(np.count_nonzero(m & ~g & p))
    fn = int(np.count_nonzero(m & g & ~p))
    tn = int(np.count_nonzero(m & ~g & ~p))
    def ratio(a, b):
        return a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, evaluated_pixels=tp+fp+fn+tn,
                iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn),
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                pixel_accuracy=ratio(tp+tn, tp+tn+fp+fn))


def error_overlay(clean, gt, pred, valid):
    out = clean.copy()
    g, p, m = gt > 0, pred > 0, valid > 0
    # BGR: green TP, red FP, blue FN.
    for selected, color in [(m & g & p, (65, 210, 65)),
                            (m & ~g & p, (45, 45, 245)),
                            (m & g & ~p, (245, 110, 35))]:
        out[selected] = (0.3 * out[selected] + 0.7 * np.array(color)).astype(np.uint8)
    return out


def pct(value):
    return 'n/a' if value is None else f'{100 * value:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotated', type=Path, default=ROOT / 'photo_area.jpg')
    parser.add_argument('--clean', type=Path, default=ROOT / 'photo_area_clean.jpg')
    parser.add_argument('--model', type=Path, default=ROOT / 'step2_mouth_ar/public/models/tooth_seg.onnx')
    parser.add_argument('--out', type=Path, default=ROOT / 'evaluation/photo_area')
    args = parser.parse_args()
    annotated, clean = cv2.imread(str(args.annotated)), cv2.imread(str(args.clean))
    if annotated is None or clean is None:
        raise ValueError('Both input images must exist and decode successfully')
    if annotated.shape != (450, 880, 3) or clean.shape != annotated.shape:
        raise ValueError('This extraction profile requires the aligned 880x450 photo_area image pair')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gt, strokes, regions = extract_ground_truth(annotated)
    valid = region_mask(gt.shape, CARDS.values())
    # An explicitly separate, more permissive score ignores a +/-3 px GT boundary band.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    boundary = cv2.dilate(gt, kernel) != cv2.erode(gt, kernel)
    certain = valid.copy()
    certain[boundary] = 0
    for name, data in [('ground_truth_mask', gt), ('annotation_strokes', strokes),
                       ('evaluation_region', valid), ('boundary_ignore', boundary.astype(np.uint8)*255)]:
        cv2.imwrite(str(out / f'{name}.png'), data)

    card = json.loads(args.model.with_suffix('.json').read_text())
    W, H = card['input']['width'], card['input']['height']
    manifest = dict(width=W, height=H, crops=[])
    for name, (x, y, w, h) in CROPS.items():
        crop = clean[y:y+h, x:x+w]
        rgb = cv2.cvtColor(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        (out / f'{name}.rgb').write_bytes(rgb.tobytes())
        manifest['crops'].append(dict(name=name, rectangle=[x, y, w, h]))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    subprocess.run(['node', str(ROOT / 'step2_mouth_ar/tools/infer_photo_masks.mjs'),
                    str(out / 'manifest.json'), str(args.model.resolve())], check=True)

    report = dict(schema='photo-area-pixel-evaluation/1',
                  sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [args.annotated, args.clean, args.model]},
                  ground_truth=dict(method='color strokes + 5x5 closing + enclosed-region fill + 1px dilation',
                                    status='automatically reconstructed; review overlay before treating as final labels',
                                    enclosed_regions=regions,
                                    manual_gap_repair=[[716, 325], [726, 325]],
                                    note='Binary union only; enclosed regions are not a validated tooth count'),
                  inference=json.loads((out / 'inference.json').read_text()),
                  coordinate_space='original image, 880x450; no mirroring or registration warp',
                  evaluation_cards=CARDS, boundary_ignore_radius_px=3,
                  assumptions=['All unmarked pixels inside the two cards count as background.',
                               'The clean and annotated photographs share image coordinates.',
                               'Color classes and tooth identities are not scored.'], results={})
    panels = [('Clean model input', clean)]
    gt_overlay = clean.copy()
    gt_overlay[gt > 0] = (0.4*clean[gt > 0] + 0.6*np.array([65, 210, 65])).astype(np.uint8)
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    review = annotated.copy()
    cv2.drawContours(review, contours, -1, (255, 255, 255), 1)
    cv2.imwrite(str(out / 'ground_truth_review.png'), review)
    panels.append(('Reconstructed hand-marked regions', gt_overlay))
    for variant in ['decoded', 'semantic']:
        pred = np.zeros(gt.shape, np.uint8)
        for name, (x, y, w, h) in CROPS.items():
            raw = np.frombuffer((out / f'{name}_{variant}.u8').read_bytes(), np.uint8).reshape(H, W)
            pred[y:y+h, x:x+w] = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f'prediction_{variant}.png'), pred)
        report['results'][variant] = {
            'strict': score(gt, pred, valid),
            'excluding_3px_boundary': score(gt, pred, certain),
            'per_card': {name: score(gt, pred, region_mask(gt.shape, [rect])) for name, rect in CARDS.items()}}
        overlay = error_overlay(clean, gt, pred, valid)
        cv2.imwrite(str(out / f'errors_{variant}.png'), overlay)
        s = report['results'][variant]['strict']
        panels.append((f'{variant.title()}: IoU {pct(s["iou"])} | Dice {pct(s["dice"])}', overlay))
    # Image-only summary; metrics and provenance also remain machine readable.
    rows = []
    for i in range(0, len(panels), 2):
        tiles = []
        for title, panel in panels[i:i+2]:
            tile = cv2.copyMakeBorder(panel, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(tile, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    preview = np.vstack(rows)
    preview = cv2.copyMakeBorder(preview, 0, 45, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(preview, 'Green: overlap (TP)    Red: extra model pixels (FP)    Blue: missed marked pixels (FN)',
                (20, preview.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / 'comparison.png'), preview)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Hand-marked photo evaluation', '',
             'Ground truth is the filled union of the colored outlines in `photo_area.jpg`. '
             'Inference uses only `photo_area_clean.jpg`. Existing ground-truth JSON is not used.', '',
             '| Output | IoU | Dice | Precision | Recall |', '|---|---:|---:|---:|---:|']
    for name, result in report['results'].items():
        for label, scores in [('strict', result['strict']), ('excluding 3px boundary', result['excluding_3px_boundary'])]:
            lines.append(f'| {name}, {label} | ' + ' | '.join(pct(scores[k]) for k in ['iou','dice','precision','recall']) + ' |')
    lines += ['', 'The decoded result uses the shipped model and instance decoder; semantic is the raw '
              'tooth-probability map thresholded at the model-card setting (0.5). Both are resized back '
              'with nearest-neighbor interpolation. Primary scores use every pixel in the two image '
              'cards, excluding captions and borders; predictions outside the fixed mouth crops are zero.', '',
              '## Interpretation and limitations', '',
              '- Review `ground_truth_review.png`: white contours show the extracted masks over your actual marks. '
              'This is an approximate reconstruction from JPEG strokes, not independently verified pixel annotation.',
              '- The 3px boundary-excluded score ignores uncertain edge pixels; it is a different evaluation set, '
              'not a replacement for the strict score.',
              '- All unmarked pixels in the cards are assumed background. If visible teeth were left unmarked, '
              'add their labels or mark those areas as ignored before final evaluation.',
              '- This is one composite image with two views, using fixed manual mouth crops. '
              'It does not measure face localization, live tracking, tooth IDs, or generalization to new people.',
              '- The clean image appears smoothed in tooth areas. Prefer the original unpainted photograph '
              'if this was produced by removing the drawn lines.',
              '- Pixel accuracy is recorded for completeness; the large background makes it less informative '
              'than tooth IoU and Dice.', '',
              '## Reproduce', '', '```bash', 'python3 tools/evaluate_photo_area.py', '```', '',
              '`metrics.json` records confusion counts, per-card scores, crop coordinates, decoder settings, '
              'source SHA-256 hashes and extraction details. `ground_truth_mask.png` and prediction PNGs '
              'are full-resolution binary masks (0 background, 255 tooth).', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(report['results'], indent=2))
    print(f'Artifacts: {out}')


if __name__ == '__main__':
    main()
