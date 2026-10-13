# Hand-marked photo evaluation

Ground truth is the filled union of the colored outlines in `photo_area.jpg`. Inference uses only `photo_area_clean.jpg`. Existing ground-truth JSON is not used.

| Output | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| decoded, strict | 66.23% | 79.68% | 78.61% | 80.79% |
| decoded, excluding 3px boundary | 77.27% | 87.18% | 87.93% | 86.44% |
| semantic, strict | 68.15% | 81.06% | 78.14% | 84.21% |
| semantic, excluding 3px boundary | 79.57% | 88.62% | 87.54% | 89.73% |

The decoded result uses the shipped model and instance decoder; semantic is the raw tooth-probability map thresholded at the model-card setting (0.5). Both are resized back with nearest-neighbor interpolation. Primary scores use every pixel in the two image cards, excluding captions and borders; predictions outside the fixed mouth crops are zero.

## Interpretation and limitations

- Review `ground_truth_review.png`: white contours show the extracted masks over your actual marks. This is an approximate reconstruction from JPEG strokes, not independently verified pixel annotation.
- The 3px boundary-excluded score ignores uncertain edge pixels; it is a different evaluation set, not a replacement for the strict score.
- All unmarked pixels in the cards are assumed background. If visible teeth were left unmarked, add their labels or mark those areas as ignored before final evaluation.
- This is one composite image with two views, using fixed manual mouth crops. It does not measure face localization, live tracking, tooth IDs, or generalization to new people.
- The clean image appears smoothed in tooth areas. Prefer the original unpainted photograph if this was produced by removing the drawn lines.
- Pixel accuracy is recorded for completeness; the large background makes it less informative than tooth IoU and Dice.

## Reproduce

```bash
python3 tools/evaluate_photo_area.py
```

`metrics.json` records confusion counts, per-card scores, crop coordinates, decoder settings, source SHA-256 hashes and extraction details. `ground_truth_mask.png` and prediction PNGs are full-resolution binary masks (0 background, 255 tooth).
