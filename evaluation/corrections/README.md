# Corrected-label baseline

The supplied `editable-tooth-labels.json` references the clean image
`Pasted image (5).png`, crop `[0, 0, 527, 500]`. Its 24 IDs are treated as
user-provided ground truth, not verified anatomical numbering.

The shipped ToothNet-lite produces 11 instances. Native-crop normalized boundary
error is **0.01204838**; only **2 of 24** GT instances have a one-to-one match at
IoU ≥ 0.5. There are 14 unmatched GT instances, one unmatched prediction, and eight
matched predictions flagged as merges. The small overall distance is not evidence
of complete tooth detection. See `baseline.json` for measurements and hashes.

This is one reference image, not a population accuracy benchmark. It is kept in
the test split; no weights or decoder thresholds were selected using this result.
Prepared data is local/ignored under `step2_mouth_ar/tools/train/_data/corrections`.
Reproduce from the repository root:

```bash
python3 step2_mouth_ar/tools/train/prepare_corrections.py \
  --manifest evaluation/corrections/source_manifest.json \
  --out step2_mouth_ar/tools/train/_data/corrections
node step2_mouth_ar/tools/evaluate_corrections.mjs \
  step2_mouth_ar/tools/train/_data/corrections/manifest.json \
  step2_mouth_ar/public/models/tooth_seg.onnx evaluation/corrections/baseline.json
```

The local clean photo is ignored by Git and must be present to reproduce. New
uploads should provide separate training, validation and test person/session
groups. Model weights are unchanged pending those data and a compatible training
checkpoint. The current environment lacks PyTorch, so the fine-tuning extension
has been syntax-checked but not run end-to-end.
