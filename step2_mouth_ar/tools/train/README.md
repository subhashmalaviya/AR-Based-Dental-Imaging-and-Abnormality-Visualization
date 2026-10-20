# Tooth model — data, training and licences

This folder rebuilds `public/models/tooth_seg.onnx` (ToothNet-lite) from
public data. Nothing here runs in the app; the app only loads the exported
ONNX file and its model card `tooth_seg.json`.

| Script | What it does |
|---|---|
| `roi_geometry.py` | Line-for-line Python port of the app's mouth rectification (`MouthTracker.js` + `MouthROI.js`), so training crops match what the app sees. |
| `fetch_easyportrait.py` | Downloads only the EasyPortrait images that contain teeth (plus closed-mouth negatives) by reading the 28 GB ZIP's central directory over HTTP range requests. |
| `build_dataset.py` | DentalAI polygons → per-tooth instance crops; EasyPortrait → mouth-rectified crops with a teeth mask; also writes the held-out test fixtures used for the README numbers. |
| `make_valid_fixtures.py` | DentalAI *validation* crops, used only to tune decoder thresholds. |
| `train_tooth_model.py` | Trains ToothNet-lite (CPU is fine, ~40 min), exports ONNX, verifies it against PyTorch, writes the model card. |
| `make_video_rois.py` | Rectified crops + annotation grids from real recordings (evaluation). |

## Data sources and licences

* **DentalAI** — Pawan Valluri (2023), *Dentalai Computer Vision Project*,
  <https://www.kaggle.com/datasets/pawanvalluri/dental-segmentation>, mirror
  <https://datasetninja.com/dentalai>. **CC BY 4.0.** Used: the `Tooth` class
  polygons only (caries / cavity / crack labels are ignored).
* **EasyPortrait** — Kvanchiani, Petrova, Efremyan, Nagaev, Kapitanov (2023),
  *EasyPortrait – Face Parsing and Portrait Segmentation Dataset*,
  <https://arxiv.org/abs/2304.13509>,
  <https://huggingface.co/datasets/gofixyourself/EasyPortrait>.
  **Creative Commons Attribution-ShareAlike 4.0 (variant — see the dataset's
  licence file).** Used: the `TEETH` mask class of ~1.9 k images.

Because EasyPortrait is ShareAlike, the trained weights (`tooth_seg.onnx`)
are released under **CC BY-SA 4.0** with attribution to both datasets. The
code in this repository is unaffected.

Datasets are **not** redistributed in this repository; the scripts download
them from the sources above.

## Reproduce

```bash
python fetch_easyportrait.py --stats mask_stats.json --out _data/ep     # mask_stats from annotations.zip
python build_dataset.py --dentalai-tar dentalai.tar --ep-dir _data/ep \
    --ep-ann annotations.zip --model ../../public/models/face_landmarker.task --out _data/ds
python make_valid_fixtures.py --npz _data/ds/da_valid.npz --out _data/ds/fixtures_da_valid
python train_tooth_model.py --data _data/ds --out _data/model --epochs 35
node ../tune_decoder.mjs --rois _data/ds/fixtures_da_valid --gt _data/ds/fixtures_da_valid/gt.json \
    --model _data/model/tooth_seg.onnx
```

## Learn from corrected editor labels

Save **editable tooth labels (JSON)** from the upload editor and keep the exact
clean image named by that file. An edited preview/screenshot is not a training
image. Create a manifest with a sample per reviewed crop:

```json
{"samples": [{"id": "session01_front", "group": "person01", "split": "train",
  "image": "photos/front.png", "labels": "labels/front.json"}]}
```

Paths are relative to the manifest. Assign a person and all related video frames
to one `group` and one split (`train`, `valid`, or `test`). Add distinct groups
for validation and final testing. The importer rejects group leakage and identical
source-image copies crossing splits, validates label runs and crop coordinates,
preserves instance IDs with nearest-neighbor resizing, and reports teeth lost at
model resolution. Review those cases before training. Negative images with an
empty reviewed mask are supported. It cannot automatically recognize the same
person in different photos: group assignment must be accurate.

```bash
python prepare_corrections.py --manifest my_samples.json --out _data/corrections
node ../evaluate_corrections.mjs _data/corrections/manifest.json \
    ../../public/models/tooth_seg.onnx _data/baseline.json
python train_tooth_model.py --data _data/ds --corrections _data/corrections \
    --init _data/model/tooth_seg_best.pt --out _data/candidate \
    --epochs 10 --lr 0.0001 --correction-fraction 0.2 --lowlight
```

Fine-tuning requires PyTorch, a compatible ToothNet `.pt` checkpoint, the original
datasets for replay, and separate corrected train/validation groups. The shipped
ONNX is an inference artifact, not a loadable PyTorch state dictionary. The
correction fraction reserves part of each epoch for reviewed labels while
retaining original-domain samples. Both original and corrected validation
metrics participate in checkpoint selection; corrected test data is never loaded
by training. Export candidates to a separate folder, not over the shipped
weights. Run `evaluate_corrections.mjs` on candidate outputs and inspect native
normalized boundary error, missed teeth, merged instances, and false positives.
Do model/threshold selection on validation groups, then report the test set once.
The evaluator uses OpenCV area-resized RGB crops; browser resampling may differ
slightly. Neither path applies editor grayscale/contrast to inference.

These masks supervise tooth regions, centers and separation, not anatomical tooth
classes. Reliable coverage of incisors, canines, premolars and molars requires
diverse reviewed examples, including posterior teeth, lighting and occlusions;
explicit anatomical classification also needs type labels. A small boundary error
alone can hide severe merges. Verify temporal ID consistency and jitter on
held-out videos before adopting new weights.

The current saved reference is prepared as **test-only**, with its measured
baseline under `evaluation/corrections/` at the repository root. More independent
groups and the training checkpoint are needed before validated fine-tuning.

### Draft annotations for clean uploads

`tools/draft_upload_labels.mjs` can produce editor-compatible `tooth-labels/1`
annotation drafts from a prepared `unreviewed-upload-session/1` manifest. Run it
from `step2_mouth_ar` with the manifest and ONNX model paths. Each sample provides
`id`, `image` (native clean crop), `rgb` (160 × 120 RGB bytes), `width`, and
`height`; the manifest declares `input: [160, 120]`. It writes native-size RLE
labels and a binary PGM mask, refusing to overwrite existing files. Convert PGM
to PNG for browser upload, load the matching clean crop and binary mask, extract,
then restore the draft JSON to retain separate tooth IDs.

These files are marked `unreviewed_model_draft`, carry the model hash/decoder
settings, and are rejected by `prepare_corrections.py`. Review every region,
correct missing/merged teeth in the editor and save a reviewed export before
using it in a supervised dataset. Never report model-versus-own-draft scores as
accuracy. Keep all photos and video frames from one session in a single split.
