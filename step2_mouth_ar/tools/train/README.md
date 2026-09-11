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
