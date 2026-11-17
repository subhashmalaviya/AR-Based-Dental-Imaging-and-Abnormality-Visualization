# Comparing checkpoints, teeth, gums and video sequences

Open **Model & target comparisons → Open model benchmark** from Dental AR, or
`/benchmark.html`. This separate benchmark page leaves the selected live detector
unchanged. All image/video inference stays on the device.

## Direct learned U-Net versus local IOP

The two default selections are **unet-v1.1-current** and **iop-mask-rcnn-local**.
No ONNX upload or conversion is needed. The shipped U-Net uses its existing ONNX
asset. IOP loads the local PyTorch weights directly from the sibling project:

- `../IOP_teeth_segmentation/weights/maskrcnn.pt`
- `../IOP_teeth_segmentation/weights/view_classifier.pt`
- `../IOP_teeth_segmentation/.venv/bin/python`

Run `npm run dev` from `step2_mouth_ar` and open `/benchmark.html` on this computer.
**Check local IOP model** shows the resolved paths and device. Select a clean photo
and optional GT JSON, click **Use photo**, then **Compare selected checkpoints**.
Saved editor crops and patient contour JSON are supported. Prepared image folders
and video sequences also run through these same two implementations. Without GT,
accuracy remains unavailable. The historical U-Net is optional and unchecked.

The IOP worker starts automatically, loads the models once and uses the upstream
view-classification, Mask R-CNN and frozen post-processing modules. It preserves
native evaluation coordinates; each model retains its own native preprocessing.
IOP instance masks are resized back with nearest-neighbour interpolation and
confidence-order overlap resolution. Its masks are scored directly, rather than
re-rasterizing the viewer's simplified display polygons. No internal ROI is
inferred from GT. Each input is classified independently (not a five-view set).
IOP reports its weight hash, classifier hash, device and thresholds. Its timing
also includes the local Python request/response transfer; U-Net runs in browser
WASM. These timings compare the two implementations, not just network kernels.
Both models currently support teeth only, not gums.

Set `DENTAL_IOP_ROOT` or `DENTAL_IOP_PYTHON` **before starting Vite** if those paths
change. The existing IOP `IOPC_*` environment overrides remain supported. The
bridge permits same-origin requests from loopback only and does not accept a
model path from a browser request. It runs in dev and preview, not on Vercel or
another static host. It never uploads patient data or downloads weights.

CLI, from `step2_mouth_ar`:

```sh
node tools/benchmark_targets.mjs \
  tools/train/_data/vector_benchmark_20261025/manifest.json \
  ../evaluation/transformer_benchmark/unet-iop-models.json \
  ../outputs/model-target-benchmark/unet-iop-comparison.json
```

## Included checkpoints

- ToothNet-lite 1.1: current shipped weights, SHA-256
  `a7fb02a9a95eeaf9d4fe3de86b7450ac89f8d5f750f3cfcd3eb7d5c0775fd198`.
- ToothNet-lite 1.0: real historical weights restored from Git `dc01add`, SHA-256
  `c50072d07f1b44cfa53129e81b866b85e410922d52e4a50f653d9a80eee628d0`.

Both predict **teeth only**. Their model cards travel with the checkpoints.
The page and CLI also accept local ToothNet-lite and RGB dental SegFormer ONNX
checkpoints. Other network architectures need their own preprocessing/output
adapter; renaming their files or editing a card does not make them compatible.
SegFormer cards may declare nonoverlapping `toothClasses` and `gumClasses`.
An empty `toothClasses` is allowed for an actual gum-only checkpoint. The exporter
`step2_mouth_ar/tools/train/export_segformer.py` supports `--gum-classes`.
Gum predictions are decoded from the model's explicit class logits, never from
the complement of teeth. Connected teeth of the same semantic class may remain
merged; these checkpoints do not guarantee individual-tooth instance detection.

## Inputs and reports

Use a prepared `evaluation-media/1` folder, or upload a video, set a fixed mouth
crop and sample at 1 or 2 FPS. Maximum 600 samples, 200 MB uploaded video and
32 million pixels per image crop (4 million for directly sampled video crops). Browser-decodable video formats are required.
Video sampling evaluates each frame independently; it does not add a tracking
model or replace the existing processed-video export tool.

Prepared folders contain a `manifest.json`, raw interleaved RGB byte files
(width × height × 3), and optionally reviewed ground-truth JSON. Prepare these
with the existing `tools/prepare_evaluation_media.py` workflow. The media manifest
must have unique IDs, a group/session and split per sample, native width/height,
`cleanImage`, `crop`, `rgb`, and optional `labels` paths. Use relative paths for
portable folders. Videos additionally require `videoSha256`, `timeMs`,
`captureProtocol` and `sourceCrop`. Groups and sources cannot cross splits.

From `step2_mouth_ar`:

```sh
node tools/benchmark_targets.mjs \
  tools/train/_data/vector_benchmark_20261025/manifest.json \
  ../evaluation/transformer_benchmark/targets-models.json \
  ../outputs/model-target-benchmark/comparison.json
```

The example config resolves checkpoint paths relative to itself. Add entries
with `id`, `kind: "segformer"`, `model` and `card` (path to JSON). The CLI writes
JSON plus a per-image/per-frame CSV and exits unsuccessfully if any model/sample
failed. Load the JSON on the benchmark page to view tables and sequence plots.
Patient results remain ignored by Git in `outputs/model-target-benchmark/`.

Reports keep targets separate and include normalized boundary error,
Hausdorff distance, SDF coverage/pass-fail, missed/extra tooth counts,
inference time, timestamp, checkpoint hash and input/annotation hashes.
Timing includes resize, inference and label conversion, but excludes image/video
decoding and scoring. It is not end-to-end webcam FPS. SDF tolerance defaults to
0.1 native pixels, with 95% required coverage. Missing one-sided boundaries get
an aggregate error penalty of 1; both-empty masks score 0. Unsupported targets,
unlabelled inputs and runtime failures are explicit, never successful zero-error
predictions. Failed entries are excluded from the reported mean and must be
examined alongside coverage before comparing models.

Temporal change compares adjacent frame boundaries in image coordinates. It is
motion-sensitive, not registered tracking stability or accuracy. Empty-to-empty
change is 0; disappearance/appearance gets 1. New videos/crops/sizes/protocols or
failed samples reset the comparison. Plots never connect different sequences.
Ground-truth error requires independently reviewed labels for each exact frame.
No model is automatically selected or deployed from these results.

## Ground truth for gums

Legacy `tooth-labels/1` editor JSON is supported as teeth. An optional top-level
`target: "gums"` explicitly identifies a reviewed gum annotation. This does not
turn existing tooth labels into gum labels. Multi-target annotation documents use:

```json
{
  "schema": "dental-targets/1",
  "annotationStatus": "reviewed",
  "cleanImage": "example.png",
  "width": 8,
  "height": 8,
  "crop": [0, 0, 8, 8],
  "targets": {
    "teeth": {"labels": [[0, 64]], "vectorContours": []},
    "gums": {"labels": [[0, 64]], "vectorContours": []}
  }
}
```

This example deliberately contains all-background labels, not real annotations.
`labels` uses the editor's `[label, runLength]` run-length encoding. Remove a target
key if unlabelled: an explicit empty mask asserts that no target is present.
Teeth preserve instance IDs and validated vector overrides. Gum labels are united
into one tissue mask and evaluated on its raster-derived external/hole boundaries;
internal separators between gum labels are excluded. Gum vector overrides are
validated but rasterized mask boundaries define the gum metric.
Filename, dimensions and crop must match the sample exactly. For directly sampled
video, upload JSON with a `frame` object containing the exact `videoSha256`,
`timeMs`, `sourceCrop`, and `captureProtocol: "browser-seek/1"`; the image name is
`video_<first 12 hash characters>_<timeMs>ms.png`, crop `[0,0,width,height]`.
The prepared CLI pack uses its recorded protocol instead. Do not compare labels
from different decoders/crops as if they were the same frame.

## Local smoke benchmark (2026-11-17)

Both real checkpoints were run on the existing 21-sample pack: one reviewed test
image and 20 unlabelled video frames, with 19 adjacent-frame pairs. No inference
failures occurred. The reviewed image had normalized boundary error **0.0120464**
for 1.1 and **0.0159777** for 1.0. However 1.1 missed 21 reference tooth instances
versus 19 for 1.0 at one-to-one IoU ≥ 0.5. A single test image cannot establish a
winner, and unlabelled frames cannot establish accuracy. Neither model supports
gums. Reports include the exact sample/model hashes for reproduction; numerical
values from older benchmark code should not be mixed with this run.

## Internet search and availability (checked 2026-11-17)

These are candidate sources, not claims of locally tested gum detection:

| Source | What is available | Integration status |
| --- | --- | --- |
| [SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth) | RGB tooth model; weights require a signed non-commercial agreement and author request | Not downloaded; no agreement signed or email sent |
| [MouthCare intraoral sample](https://huggingface.co/datasets/MouthCare/intraoral-sample) | Gated image dataset with tooth and gum annotations | Requires approved access and its evaluation license; not downloaded |
| [Teeth/gum Roboflow dataset](https://universe.roboflow.com/chen-xiang-wei-iupgq/teeth-gum-segmentation-300-mlbn1-qybrz) | Public dataset listing with Gum/Teeth labels, CC BY 4.0 | Training-data lead; no verified ready-to-run compatible checkpoint |
| [Sankpan/vit-teeth-segment](https://huggingface.co/Sankpan/vit-teeth-segment) | SegFormer safetensors; labels background/teeth | No gum class; RGB training modality undocumented; card only says `cc`, not an unambiguous license |
| [Kutay0/gingiva-periodontal-detection](https://huggingface.co/Kutay0/gingiva-periodontal-detection) | `best.pt` file, no explanatory model card | Target classes, segmentation output, preprocessing and license unverified; not treated as a gum segmenter |
| [GingivitisSeg](https://github.com/zeynepozdemir/GingivitisSeg) | Repository linked from a gingivitis segmentation paper | Repository empty when checked; gingivitis regions are not equivalent to all gum tissue |
| [LadySakura YOLO dental checkpoints](https://huggingface.co/LadySakura/YOLO_Teeth_Dentalai) | Multiple saved epochs; documented bounding-box output | No verified mask output or model license; not scored as boundary segmentation |
| [KAIST individual tooth segmentation](https://github.com/mireiffe/individual_tooth_segmentation) | Optical-image segmentation code, checkpoint link, expert-labelled test images | Separate architecture adapter and checkpoint/license verification needed |

No usable, verified gum checkpoint or local reviewed gum GT was obtained from
these sources. The feature is ready to evaluate a compatible model once supplied;
it does not currently produce validated gum accuracy. Patient photos/videos were
not sent to any external service during the search or evaluation.

### Direct IOP verification

On 2026-11-17, the U-Net/IOP CLI configuration completed all 21 existing samples
(one reviewed image plus 20 video frames) without model or sample failures.
Input RGB hashes matched for both implementations. On that single reference,
normalized boundary error was 0.0120464 for U-Net and 0.0687896 for IOP. This is
one image, not evidence of a general winner. Local IOP used CPU inference and
its actual Mask R-CNN checkpoint SHA-256
`be6a02d5e55c580e7679e447e3e677b8028d891150e6b287de1de71e679cde95`.
