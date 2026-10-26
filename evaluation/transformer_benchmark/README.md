# Vector and video benchmark — 25 October 2026

Implemented: editable cubic vector GT borders, SVG non-scaling 1 CSS pixel strokes, direct curve-to-prediction Euclidean boundary distances, local video processing/export and frame extraction, and a local SegFormer ONNX adapter shared by live, photo and video detection. The shipped U-Net remains the default. **No Transformer was selected or deployed as a measured improvement.**

## Results actually measured

`baseline.json` and `baseline.svg` contain a real run of the shipped U-Net on the existing corrected reference plus 20 samples from the supplied 19.33-second video. The video uses a fixed native crop `[30,170,420,570]` at 1 FPS. It is an offline fixed-crop diagnostic, not a reproduction of the live face-landmark/tracking pipeline.

- Existing corrected reference: normalized vector boundary error **0.0135575512**. Only **2 of 24** annotated teeth match at IoU ≥ 0.5; 13 regions were predicted. Low overall distance does not mean complete tooth detection.
- Video: **20 samples**, **19 temporal boundary changes**, **0 GT accuracy measurements**. Predicted region counts range from 3 to 14; these are predictions, not verified tooth counts.
- No annotated validation split exists. The existing labelled reference is a **test** example and is not used to select a model.
- Browser interaction/video decoding has not been visually verified in this environment. CLI inference, geometry/decoder/metadata tests and production build are the available checks.

The new vector scores use pixel-cell edges and continuous curves; older pixel-center ASSD scores are a different definition and are not directly comparable.

## Model survey

Public primary sources were searched for RGB intraoral tooth segmentation, Transformer checkpoints, SegFormer, TeethSEG, SAM-based pipelines and 3D/X-ray alternatives. This is a scoped survey of identifiable candidates, not a claim to enumerate every private or published model. Published IoU scores from other datasets cannot determine the best ASSD on these images.

| Candidate | Relevant evidence and availability | Decision for this project |
| --- | --- | --- |
| [SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth) | RGB intraoral pipeline: YOLO localization/enumeration + Light HQ-SAM Transformer segmentation. Code MIT; pretrained weights require a signed non-commercial agreement and author-provided access. | Strong domain fit for a future benchmark. Weights were not available; not run. The code license does not grant the weight license. |
| [TeethSEG](https://zoubo9034.github.io/TeethSEG/) | ViT/CLIP-based dental segmentation; dataset includes rendered arches, plaster views and RGB intraoral photos. Official project page still labels code as coming soon. | Relevant research candidate. No usable official checkpoint verified in this session; not run. |
| [Sankpan/vit-teeth-segment](https://huggingface.co/Sankpan/vit-teeth-segment) | Public SegFormer listing, about 3.72M parameters. Model card content is empty and license is only tagged `cc`. | Modality, tooth label mapping and exact license could not be verified. Not adopted or presented as RGB-ready. |
| [vimassaru/segformer-b0-finetuned-teeth-segmentation](https://huggingface.co/vimassaru/segformer-b0-finetuned-teeth-segmentation) | Dental checkpoint tagged X-ray, with FDI classes and its own evaluation metrics. | Excluded from RGB camera model selection because the imaging modality differs. |
| [SegFormer B0–B5](https://github.com/NVlabs/SegFormer) | General semantic segmentation Transformer family. | Supported by the new ONNX adapter after RGB dental fine-tuning and export. Generic checkpoints are not tooth detectors; same-class touching teeth can merge. |
| Segmenter, Swin/SwinV2, BEiT and ViT-Adapter | Compared as Transformer baselines in the [TeethSEG paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zou_Teeth-SEG_An_Efficient_Instance_Segmentation_Framework_for_Orthodontic_Treatment_based_CVPR_2024_paper.html). | Architectural candidates; compatible RGB dental checkpoints were not established. Paper rankings are not rankings on this user's data. |
| [SAM 2 / 2.1](https://github.com/facebookresearch/sam2) | General promptable image/video segmentation with public checkpoints. | Potential prompt-assisted annotation/tracking candidate, not an autonomous tooth enumerator. Supplying GT boxes/points would change the benchmark and leak annotations into inference. |
| [TSegFormer](https://github.com/huiminxiong/TSegFormer) | Geometry-guided Transformer for 3D intraoral point clouds. | Excluded: 3D scan inputs do not match RGB images/video. |
| [MaskDINO intraoral research fork](https://github.com/zykev/maskdino) | Describes intraoral caries/orthodontic training with several encoders. | Additional research lead; no verified ready-to-use compatible RGB tooth checkpoint/license established here. |

Neither a lower boundary score nor a Transformer architecture guarantees all teeth will be found. To select a winner, obtain usable RGB weights and review labels from independent sessions. Benchmark the same crops, report missed/extra teeth, and confirm the chosen candidate on held-out data and live-device latency.

## Editor use

1. Load the matching clean/marked images as before. Select a tooth.
2. Choose **Freehand vector border**, hold and drag around the complete tooth, then release. The continuous stroke becomes a smooth closed curve automatically.
3. Redraw the complete outline to edit the selected tooth, or Undo. **Smooth existing border** fits an existing single contour; inspect the fit before keeping it. Holes/disconnected contours remain exact pixel-edge vectors until redrawn.
4. Borders stay 1 CSS pixel wide at every zoom. Grayscale/contrast affect only the photograph. Save JSON to retain vector geometry and pixel labels together.
5. Brush edits convert vector overrides back to exact pixel-edge contours. Undo restores both curves and pixels. Overwriting another tooth invalidates its old curve so scoring cannot retain a stale border.

ASSD is the sum of both directed distance integrals divided by total border length, then divided by the native crop diagonal. Cubic curves are flattened within 0.1 px and integrated with samples spaced at most 0.5 px; nearest distances are Euclidean point-to-segment distances. It is a numerical approximation, not exact analytic Bézier integration. Tooth separators remain included. Missing boundaries are unavailable, never zero. Per-tooth pairing still uses one-to-one IoU matching; that does not change the boundary distance formula.

## Video use

Open **Upload Video — teeth detection & frame extraction** and choose a video. Optionally select a fixed mouth crop that contains the mouth throughout its motion. **Process video — detect teeth** runs the selected model on every output frame at 10/15/24/30 FPS and produces a playable/downloadable, silent WebM. Only cyan tooth outlines are added to the full original image. Inference speed never changes playback timestamps. This replaces the sampled-frame table/plot as the main browser workflow; CLI benchmark plots below remain available.

To calculate frame error, pause or scrub the original video at any time (or enter seconds), then **Use this frame for comparison**. No prior video processing is required. The clean native crop opens in the existing photo comparison editor. Draw ground truth freehand, restore labels, or upload a hand-marked copy of the same frame, then **Detect & compare**. **Download clean frame (PNG)** exports the same crop without overlays for external annotation. GT JSON retains source video SHA-256, requested seek timestamp and crop. Browser seek frames and OpenCV frame-index samples can differ; their GT files must not be silently interchanged.

Export uses VP8 WebCodecs when available, or a canvas WebP/VP8 keyframe fallback, with offline WebM timestamps/duration/keyframe cues. Unsupported browsers get an explicit encoder error. Limits: 200 MB source, 10 minutes, UHD 4K; encoded output is capped at 250 MB. Frames are processed sequentially; decoded frames/masks are not all retained in memory. The image editor accepts crops up to 4 megapixels. Stop discards an incomplete video. Automated export checks decode and seek real VP8 and WebP-derived WebM with FFmpeg; interactive browser export remains unverified in this environment.

## Reproduce CLI baseline

From repository root (use a new output folder if it already exists):

```bash
python3 step2_mouth_ar/tools/prepare_evaluation_media.py \
  --video 'testvideophoto/WhatsApp Video 2026-10-20 at 16.40.31.mp4' \
  --fps 1 --crop 30 170 420 570 \
  --photos evaluation/corrections/source_manifest.json \
  --out step2_mouth_ar/tools/train/_data/vector_benchmark_20261025
node step2_mouth_ar/tools/benchmark_models.mjs \
  step2_mouth_ar/tools/train/_data/vector_benchmark_20261025/manifest.json \
  evaluation/transformer_benchmark/models.json \
  evaluation/transformer_benchmark/baseline.json
```

The preparer preserves native RGB crops/PNG frames and requested/decoded timestamps. For CLI frame GT, annotate the extracted PNG in the photo editor, retaining its filename and whole-image crop, and set that sample's `labels` path in the media manifest. Do not use the model's draft labels as GT. Assign all frames/crops from one session to one split. The report includes per-sample hashes, errors, instance matching, timing, an SVG plot and explicit selection status. CLI uses half-pixel bilinear RGB resizing; browser canvas resizing can differ slightly in filtering. Use the same evaluation runner for all candidates.

Selection uses only `valid` samples, identical coverage across models, lower normalized error and no increase in missed/extra instances at IoU ≥ 0.5. A missing prediction receives a penalty of 1 in aggregate rankings; per-frame distance remains null. Unlabelled temporal values never enter accuracy rankings. Nothing is deployed automatically.

## Load/benchmark a compatible Transformer

`step2_mouth_ar/tools/train/export_segformer.py` exports an already obtained **RGB dental-finetuned** local Hugging Face SegFormer (safetensors). It downloads no weights and requires PyTorch, Transformers, safetensors and ONNX, which are not installed in this workspace. The exporter has been syntax-checked, not run on a real Transformer checkpoint.

```bash
python3 step2_mouth_ar/tools/train/export_segformer.py \
  --checkpoint /path/to/local-rgb-dental-segformer \
  --out /path/to/segformer.onnx --name 'Dental SegFormer candidate' \
  --source-url https://example.org/actual-checkpoint \
  --license 'actual checkpoint license' --rgb-intraoral --tooth-classes 1
```

Specify the actual tooth class IDs; `1` is an example, not a universal mapping. The model card records size/normalization/classes/area threshold/hash. Decoder uses NCHW raw logits, bilinear logit resizing before argmax, and separate 4-connected components per tooth class. It does not invent individual boundaries inside a connected binary tooth class. Live upper/lower assignment is geometric, not an anatomical tooth-type prediction.

Add a model entry `{ "id": "candidate", "kind": "segformer", "model": "path.onnx", "card": "path.json" }` to a benchmark config (paths relative to config). After measuring improvement, load the ONNX and card through **Load a Transformer candidate**. The same adapter is used for live detection, uploaded images and videos. No external service receives photos. Unsupported cards, wrong output dimensions and mismatched optional model hashes fail explicitly.
