# Step 3 test plan — tooth detection, tracking and recording

Every number reported for this project must come from a recording made with
the app and scored with the evaluation tool. This plan says what to record and
how to score it, so results are repeatable and comparable between detector
versions.

## 1. Equipment and settings

* Phone (front camera) or laptop webcam; note the device and browser.
* App settings: detector = **Learned U-Net**, tooth smoothing = Balanced,
  detect every N = 1. Record **raw** video with **metadata on** (raw video is
  what ground truth is annotated on; annotated video is for presentations).
* One subject per session; note lighting and background.

## 2. Recordings (≈10–15 s each)

| # | Condition | How | Expected visible teeth |
|---|---|---|---|
| 1 | Straight face, teeth together | look at camera, lips apart, teeth closed | count before recording |
| 2 | Natural smile | normal smile | 〃 |
| 3 | Wide open mouth | open as far as comfortable | upper + lower arches |
| 4 | Slightly open mouth | lips ~1 cm apart | often upper incisors only |
| 5 | Head turned left | ~20–30° yaw, keep smiling | left side foreshortened |
| 6 | Head turned right | ~20–30° yaw | right side foreshortened |
| 7 | Head tilted | ~15° roll, then pitch back | |
| 8 | Near camera | mouth fills ~40% of frame width | |
| 9 | Far from camera | mouth ~10% of frame width (arm's length+) | |
| 10 | Different lighting | dim room, window side-light, warm lamp | |
| 11 | Partial visibility | lip partly covering teeth, one side only | |
| 12 | Different background | plain wall vs cluttered/bright background | |

Name files by condition, e.g. `T05_turn_left_dental_tracking_20260911_101530.webm`
plus its `.json`.

## 3. Scoring each recording

1. Open `eval.html` (served with the app; offline, nothing uploaded).
2. Load the video and its metadata JSON.
3. Annotate **at least 10 frames** spread across the clip (use *Next frame with
   mouth*, skip ~1 s between frames). For each visible tooth click the crown
   centre and set upper/lower; tick *partial* for teeth cut by the lip, in deep
   shadow, or whose borders you cannot resolve; draw *ignore* regions where teeth
   are visible but cannot be told apart.
4. Read the metrics panel and click **Export report** (and **Export ground
   truth** so the annotation can be reused for the next model version).

## 4. What to report per test

| Metric | Source |
|---|---|
| expected visible teeth (mean per frame) | GT teeth / annotated frames |
| detected teeth (mean per frame) | detections / annotated frames |
| detection rate (recall) | report `accuracy.recall` |
| missed teeth | `accuracy.fn` |
| false positives | `accuracy.fp` |
| duplicate detections | `accuracy.duplicates` |
| merged teeth | `accuracy.merges` |
| average confidence | recording statistics / metadata |
| FPS | `recording.meanFps` |
| tracking stability | `recording.idContinuity` (Jaccard of IDs between frames) |

Batch summary of many recordings (Node):

```bash
node tools/analyze_recording.mjs recordings/*.json            # stats only
node tools/analyze_recording.mjs rec.json --gt rec_gt.json    # + accuracy
```

## 5. Rules

* Do not report accuracy for conditions that were not recorded and annotated.
* Report *all teeth* and *clear teeth only* separately.
* Keep the ground-truth files: re-scoring an improved model on the same
  annotations is the only fair comparison.
