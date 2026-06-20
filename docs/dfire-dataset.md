# D-Fire Evaluation Dataset

D-Fire is a public image dataset for fire and smoke detection. Its labels use
YOLO rows with normalized coordinates:

```text
<class_id> <center_x> <center_y> <width> <height>
```

The adapter converts those YOLO boxes to VRS alert-box coordinates:
`x_min, y_min, width, height`, still normalized to `0..1`.

Use a small local subset for VRS evaluation. Do not commit the full dataset.
The default `just eval-dfire` recipe uses the generic fast detector-only image
path and the smaller YOLOE-S config. Any future dataset adapter that yields
still-image items can use the same path; D-Fire is just the first committed
adapter.

## Layout

```text
datasets/dfire-mini/
  images/
    sample-001.jpg
    sample-002.jpg
  labels/
    sample-001.txt
    sample-002.txt
```

Images with an empty or missing label file are treated as quiet samples. Set
`require_labels=True` when constructing `DFireDataset` if every image must have
an explicit label file.

## Class Map

The built-in adapter uses the common D-Fire YOLO mapping:

```text
0 smoke
1 fire
```

Pass a custom `class_map` to `DFireDataset` if your downloaded split uses a
different class order.

## Run Detector-Only Eval

```bash
python scripts/eval.py \
  --dataset datasets/dfire-mini \
  --dataset-format dfire \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --mode detector_only \
  --out runs/eval-dfire
```

By default, scoring is image-level class presence: a fire alert at `0.0s`
matches a fire label in the image. To require box localization, pass an IoU
threshold:

```bash
python scripts/eval.py \
  --dataset datasets/dfire-mini \
  --dataset-format dfire \
  --mode detector_only \
  --bbox-iou-threshold 0.5 \
  --out runs/eval-dfire-bbox
```

## Sweep Fast-Path Thresholds

Use the sweep command before promoting fire/smoke threshold changes to a live
RTSP policy. It runs YOLOE once at the lowest requested score floor, then
rescoring cached detections across the threshold grid:

```bash
just dfire_dataset=/data/vrs/dfire-300-stratified \
  eval_policy=configs/policies/dfire_eval.yaml \
  eval-dfire-sweep
```

Outputs:

```text
runs/eval-dfire-sweep/sweep.json
runs/eval-dfire-sweep/best_policy.yaml
runs/eval-dfire-sweep/best_config.yaml
```

Use `eval-dfire-sweep-bbox` when you want the sweep to require localization
quality via `dfire_iou`.

## Sweep Prompts And Models

If threshold tuning does not recover recall, sweep the detector prompt bank and
YOLOE model variant:

```bash
just dfire_dataset=/data/vrs/dfire-300-stratified \
  eval_policy=configs/policies/dfire_eval.yaml \
  eval-dfire-prompt-sweep
```

Outputs:

```text
runs/eval-dfire-sweep-prompts/sweep.json
runs/eval-dfire-sweep-prompts/best_policy.yaml
runs/eval-dfire-sweep-prompts/best_config.yaml
```

The generated policy/config pair should be validated with `eval-dfire` before
any live RTSP policy promotion.

## Report Sections

`scripts/eval.py` writes a versioned `report.json`. The legacy `metrics`
section remains for CI compatibility. Newer reports also split quality by
pipeline stage:

- `detector_quality` — populated for `--mode detector_only` runs.
- `full_cascade_quality` — populated for verifier-enabled full-cascade runs.

For D-Fire detector-only bbox runs, use `detector_quality` as the primary
precision/recall/F1 section and expect `full_cascade_quality` to be `null`.
