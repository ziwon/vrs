# D-Fire YOLOE-26 Detector Refresh

Status date: 2026-06-21.

This note records the Task 3 detector model refresh check for D-Fire. The goal
was to compare the current accuracy-oriented YOLOE-L baseline against a
YOLOE-26 candidate while holding prompts, thresholds, image size, policy, and
dataset fixed.

## Setup

| Field | Value |
|---|---|
| Dataset | `/data/vrs/dfire-300-stratified` |
| Items | 300 images, 424 labeled events |
| Policy | `configs/policies/dfire_eval.yaml` |
| Config | `configs/tiny.yaml` |
| Baseline | `yoloe-11l-seg.pt` |
| Candidate | `yoloe-26l-seg.pt` |
| Image size | 640 |
| Score floor | 0.20 effective detector floor |
| IoU/NMS setting | 0.50 |
| Optimization target | macro F1 |

Commands:

```bash
just dfire_dataset=/data/vrs/dfire-300-stratified eval-dfire-model-refresh
just dfire_dataset=/data/vrs/dfire-300-stratified eval-dfire-model-refresh-bbox
```

Raw reports were written under:

```text
runs/eval-dfire-model-refresh/model_refresh.json
runs/eval-dfire-model-refresh-bbox/model_refresh.json
```

## Image-Level Result

| Model | Macro F1 | Overall P | Overall R | Overall F1 | p50 ms | p95 ms | Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yoloe-11l-seg.pt` | 0.0872 | 0.5862 | 0.0401 | 0.0751 | 11.1605 | 15.2897 | 29 |
| `yoloe-26l-seg.pt` | 0.1348 | 0.5455 | 0.0708 | 0.1253 | 12.1570 | 16.3912 | 55 |

Per-class F1:

| Model | Fire F1 | Smoke F1 |
|---|---:|---:|
| `yoloe-11l-seg.pt` | 0.0000 | 0.1744 |
| `yoloe-26l-seg.pt` | 0.0453 | 0.2243 |

Decision gate output:

```text
action=adopt_candidate
best_model=yoloe-26l-seg.pt
metric_gain=+0.0476 macro_f1
p95_latency_ratio=1.0720
```

## BBox-IoU Result

This run required `bbox_iou_threshold=0.5`.

| Model | Macro F1 | Overall P | Overall R | Overall F1 | p50 ms | p95 ms | Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yoloe-11l-seg.pt` | 0.0615 | 0.4138 | 0.0283 | 0.0530 | 11.7247 | 18.2875 | 29 |
| `yoloe-26l-seg.pt` | 0.0992 | 0.4000 | 0.0519 | 0.0919 | 11.5182 | 15.5013 | 55 |

Per-class F1:

| Model | Fire F1 | Smoke F1 |
|---|---:|---:|
| `yoloe-11l-seg.pt` | 0.0000 | 0.1231 |
| `yoloe-26l-seg.pt` | 0.0302 | 0.1682 |

Decision gate output:

```text
action=adopt_candidate
best_model=yoloe-26l-seg.pt
metric_gain=+0.0377 macro_f1
p95_latency_ratio=0.8476
```

## Caveats

- Absolute recall is still poor for both models. This result supports using
  YOLOE-26L as the better candidate in the current eval harness, but it does
  not make fire/smoke detection production-quality.
- `yoloe-26l-seg.pt` failed the first FP16 segmentation-mask postprocess with a
  dtype mismatch in Ultralytics and was retried with `half=false`. The model
  refresh report records `half_fallback=true` for YOLOE-26L.
- The measured detector latency includes per-image Python and Ultralytics
  overhead, not a batched streaming benchmark.
- This is D-Fire image evidence only. Validate again on RTSP/video clips before
  promoting a default model for the live pipeline.

## Decision

For Task 3, keep both the evidence and the migration conservative:

- Use `yoloe-26l-seg.pt` as the preferred detector-refresh candidate for the
  D-Fire fire/smoke policy.
- Do not change the repository-wide default detector yet.
- Before any live default change, run the same candidate on MIVIA/RTSP clips and
  decide whether the FP32 fallback cost is acceptable in the target runtime.
