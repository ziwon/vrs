# Jetson Orin Qwen2.5-VL-7B VLM Verifier Smoke Plan

Status date: 2026-05-14.
Status: planned VRS smoke profile.

This note defines the first VRS-level 7B VLM verifier smoke for Jetson Orin 64GB. It is intentionally separate from the `efficient-llm-inference-systems/week03` text-only 7B experiment: that experiment validated transformer serving capacity with `Qwen/Qwen2.5-7B-Instruct`, while this VRS profile validates the application slow path with a vision-language verifier.

## Scope

| Field | Value |
|-------|-------|
| Host target | Jetson Orin 64GB |
| Detector | YOLOE-S segmentation, `yoloe-11s-seg.pt`, FP16 |
| Verifier | `Qwen/Qwen2.5-VL-7B-Instruct`, BF16, local `transformers` backend |
| Config | `configs/jetson-qwen25vl7b.yaml` |
| Smoke policy | `configs/policies/falldown_orin_smoke.yaml` |
| Baseline comparison | `docs/benchmarks/jetson-orin-qwen3vl2b-vlm-2026-05-12.md` |

## Why Qwen2.5-VL-7B

The current Qwen3-VL lineup used by this repo has a validated 2B smoke profile and an 8B-class option, but not a 7B model ID matching `Qwen3-VL-7B`. For a literal 7B VLM smoke, use `Qwen/Qwen2.5-VL-7B-Instruct`. If the goal is to stay in the Qwen3-VL family, create a separate `Qwen/Qwen3-VL-8B-Instruct` profile and branch.

## Conservative First-Pass Profile

The 7B VLM smoke starts smaller than the optimized 2B profile:

```yaml
verifier:
  model_id: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bf16
  device: cuda
  context_window_s: 2.0
  keyframes: 1
  clip_fps: 1
  max_frame_width: 336
  max_new_tokens: 96
  temperature: 0.0
  timeout_s: 180.0
```

Rationale:

- `keyframes: 1` limits visual-token growth for the first load and verify path.
- `max_frame_width: 336` keeps the visual processor workload lower than the 2B 448 px profile.
- `max_new_tokens: 96` is enough for the structured verifier JSON while avoiding the 64-token truncation risk seen in the 2B latency matrix.
- `timeout_s: 180` gives the first 7B VLM run room to download, warm up, and generate without being confused with a pipeline failure.

## Suggested Commands

```bash
cd ~/Documents/yp/vrs
source .venv-jetson/bin/activate

python -m pytest tests/test_smoke.py -q

python scripts/run_rtsp.py \
  --config configs/jetson-qwen25vl7b.yaml \
  --policy configs/policies/falldown_orin_smoke.yaml \
  --rtsp-url rtsp://<your-host>:8554/falldown \
  --runtime-s 30 \
  --out runs/orin_qwen25vl7b_smoke
```

If the first run loads but verifier latency is too high, keep the same model and reduce only one variable at a time:

1. `max_frame_width: 224`
2. `max_new_tokens: 64`, but only if JSON parsing remains stable
3. `keyframes: 1` must remain fixed for the first smoke series

## Measurements To Record

Record these before calling the profile smoke-tested:

- model load time
- detector latency min/median/mean/max
- candidate count and verifier call count
- verifier latency p50/p95, or every verifier call if the run is short
- final PyTorch memory used and visible total memory
- verdict, rationale, JSON parse status, and failure-policy behavior
- whether Qwen2.5-VL accepts the same `videos=[pil_frames]` path as Qwen3-VL in `vrs/runtime/cosmos_loader.py`

## Expected Interpretation

The text-only 7B experiment in `efficient-llm-inference-systems/week03` showed that 7B BF16 weights can fit on Orin and that KV memory is not the first practical limiter at 1K context and modest batch. This VRS 7B smoke answers a different question: whether a 7B vision-language verifier is useful inside the detector-candidate-verifier cascade.

The expected bottleneck is verifier latency, not memory capacity. The 2B VLM optimized verifier call was already around 8 seconds end to end; a 7B VLM may be substantially slower unless the visual-token budget is kept small or the backend is changed to a served/optimized runtime.
