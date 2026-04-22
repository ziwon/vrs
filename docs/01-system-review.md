# VRS System Review

Status date: 2026-04-23.

This document is the current engineering review for the repository. It replaces
the older planning notes now kept under `docs/archive/`.

## Executive Summary

VRS is a two-stage video reasoning pipeline:

- Fast path: YOLOE open-vocabulary detection produces localized candidates.
- State path: persistence, cooldown, optional IoU tracking, bounded queues, and
  per-stream fanout turn frame detections into candidate alerts.
- Slow path: a VLM verifier checks candidates, returns structured JSON, and can
  be swapped between runtime backends. Cosmos-Reason2-2B is the current
  baseline, not a locked product choice.
- Output path: JSONL alerts and optional annotated MP4 with privacy blur.

The CPU-testable product surface is in good shape. The `.venv` test run passes:

```text
.venv/bin/python -m pytest -q
168 passed
```

The remaining highest-risk work is not normal unit-test coverage. It is live
GPU validation, real-dataset accuracy measurement, and production operational
hardening.

## What Is Implemented

The following components are present and covered by unit tests where practical:

- Config validation for single-stream and multistream manifests.
- Watch-policy parser with per-event detector prompts, verifier definition,
  severity, minimum score, and persistence.
- Single-stream cascade in `vrs/pipeline.py`.
- Multistream cascade with shared detector/verifier workers, bounded queues,
  queue drop counters, and shutdown diagnostics.
- Detector backend protocol with Ultralytics YOLOE and TensorRT-engine loader
  branch.
- Cosmos-shaped verifier backend protocol with transformers default, vLLM
  backend skeleton, and reserved TRT-LLM branch. The protocol should be
  generalized to Qwen-class VLMs next.
- Verifier constrained-output support through optional XGrammar.
- Verifier failure policy: `pass_through` or `reject`.
- Per-stream IoU tracking and per-track cooldown grouping.
- Evaluation harness for event-level labeled video directories.
- CI regression gate for eval-report F1 drops.
- Stage-A threshold calibration suggestions.
- Annotated-video JSONL/MP4 sinks.
- Optional YuNet face detection and Gaussian face blur for retained MP4s.

## Correctness Fixes In This Review

Two issues were fixed during this review:

- `pyproject.toml` required Python `>=3.12`, but the project `.venv` is Python
  3.11.13 and the full test suite passes there. Package metadata and README
  install instructions now state Python `>=3.11`.
- Multistream shutdown could lose an in-flight verified alert. `stop()` closed
  sink queues before the verifier worker had finished, and `SinkWorker` exited
  as soon as `stop_event` was set even if the queue still contained messages.
  Sink queues now stay open until detector/verifier workers join, and
  `SinkWorker` drains closed backlogs before exit.

## Defects And Gaps

These are the important remaining issues as of this review:

- Real accuracy is still unknown. Synthetic clips test plumbing, not model
  quality. D-Fire, Le2i/UP-Fall, UCF-Crime, or deployment-specific labeled data
  must be wired into eval before precision/recall claims are credible.
- The TensorRT detector branch is structurally present but not validated on a
  target GPU. TRT engines are hardware-specific, and prompt/class mapping must
  be frozen at export time.
- The default detector is `yoloe-11l-seg.pt`. Ultralytics publishes newer
  YOLOE-26 variants as of 2026; migration should be treated as an evaluated
  model change, not a documentation-only swap.
- The vLLM backend is structurally present but not validated against a live
  Cosmos-Reason2-2B deployment. Version pinning should happen only after a GPU
  smoke test.
- Internal benchmarking has shown Cosmos-Reason2-2B underperforming for this
  CCTV verifier role. This matches the external model-family concern: Cosmos is
  built from a Qwen3-VL-2B base, while Qwen3.5/Qwen3.6 releases are newer
  multimodal foundation models with stronger claimed visual/reasoning
  capability. Treat Cosmos as a baseline and require side-by-side eval against
  Qwen3.5/Qwen3.6-class VLMs before production model lock-in.
- TRT-LLM and speculative decoding are still not implemented.
- Cross-camera incident correlation is missing. Current grouping is only
  per-stream/per-track.
- Calibration is suggestion-only. There is no autonomous threshold application,
  cooldown, rollback, or audit policy yet.
- Event windows are global. Slow events such as smoke need per-class verifier
  context windows instead of the current global `verifier.context_window_s`.
- Runtime observability is incomplete. Queue drops are logged, but there is no
  Prometheus/OpenTelemetry endpoint, latency histogram, model error counter, or
  per-class verifier flip dashboard.
- Privacy blur is output-only. Detector and verifier still receive raw frames,
  which is correct for reasoning but should be explicitly documented for
  deployments with stricter data-minimization requirements.
- JSONL alerts are unsigned. There is no tamper-evident audit log.

## Model And Runtime Notes

The README now treats YOLOE and Cosmos-Reason2-2B as 2026-era baselines rather
than universally best choices. Deployment claims must remain conditional until
the exact model versions are pinned and tested on target hardware.

NVIDIA's 2026 Cosmos-Reason2-2B model card lists FPS=4 as the recommended video
input rate, confirms 256K context and bbox/point output support, and lists a
24 GB minimum GPU memory requirement for the reference inference path. The
project's 16 GB target is therefore a quantized/backend-specific target, not a
safe BF16 default claim.

Qwen's 2026 Qwen3.5/Qwen3.6 releases are the main challenger path. Official
Qwen materials describe Qwen3.5 as a native multimodal foundation-model leap
over Qwen3-VL-era models, and Qwen3.6 adds newer open models with vLLM/SGLang
serving guidance. For this repository, the right engineering response is not to
swap model IDs blindly; it is to add a Qwen-compatible verifier backend and run
the same alert-level eval against Cosmos and Qwen candidates.

Operationally, the bottleneck is verifier throughput, not detector VRAM. On a
quiet site the cascade can scale well; on a noisy site the verifier queue
dominates capacity and queue drops become the primary signal that the deployment
is underprovisioned or thresholds are too loose.

## Verification Checklist

Use this before declaring a deployment ready:

1. Run `.venv/bin/python -m pytest -q`.
2. Run `scripts/make_test_clips.py` and `scripts/bench.py` on the target GPU.
3. Run `scripts/eval.py` against labeled real-world footage.
4. Record per-class precision, recall, F1, verifier flip rate, queue drops, and
   verifier latency p50/p95/p99.
5. If using `detector.backend: tensorrt`, export the engine on matching target
   hardware and smoke-test class mapping.
6. If using `verifier.backend: vllm`, pin the validated vLLM version and run a
   live GPU round trip before production.
7. Before production model lock-in, compare Cosmos against at least one
   Qwen3.5/Qwen3.6-class verifier candidate on the same labeled clips and
   report F1, flip rate, malformed JSON rate, latency, and memory.
8. Confirm privacy policy: whether annotated MP4 output is retained, whether
   face blur is enabled, and whether raw frames may be passed to the verifier.
