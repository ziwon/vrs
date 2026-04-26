# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project summary

VRS is a two-stage CCTV / video-understanding cascade for a single local GPU:

- **Fast path** — Ultralytics YOLOE (open-vocabulary detector) runs every frame. Text prompts are set from the watch policy at load time; at inference YOLOE returns boxes keyed back to policy event names.
- **Slow path** — a pluggable VLM verifier checks each candidate alert and returns strict JSON (`true_alert`, confidence, optional bbox/trajectory, rationale). The current implementation uses Cosmos-Reason2-2B as a baseline, not a locked model choice. Internal benchmarks and 2026 Qwen3.5/Qwen3.6 releases make Qwen-class VLMs the priority comparison path.

Do not describe Cosmos-Reason2-2B as the final or best verifier. Treat it as the baseline currently wired through the `CosmosBackend` protocol, and keep docs/configs open to a future Qwen/OpenAI-compatible backend.

## Common commands

```bash
# Install (pick the torch wheel for your GPU arch; see README for cu128/cu121 notes)
uv python install 3.11
uv sync --python 3.11 --extra cu128    # Blackwell
# Use exactly one CUDA extra on GPU hosts: cu128 or cu121.
# Optional W4A16 extra for tiny.yaml / Jetson:
uv sync --extra cu128 --extra quant

# Run single-stream mp4 / RTSP
uv run scripts/run_mp4.py     --video <path>  --config configs/default.yaml --policy configs/policies/safety.yaml --out runs/demo
uv run scripts/run_rtsp.py    --rtsp  <url>   --config configs/default.yaml --policy configs/policies/safety.yaml --out runs/live

# Run multi-stream (shared YOLOE + shared VLM verifier, N cameras)
uv run scripts/run_multistream.py \
  --config configs/default.yaml --policy configs/policies/safety.yaml \
  --streams configs/multistream.yaml --out runs/live

# Bench on the local GPU (makes synthetic plumbing clips, then measures)
uv run scripts/make_test_clips.py --out runs/test_clips
uv run scripts/bench.py --clips runs/test_clips --out runs/bench

# Evaluate the cascade against a labeled dataset (P/R/F1 + flip-rate)
uv run scripts/eval.py --dataset <dir> --config configs/default.yaml \
  --policy configs/policies/safety.yaml --out runs/eval

# Regression-gate a new report against a committed baseline
# exit codes: 0 pass, 1 regression, 2 structural error
uv run python -m vrs.eval.ci --baseline runs/baseline/report.json \
  --current runs/eval/report.json --max-f1-drop 0.02

# Tests (CPU-only smoke tests — no CUDA / YOLOE / verifier model required)
uv run python -m pytest -q
uv run python -m pytest -q tests/test_smoke.py::test_event_state_requires_min_persist    # single test
```

Outputs always land under `runs/<name>/` (and `runs/<name>/<stream_id>/` in multi-stream). Each production run writes `alerts.jsonl` plus event images under `thumbnails/`; `annotated.mp4` is opt-in for debugging/demo only.

## Architecture

The cascade is deliberately thin glue over composable components. Read in this order to build the mental model:

1. `vrs/schemas.py` — the data contracts: `Frame → Detection → CandidateAlert → VerifiedAlert`. Everything flowing between stages is one of these dataclasses.
2. `vrs/policy/watch_policy.py` — the operator-facing YAML (`configs/policies/safety.yaml`) is the only place event classes live. `WatchPolicy` exposes both a flat YOLOE prompt list *and* a prompt-index → event-name map, so adding a class is a single YAML block.
3. `vrs/triage/yoloe_detector.py` — wraps Ultralytics YOLOE. On init it calls `model.set_classes(prompts, model.get_text_pe(prompts))`, which reparameterizes the detection head from text embeddings so per-frame inference has zero text cost. Supports single-frame `__call__` and batched `batch([...])`. Conforms to the `Detector` Protocol in `vrs/triage/backends.py`; `TensorRTYOLOEDetector` in `tensorrt_detector.py` loads a pre-exported `.engine` (via Ultralytics or NVIDIA TAO's `tao-deploy`) through the same `predict()` surface. The pipelines call `build_detector(cfg, policy, backend=cfg.get("backend", "ultralytics"))`; `scripts/export_yoloe_trt.py` wraps the Ultralytics export case.
4. `vrs/triage/event_state.py` — per-class persistence + cooldown. Holds a rolling ring buffer of recent `(Frame, detections)` so when a class fires it can hand the verifier a short clip (keyframes sampled around the peak), not just one frame. Cooldown is keyed on `(class, track_id)` so one persistent physical object raises one alert; untracked runs (`track_id is None` on every detection) collapse to a single `None` bucket and reproduce the pre-tracking per-class debounce exactly.
4b. `vrs/triage/tracking.py` — `Tracker` protocol + two implementations. `SimpleIoUTracker` is a pure-Python greedy IoU associator (no ultralytics dep), class-segregated so a jittery smoke box can't steal a fire's track id, with stale-track expiry. `NullTracker` is the no-op fallback used when `tracker.backend: none` or no `tracker:` block is configured. Multi-stream builds one tracker per stream so IDs don't collide across cameras.
5. `vrs/runtime/cosmos_loader.py` — loads the current Cosmos-Reason2-2B baseline in `bf16`/`fp16`/`w4a16`. `chat_video(..., response_schema=...)` handles BGR→PIL RGB, the Qwen3-VL `{type:"video", video:[...], fps:N}` message schema, and processor-version quirks (falls back to `images=` when the processor doesn't accept `videos=`). When `response_schema` is set it calls `_build_logits_processor` (which delegates to `vrs/verifier/constrained.build_logits_processor`) and passes the resulting XGrammar `LogitsProcessor` to `generate`. The class conforms to the currently named `CosmosBackend` Protocol in `vrs/runtime/backends.py`; alternate implementations (`VLLMCosmosBackend` in `vllm_cosmos.py`, reserved `trtllm`) plug in through `build_cosmos_backend(cfg, backend=...)` and handle their own guided-decoding surface internally — `AlertVerifier` only ever touches `chat_video`. Next model-work item: generalize this protocol or add a Qwen/OpenAI-compatible backend and compare Qwen3.5/Qwen3.6-class VLMs against Cosmos on the same labeled clips.
6. `vrs/verifier/alert_verifier.py` — builds the strict-JSON prompt, calls the configured VLM backend, parses the response with a balanced-brace JSON extractor (`_find_json_object`) that tolerates LLM prose and nested braces. Verifier failures are governed by `verifier.failure_policy`: default `pass_through` returns `true_alert=True, confidence=0.0` with a diagnostic rationale so detector hits are not silently dropped; `reject` returns `true_alert=False, confidence=0.0` for deployments that prioritize false-alarm suppression. When the optional `xgrammar` dep is installed (`uv sync --extra constrained`), `vrs/verifier/constrained.py` compiles the response schema (with `false_negative_class` restricted to policy class names + null) into a transformers `LogitsProcessor`, and a fresh processor is passed to `generate()` per `verify()` call — parse failures become structurally impossible and the balanced-brace parser becomes a safety net only.
7. `vrs/pipeline.py` (single-stream) and `vrs/multistream/pipeline.py` (multi-stream) — orchestrate everything. Single-stream is a synchronous loop; multi-stream spawns one thread per component: `DecoderThread` per source, one shared `DetectorWorker`, one shared `VerifierWorker`, and one `SinkWorker` per stream.
8. `vrs/eval/` — evaluation harness. `schemas.py` defines ground-truth events, per-class metrics, and `RunScore` (which carries P/R/F1 + verifier-flip rate + FN-flag rate). `metrics.py` does greedy one-to-one temporal matching of `alerts.jsonl` records against GT events, per class, with a configurable tolerance. `harness.py` iterates a `Dataset` + a `pipeline_factory` and returns a `HarnessResult`. `datasets/labeled_dir.py` is a generic adapter for a directory of mp4 + sidecar JSON labels — enough to run end-to-end locally without external datasets. `scripts/eval.py` is the harness CLI entrypoint. `ci.py` is the regression gate — `uv run python -m vrs.eval.ci --baseline … --current …` compares two `report.json` files and exits non-zero if any per-class or overall F1 drops by more than `--max-f1-drop`. D-Fire / Le2i adapters are the only remaining pieces of #6 (parked pending dataset acquisition).
9. `vrs/calibration/` — Stage-A self-calibration. `suggest(stream_id, class, current_min_score, window, ...)` is a stateless decision function that reads a rolling window of `(was_flipped, had_fn_flag)` entries and returns an optional `Suggestion` with a proposed new `min_score`. `Calibrator` wraps it with a per-`(stream_id, class)` window and a `CalibrationSink` (lazy-open JSONL — no empty file if nothing gets written). Every `VerifiedAlert` is fed to the calibrator; window clears on each emission so the cadence is naturally rate-limited to one suggestion per `min_sample` alerts. Loosen-arm is gated on operator-supplied `target_alerts_per_hour` (default null → tighten-only). Disabled by default (`calibration.enabled: false`). Multi-stream shares one calibrator instance — suggestions carry `stream_id` so a single JSONL at the run root stays readable.

### Multi-stream threading model

```
RTSP[i] ─► DecoderThread[i] ──► frame_q (BoundedQueue, drop_oldest)
                                  │
                                  ▼
                         DetectorWorker (batched YOLOE; owns all EventStateQueues)
                                  │
                                  ▼ candidate_q
                         VerifierWorker (VLM verifier, shared)
                                  │
                                  ▼ per-stream sink_q[i]
                         SinkWorker[i] — JsonlSink + EventThumbnailSink (+ optional VideoAnnotator)
```

Key ownership rules:
- Only the `DetectorWorker` thread touches the YOLOE CUDA model; only the `VerifierWorker` touches the verifier model/backend. This avoids CUDA context races without explicit locking.
- Per-stream `EventStateQueue` instances live *inside* the `DetectorWorker` (one per stream id) — do not try to run them from the decoder side.
- `BoundedQueue` (`vrs/multistream/queues.py`) offers `DROP_OLDEST` (default for live video), `DROP_NEWEST`, or `BLOCK`. Live-video correctness favors dropping over stalling, so a noisy camera never starves the others. `puts_dropped` is the backpressure metric.

### Configuration shape

- `configs/default.yaml` — accuracy-oriented local profile: YOLOE-L FP16 + Cosmos-Reason2-2B BF16 baseline. Validate memory on target hardware; NVIDIA's 2026 model card lists 24 GB minimum for the reference BF16 Cosmos path.
- `configs/tiny.yaml` — 8-16 GB / Jetson-oriented profile after quantized-runtime validation: YOLOE-S FP16 + `embedl/Cosmos-Reason2-2B-W4A16`.
- `configs/multistream.yaml` — decoder backend, queue sizes, batch size, stream manifest. Merged into the base config by `build_multistream_pipeline`; stream overrides in this file win.
- `configs/policies/*.yaml` — watch policy. `_validate_item` enforces severity ∈ {info,low,medium,high,critical}, `min_score ∈ [0,1]`, `min_persist_frames >= 1`, non-empty detector prompts and verifier definition.

Decoder backends (`multistream.decoder_backend`):
- `opencv` — default, CPU FFmpeg decode, works everywhere.
- `nvdec` — `cv2.cudacodec` NVDEC. Falls back to opencv with a warning if OpenCV wasn't built with CUDA.
- `deepstream` — reserved/unimplemented; the `Reader` interface is kept small so adding it is additive.

## Conventions worth knowing

- **Heavy imports are lazy.** Files that top-level import `cv2`, `ultralytics`, `transformers`, or `torch` are only loaded inside methods / workers. This is what lets the `pytest` suite run CPU-only without installing YOLOE or Cosmos. Respect this when editing — do not promote those imports to module scope.
- **Frames are BGR uint8** end-to-end (OpenCV native). Conversion to RGB happens inside `cosmos_loader._bgr_list_to_pil_rgb` only.
- **Verifier failure policy is explicit.** Default `verifier.failure_policy: pass_through` maps parser errors, missing fields, missing keyframes, and model exceptions to `true_alert=True, confidence=0.0` with a diagnostic rationale. `reject` is the named alternative and returns `true_alert=False, confidence=0.0`. Do not add a third implicit branch that silently drops the candidate.
- **Verifier model choice is not settled.** Cosmos-Reason2-2B remains the wired baseline because the prompt/schema path already exists, but internal benchmarks were weak. Qwen3.5/Qwen3.6-class VLMs should be evaluated through a served/backend abstraction before production model lock-in. Any model change must be measured with `scripts/eval.py` on identical labeled clips and report F1, flip rate, malformed JSON rate, latency, and memory.
- **Logging.** Library code uses `logging.getLogger("vrs.<module>")` with a `NullHandler`. CLI scripts call `vrs.setup_logging()` to attach the human-readable handler; library consumers attach their own.
- **SinkWorker survives per-message failures.** One `jsonl.write()` or `annotator.write()` exception in `vrs/multistream/workers.py` is logged + skipped, not propagated, so a single bad alert can't silently kill the thread and drop every subsequent alert for that stream. `tests/test_failure_paths.py` pins both the JSONL-failure and annotator-failure branches. Don't add a "log and re-raise" branch — that's the pre-fix behavior and the regression tests will catch it.
- **JSON parser** (`_find_json_object`) is a hand-rolled balanced-brace scanner that tracks string state — do not replace it with a regex. Tests in `test_smoke.py` pin its behavior for nested braces, code fences, and braces inside strings.
- **Dates/versions:** project is on Python 3.11+, `transformers>=4.46`, `ultralytics>=8.3.0`. Default detector is `yoloe-11l-seg.pt`; tiny profile uses `yoloe-11s-seg.pt`.

## When adding a new event class

Edit only `configs/policies/<your>.yaml` — add a `watch:` block with `name`, `detector` (list of noun phrases), `verifier` (one sentence), `severity`, `min_score`, `min_persist_frames`. No embeddings to recompute, no prompt bank to score, no retraining. The new name appears in `alerts.jsonl` as `class_name`.
