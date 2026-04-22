# Planning Notes — code-accurate as of v0.3

This document is a current-state engineering plan, grounded in the code that is
actually in the repository today. It separates three things that were previously
mixed together:

1. what is already implemented,
2. what is currently broken or misleading,
3. what should land next.


## Status — 2026-04-23

Tier 0 is complete: all five immediate-correctness items have landed.

Tier 1 #6 (real-dataset evaluation harness) is substantively complete for
everything that doesn't depend on external datasets:

- schemas, metrics (per-class P/R/F1 + verifier-flip rate + FN-flag rate),
- the video-iteration harness + `scripts/eval.py` CLI,
- a generic labeled-directory dataset adapter,
- a CI regression gate (`python -m vrs.eval.ci --baseline … --current …`)
  that fails the build on per-class or overall F1 drops.

The remaining pieces of #6 — D-Fire and Le2i adapters — need the actual
datasets in hand to be verified, so they are parked behind dataset
acquisition and not blocking other tracks.

Tier 1 #7 (constrained decoding) also landed (2026-04-22): when the
optional `xgrammar` dependency is installed, the verifier compiles a JSON
schema (whose `false_negative_class` enum is built from the active watch
policy) into a transformers `LogitsProcessor` and passes it to Cosmos's
`generate()`. The parser fallback remains in place for hosts without
xgrammar, but is now a strict safety net rather than a first line of
defense.

Tier 1 #8 (in-stream tracking + per-track grouping) landed (2026-04-22):
a pure-Python `SimpleIoUTracker` (no new heavy deps) assigns stable
`track_id`s to YOLOE detections, and `EventStateQueue` keys cooldown on
`(class, track_id)` instead of just class. One persistent fire now
produces one alert per track; two simultaneous fires each get their own.
Untracked runs (tracker disabled) fall back to exactly the old per-class
cooldown behavior. Cross-camera incident correlation (the other half of
improvement #4) is deferred to #8.B.

Tier 1 #9 (self-calibration Stage A — log-only) landed (2026-04-22):
`vrs/calibration/` keeps a rolling flip-rate window per
`(stream_id, class_name)` and every verifier verdict feeds it. When the
flip rate crosses thresholds the calibrator emits a suggestion to
`<out>/calibration_suggestions.jsonl` with the current and proposed
`min_score`, the flip rate that triggered it, and a short reason. The
loosen arm is gated on an operator-provided `target_alerts_per_hour`
(default `null` → tighten-only is the safe default). Disabled by default
(`calibration.enabled: false`); opt in per deployment. Stage B
(autonomous apply with caps + per-class cool-downs) reuses the same
stateless `suggest()` function.

Tier 1 #10 (verifier latency — backend abstraction) partially landed
(2026-04-22): the verifier now talks to a `CosmosBackend` Protocol
instead of the concrete transformers class, a `build_cosmos_backend()`
factory picks `transformers` / `vllm` / `trtllm` at construction, and
constrained-decoding setup was moved from `AlertVerifier` into each
backend's `chat_video` so each engine can map the JSON schema to its
native guided-decoding surface (XGrammar logits processors for
transformers; `GuidedDecodingParams` for vLLM). A vLLM backend skeleton
ships at `vrs/runtime/vllm_cosmos.py` — structurally complete and
Protocol-conformant (pinned by tests with a faked vllm module) but
awaiting a GPU smoke run before a deployment flips to it. The TRT-LLM
backend is a reserved factory branch; it unlocks the doc's speculative-
decoding path but hasn't been started.

Path 2 (detector-side TRT) also landed (2026-04-22): a `Detector`
Protocol + `build_detector(cfg, policy, backend)` factory in
`vrs/triage/backends.py` mirror the verifier-side abstraction. A
`TensorRTYOLOEDetector` skeleton loads a pre-exported `.engine` file
through Ultralytics' native TRT dispatch (which preserves the same
`predict()` surface) and `scripts/export_yoloe_trt.py` wraps the common
export case. The NVIDIA TAO alternate path is documented and produces an
identical `.engine` that this runtime accepts; which path an operator
uses is orthogonal to the runtime code. Like the vLLM backend, this is
structurally complete + Protocol-conformant but awaiting a GPU smoke
run — TRT engines are GPU-architecture-specific so a smoke against the
target hardware is mandatory before flipping a deployment.

Tier 1 #11a (failure-path tests) landed: five categories named by the
planning doc (RTSP reconnect, malformed verifier outputs beyond parser
helpers, sink failures, shutdown under in-flight work, multi-stream
config validation) are now pinned by `tests/test_failure_paths.py`
(28 tests). One real bug surfaced along the way — `SinkWorker` used to
let one write exception tear down the thread, silently dropping every
subsequent alert for that stream; fixed with per-message `try/except`
in `workers.py` and pinned by
`test_sink_worker_survives_write_failure_and_keeps_draining` +
`test_sink_worker_survives_annotator_write_failure`.

Tier 2 #11b (privacy controls — face blurring) landed (2026-04-23):
`vrs/privacy/` ships a `FaceDetector` Protocol, a `NullFaceDetector`
pass-through, a YuNet backend (`YuNetFaceDetector` via OpenCV's
`FaceDetectorYN`), and a `blur_faces` function. `VideoAnnotator` runs
the face detector on raw frame pixels *before* drawing any operator
overlays, then applies Gaussian blur to each detected face box. Both
pipelines (single-stream and multi-stream) thread the `privacy` config
block through to the annotator. `build_face_detector()` factory
gracefully degrades to `NullFaceDetector` on misconfigured models so a
bad path never takes a deployment down. 18 tests in
`tests/test_privacy.py` cover blur behavior, factory routing, YuNet
wiring via a fake `cv2.FaceDetectorYN`, and VideoAnnotator integration.

Tier 2 #12 (verifier failure policy) landed (2026-04-23): verifier
failure handling is now an explicit `verifier.failure_policy` setting.
`pass_through` remains the default and surfaces parser/model failures as
low-confidence alerts; `reject` suppresses malformed verifier outputs
when a deployment prioritizes false-alarm volume. Both single-stream and
multi-stream pipelines thread the setting into `AlertVerifier`, and
failure-path tests pin both policies.

The next open items are the deferred pieces from earlier tiers: D-Fire /
Le2i dataset adapters, cross-camera incident grouping, Stage B
calibration, and the TRT-LLM backend.
Everything below this line is retained for historical context and to
preserve the rationale for each item — inline ✅ callouts mark what has
landed.


## What Is Already Solid

These are not backlog items anymore.

- Logging already exists.
  - `vrs.setup_logging()` configures timestamped logs for scripts.
  - Single-stream and multi-stream paths both log verified alerts.
- Basic config validation already exists at `load_config()` time.
  - Missing top-level sections and required keys fail early with `ValueError`.
- Packaging already exists.
  - The repo has a `pyproject.toml` and installs as `vrs`.
- The verifier parser already uses balanced-brace extraction.
  - It is still fragile compared with constrained decoding, but it is not the
    old greedy-regex implementation.
- The multi-stream architecture is structurally sound.
  - One shared detector, one shared verifier, bounded queues, per-stream sinks.

This matters because the planning work should start from the actual repo, not
from an earlier snapshot.


## Tier 0: Immediate Correctness and Operability

These are the highest-priority issues because they affect whether the current
system behaves correctly at all.

### 1. Fix multi-stream manifest loading  ✅ done

Current state:

- `build_multistream_pipeline()` loads `streams_path` through `load_config()`.
- `load_config()` requires `ingest`, `detector`, `event_state`, `verifier`, and
  `sink`.
- `configs/multistream.yaml` intentionally contains only `multistream` and
  `streams`.

Result:

- `python scripts/run_multistream.py --streams configs/multistream.yaml` fails
  before the pipeline starts.

This is the most concrete functional bug in the repo today. The fix is simple:
the streams manifest loader must accept a partial manifest instead of reusing the
full single-stream config validator.

### 2. Fix the synthetic fire clip generator regression  ✅ done

Current state:

- The test suite currently fails in `tests/test_make_test_clips.py` for the
  `fire` generator.
- `scripts/make_test_clips.py` hardcodes a fire patch size of `280x420`, then
  blends it into whatever output frame size the caller requested.
- Small test resolutions cause a shape mismatch during alpha blending.

Why it matters:

- This is a real regression in the current test suite.
- The synthetic clips are explicitly part of the smoke-test workflow in the
  README, so this is user-facing repo breakage, not just test-only polish.

### 3. Unify verifier context-window ownership  ✅ done

Current state:

- Single-stream reads `verifier.context_window_s`.
- Multi-stream does not pass verifier context settings into `EventStateQueue`;
  it falls back to `EventStateQueue` defaults unless equivalent values happen to
  exist under `event_state`.

Why it matters:

- The same config key does not mean the same thing across single-stream and
  multi-stream execution.
- The README and planning docs talk about clip windows as if they are global
  verifier settings, but the implementation currently diverges.

This should be fixed before adding per-class temporal verifier windows.

**Landed (2026-04-22):** `DetectorWorker` now takes a `verifier_cfg` kwarg
and reads `keyframes` / `context_window_s` from there, matching single-
stream. Regression test
`test_detector_worker_reads_keyframes_and_window_from_verifier_cfg` pins
the wiring. Behavior change: multi-stream now honors
`configs/default.yaml`'s `verifier.context_window_s: 4.0` (previously
silently fell back to 3.0).

### 4. Surface queue drops as first-class operator signals  ✅ done

Current state:

- `BoundedQueue` tracks `puts_dropped`.
- `MultiStreamPipeline.queue_stats()` exposes drop counts.
- Nothing emits warnings or periodic status when drops occur under load.

Why it matters:

- If the verifier cannot keep up, the system silently degrades from "verify
  every candidate" to "drop oldest candidates".
- That is an acceptable backpressure policy for live video, but it must be
  visible to the operator.

Minimum fix:

- emit warning logs when candidate queue or frame queue drop counts increase,
- include stream context where possible,
- make the degraded mode obvious in logs.

**Landed (2026-04-22):** `MultiStreamPipeline.run()` now samples drop
counters every `multistream.drop_log_interval_s` seconds (default 5.0, 0
disables) and emits one WARNING per window naming every queue whose
counter grew — e.g. `frame_q+5, sink[cam_lobby]+2`. Healthy pipelines stay
silent. Per-stream sink queues are included in the snapshot, so each
stream id is named when that stream's sink backs up.

### 5. Add shutdown diagnostics  ✅ done

Current state:

- `MultiStreamPipeline.stop()` performs best-effort joins with timeouts.
- It does not log which thread failed to stop in time.

Why it matters:

- Hung shutdowns are difficult to debug without thread-specific diagnostics.

This is a small patch with outsized operational value.

**Landed (2026-04-22):** `stop()` collects any thread that exceeds its
join timeout and emits one WARNING naming every hung thread with its
timeout — e.g. `decoder[cam_lobby](>2s), verifier(>5s)`. Clean shutdowns
stay silent. The logic is factored into a static helper
`_join_or_track_hung` so it's unit-testable without a full pipeline.


## Tier 1: Core Product Roadmap

These are the highest-leverage product improvements once Tier 0 is cleared.

### 6. Build the real-dataset evaluation harness  ⏳ partially landed

This remains the most important roadmap item.

What is true today:

- Tests validate pure-Python behavior and worker plumbing.
- There is no dataset-backed measurement of precision, recall, F1, or verifier
  behavior on real video.

Why it stays first:

- Without this, every threshold, verifier, tracker, and latency optimization is
  ungrounded.

Recommended structure still stands:

```text
vrs/eval/
  datasets/
  metrics.py
  harness.py
```

The most important additional metric is still verifier-flip rate.

**Landed (2026-04-22) — Step 1.A + 1.B:**

```text
vrs/eval/
├── schemas.py            GroundTruthEvent, EvalItem, ClassMetrics, RunScore
├── metrics.py            score_alerts_against_truth, aggregate_scores
├── harness.py            evaluate(dataset, pipeline_factory, out_dir, …)
└── datasets/
    ├── base.py           Dataset protocol
    └── labeled_dir.py    directory-of-mp4 + sidecar-JSON adapter
scripts/eval.py           CLI wrapper
tests/test_eval.py        16 tests
```

Design choices worth noting:

- `true_alert=False` alerts do **not** count as FPs against ground truth —
  a verifier that correctly rejects a detector hit shouldn't penalize
  cascade precision. Flipped alerts are captured separately in
  `flip_rate`.
- Matching is greedy one-to-one per class, with a configurable
  `tolerance_s` around ground-truth windows (default 1.0 s). Two alerts
  inside one event window → 1 TP + 1 FP, not 2 TP — so a cooldown misfire
  can't inflate recall.
- `aggregate_scores` sums counts across runs and recomputes ratios rather
  than macro-averaging, to keep class imbalance honest.

**Also landed (Step 1.C — CI regression gate):**

`vrs/eval/ci.py` implements `compare_reports(baseline, current, max_f1_drop)`
and a CLI entry (`python -m vrs.eval.ci --baseline … --current …
--max-f1-drop 0.02`). It gates on per-class *and* overall F1 deltas, treats
a class that disappears from the current report as a regression (since
silent class loss is almost always a bug), welcomes new classes
informationally, and reports flip-rate / fn-flag-rate deltas as
diagnostics without gating on them. Exit codes 0 / 1 / 2 for pass / fail /
structural-error make it CI-pipeline-friendly.

**Remaining for #6 — data-blocked, parked:**

- `datasets/dfire.py` — image-level adapter + a detector-only eval path
  (D-Fire is image-labeled, not video-labeled, so it needs IoU matching
  rather than the current temporal matcher).
- `datasets/le2i.py` — frame-range → time-range conversion via native FPS.

Both can be added opportunistically once the datasets are local; they do
not block #7 onward.

### 7. Add constrained decoding to the verifier  ✅ done

This is still the cleanest reliability win.

Current state:

- The parser is better than before, but still heuristic.
- Parse failures currently fall back to pass-through behavior.

Why this matters:

- Constrained decoding removes an entire class of parser edge cases,
- reduces fallback logic,
- makes verifier output semantics much easier to reason about.

**Landed (2026-04-22):** `vrs/verifier/constrained.py` builds the JSON
schema (with `false_negative_class` restricted to an enum of the active
watch-policy classes + null) and, when `xgrammar` is installed, compiles
it into a transformers `LogitsProcessor`. `AlertVerifier` builds a fresh
processor per `verify()` call (xgrammar matchers are stateful) and passes
it through `CosmosReason2.chat_video(..., logits_processors=…)` to
`model.generate()`. When `xgrammar` is absent, the verifier falls back to
unconstrained generation + the existing balanced-brace parser, and logs
an INFO message once per process pointing at the `constrained` extra
(`pip install 'vrs[constrained]'`). Any setup error inside xgrammar is
downgraded to a WARNING and triggers the same fallback — the verifier
never crashes on constrained-decoding issues.

### 8. Add tracking before self-calibration  ✅ done (in-stream)

Tracking should come before automatic threshold tuning.

Reason:

- Self-calibration based on alert counts is biased when one persistent physical
  event becomes repeated cooldown-based alerts.
- A tracker gives the system a more stable unit of analysis: one occurrence,
  one track, one grouped incident.

Recommended order:

1. add lightweight tracking,
2. group alerts by track,
3. then prototype calibration against cleaner signals.

**Landed (2026-04-22) — Step 8.A (in-stream):** `vrs/triage/tracking.py`
ships `SimpleIoUTracker` (greedy IoU association, class-segregated, stale
track expiry) and a `NullTracker` pass-through. `Detection.track_id` and
`CandidateAlert.track_id` are optional fields (default ``None``) — that
preserves wire compatibility for anything reading `alerts.jsonl`.
`EventStateQueue` now fires one candidate per distinct `track_id` present
on the peak frame and debounces per `(class, track_id)`; untracked runs
collapse to a single `None` bucket and reproduce the pre-tracking
behavior exactly. Multi-stream builds one tracker instance per stream so
IDs don't collide across cameras. Config under `tracker:` in
`configs/default.yaml`.

**Remaining — Step 8.B (cross-camera):** the "two cameras see the same
fire → one incident" correlator in the improvements doc. Spatio-temporal
matching across streams; more scope than an in-pipeline addition and
deferred.

### 9. Attack verifier latency  ⏳ abstraction landed; engines pending

This remains the system's primary scaling limit.

Current state:

- Detector throughput is not the first bottleneck.
- Verifier latency directly drives candidate backlog and queue drops.

Work items:

- TensorRT-LLM or equivalent accelerated inference path,
- constrained decoding on top,
- prompt/token-budget reduction where it does not cost accuracy.

**Landed (2026-04-22) — backend abstraction:** the verifier no longer
depends on a concrete runtime. `vrs/runtime/backends.py` defines a
`CosmosBackend` Protocol (`chat_video(..., response_schema)`) and a
`build_cosmos_backend(cfg, backend)` factory. `AlertVerifier` accepts
anything conforming to the Protocol — transformers-specific attribute
access (`cosmos.processor.tokenizer`, `cosmos.model.config.vocab_size`)
was moved into the transformers backend itself, so new engines plug in
without touching verifier code.

**Landed — vLLM skeleton (awaiting GPU validation):**
`vrs/runtime/vllm_cosmos.py` is a real implementation targeting vLLM
≥ 0.6.5 — chat-message packing with Qwen3-VL image content,
`GuidedDecodingParams(json=schema)` for constraint enforcement, graceful
downgrade if the vLLM version lacks guided decoding. The module is
structurally tested (fake vllm module substituted into `sys.modules`)
but has not been round-tripped against a live GPU in this repo's CI.
Before flipping a deployment to the vLLM backend, run a short smoke
against the real model and pin the vLLM version you validated against.

**Remaining:** the TRT-LLM backend (speculative decoding with a tiny
Qwen3-0.5B draft is where the stacked 10 s → 3-4 s win comes from) and
the measurement work — once vLLM is validated, add a latency-per-verify
metric to `queue_stats()` and watch it move under load.


## Tier 2: Safety, Privacy, and Test Depth

### 10. Expand failure-path tests  ✅ done (2026-04-23)

Current tests cover:

- config validation,
- policy loading,
- event-state logic,
- parser helpers,
- queue behavior,
- worker fanout.

Still missing:

- RTSP reconnect behavior,
- malformed or partial model outputs beyond the current parser helper cases,
- sink failures,
- shutdown under in-flight work,
- multi-stream config/manifest validation.

**Landed:** `tests/test_failure_paths.py` covers all five categories the
list above named, 28 tests total:

- **RTSP reconnect** — `OpenCVReader` and `StreamReader` both reconnect
  on live sources after a failed `read()`, and both bail out cleanly
  when the reconnect itself fails; file sources never attempt to
  reconnect.
- **Malformed verifier outputs** — truncated JSON, missing
  `true_alert` key, out-of-[0,1] confidence (clamped), illegal
  `false_negative_class` (nulled), backend exception (diagnostic
  pass-through), empty keyframes (short-circuit, backend not called),
  empty-string response (safe fallback).
- **Sink failures** — the SinkWorker now wraps per-message writes in
  try/except so one exception doesn't kill the thread and lose every
  subsequent alert for that stream. Both the JSONL-failure and
  annotator-failure branches are pinned.
- **Shutdown under in-flight work** — an in-progress verify finishes
  and flushes to the sink before stop() completes, and a closed-but-
  non-empty `BoundedQueue` still delivers its backlog so last-moment
  alerts survive shutdown.
- **Manifest validation** — non-mapping YAML, missing `streams`, non-
  list `streams`, non-dict stream entry, missing `id`, missing source
  key, all three source aliases (`rtsp` / `source` / `video`), empty
  streams list, duplicate stream ids.

### 11. Make privacy controls a real product requirement  ✅ done (2026-04-23)

Face blurring should not live at the bottom of a speculative backlog if the
target deployments include jurisdictions such as Korea or the EU.

It does not need to block core plumbing work, but it should be treated as a
planned product requirement, not as decorative polish.

**Landed (2026-04-23):** `vrs/privacy/` ships face detection +
Gaussian blur, wired into both single-stream and multi-stream annotated
mp4 writers. YuNet backend (~300 KB ONNX, CPU-only, ships with
OpenCV ≥ 4.8) runs before any overlay is drawn. Factory degrades to a
no-op on bad config so a privacy misconfiguration never crashes the
pipeline. `configs/default.yaml` carries the full `privacy:` block
(disabled by default; flip `enabled: true` once the ONNX model is
deployed). 18 tests cover blur, factory, YuNet wiring, and annotator
integration.

### 12. Make verifier failure policy explicit  ✅ done (2026-04-23)

Current behavior is intentional but easy to misread:

- parse/model failure does not drop the detector hit,
- the system surfaces a low-confidence pass-through alert instead.

That may be the right operational choice, but it should be named and documented
as policy. If the team wants a different behavior, change it explicitly and add
tests for the chosen semantics.

**Landed:** `verifier.failure_policy` accepts `pass_through` (default) or
`reject`. `pass_through` keeps the safety-oriented behavior: malformed JSON,
missing keyframes, backend exceptions, or missing `true_alert` produce a
`VerifiedAlert` with `true_alert=True`, `confidence=0.0`, and a diagnostic
rationale. `reject` keeps the audit record but returns `true_alert=False`
for failure cases. Both pipeline constructors pass the config through to
`AlertVerifier`, and `tests/test_failure_paths.py` pins both branches.


## Priority Order

Done (pre-session or 2026-04-22 session):

1. ~~fix multi-stream manifest loading~~ ✅
2. ~~fix the synthetic fire clip regression~~ ✅
3. ~~unify verifier context-window config behavior~~ ✅
4. ~~add queue-drop diagnostics~~ ✅
5. ~~add shutdown diagnostics~~ ✅

Open:

6. eval harness — ⏳ core + CI regression gate landed; D-Fire / Le2i
   adapters parked pending dataset acquisition (not blocking),
7. ~~add constrained decoding~~ ✅
8. ~~add tracking~~ ✅ (in-stream); cross-camera incident grouping
   deferred to 8.B
9. ~~prototype self-calibration on top of tracking~~ ✅ Stage A (log-only);
   Stage B (autonomous apply) deferred
10. reduce verifier latency — ⏳ backend Protocol + vLLM skeleton landed;
    awaiting GPU validation, TRT-LLM backend still open,
11. deepen failure-path tests — ✅ done (2026-04-23); privacy controls
     (face blurring) — ✅ done (2026-04-23),
12. ~~make verifier failure policy explicit~~ ✅ done (2026-04-23).


## Summary

As of 2026-04-23 Tier 0 is clear and all actionable Tier 1 items have
landed. Tier 2 is also clear: failure-path tests, privacy controls, and
explicit verifier failure policy are done.

Total test count: **166** (up from 120 at the start of Tier 1).

Remaining open work:

- **#6** D-Fire / Le2i dataset adapters — parked pending dataset
  acquisition (not blocking).
- **#8.B** cross-camera incident grouping — deferred, larger scope.
- **#9.B** autonomous self-calibration (Stage B) — deferred.
- **#10** vLLM GPU validation + TRT-LLM backend — awaiting hardware.
