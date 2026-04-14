Planning Notes — code-accurate as of v0.3

This document is a current-state engineering plan, grounded in the code that is
actually in the repository today. It separates three things that were previously
mixed together:

1. what is already implemented,
2. what is currently broken or misleading,
3. what should land next.


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

### 1. Fix multi-stream manifest loading

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

### 2. Fix the synthetic fire clip generator regression

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

### 3. Unify verifier context-window ownership

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

### 4. Surface queue drops as first-class operator signals

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

### 5. Add shutdown diagnostics

Current state:

- `MultiStreamPipeline.stop()` performs best-effort joins with timeouts.
- It does not log which thread failed to stop in time.

Why it matters:

- Hung shutdowns are difficult to debug without thread-specific diagnostics.

This is a small patch with outsized operational value.


## Tier 1: Core Product Roadmap

These are the highest-leverage product improvements once Tier 0 is cleared.

### 6. Build the real-dataset evaluation harness

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

### 7. Add constrained decoding to the verifier

This is still the cleanest reliability win.

Current state:

- The parser is better than before, but still heuristic.
- Parse failures currently fall back to pass-through behavior.

Why this matters:

- Constrained decoding removes an entire class of parser edge cases,
- reduces fallback logic,
- makes verifier output semantics much easier to reason about.

### 8. Add tracking before self-calibration

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

### 9. Attack verifier latency

This remains the system's primary scaling limit.

Current state:

- Detector throughput is not the first bottleneck.
- Verifier latency directly drives candidate backlog and queue drops.

Work items:

- TensorRT-LLM or equivalent accelerated inference path,
- constrained decoding on top,
- prompt/token-budget reduction where it does not cost accuracy.


## Tier 2: Safety, Privacy, and Test Depth

### 10. Expand failure-path tests

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

### 11. Make privacy controls a real product requirement

Face blurring should not live at the bottom of a speculative backlog if the
target deployments include jurisdictions such as Korea or the EU.

It does not need to block core plumbing work, but it should be treated as a
planned product requirement, not as decorative polish.

### 12. Make verifier failure policy explicit

Current behavior is intentional but easy to misread:

- parse/model failure does not drop the detector hit,
- the system surfaces a low-confidence pass-through alert instead.

That may be the right operational choice, but it should be named and documented
as policy. If the team wants a different behavior, change it explicitly and add
tests for the chosen semantics.


## Priority Order

If implementing from the current codebase, the order should be:

1. fix multi-stream manifest loading,
2. fix the synthetic fire clip regression,
3. unify verifier context-window config behavior,
4. add queue-drop and shutdown diagnostics,
5. build the eval harness,
6. add constrained decoding,
7. add tracking and incident grouping,
8. prototype self-calibration on top of tracking,
9. reduce verifier latency,
10. deepen failure-path tests and privacy controls.


## Summary

The repo is in a better state than an older critique would suggest: logging,
packaging, config validation, and the basic queue/worker architecture are
already present. The real problems are narrower and more concrete:

- one confirmed multi-stream config bug,
- one confirmed test regression,
- one config-consistency bug around verifier context windows,
- weak operator visibility under overload,
- and the still-missing eval harness.

That is a manageable plan. The next work should focus on making the current
system correct and observable before adding higher-level automation.
