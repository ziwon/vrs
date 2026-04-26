# Improvements — planning doc

A retrospective of what the v0.3 build got right, got wrong, and a ranked
backlog of concrete improvements. Use this as the source of truth for what
to tackle next; keep it updated as items land.

## What v0.3 got wrong (being honest)

- **VRAM was over-optimized.** 16 GB is not tight for this cascade; the real
  scale constraint is **verifier latency**, not memory. The "how many
  streams" conversation should have been framed around Cosmos's
  tokens-per-second from day one, not GB of weights.
- **No accuracy harness.** Everything we ship validates plumbing, not
  correctness. This is the single largest hole in the repo today — every
  other optimization is speculation until we can measure P/R/F1 against a
  real dataset.
- **No feedback loop from verifier to detector.** The verifier produces
  ground-truth-like signal (`true_alert` / `false_alarm`) on every
  candidate. We throw it away. That's leaving the most valuable signal
  in the system on the floor.

## Highest-leverage improvements (ranked)

### 1. Real-dataset evaluation harness  ⏳ partially landed (2026-04-22)

Until we can say "F1 on D-Fire is X, precision on Le2i Falldown is Y,
verifier-flip rate is Z on UCF-Crime," every other change is guesswork.

Proposed structure:

```
vrs/eval/
├── datasets/
│   ├── base.py           ✅ Dataset protocol
│   ├── labeled_dir.py    ✅ generic directory-of-mp4 + sidecar-JSON adapter
│   ├── dfire.py          ⏳ fire/smoke bboxes (image-level; needs IoU matcher)
│   ├── le2i.py           ⏳ falldown clip labels (frame-range → time-range)
│   └── ucf_crime.py      ⏳ anomaly timestamps
├── schemas.py            ✅ GroundTruthEvent, EvalItem, ClassMetrics, RunScore
├── metrics.py            ✅ per-class P/R/F1 + verifier-flip + FN-flag rates
├── harness.py            ✅ iterate dataset → run cascade → score
└── ci.py                 ✅ regression gate: fail on F1 delta < -0.02
scripts/eval.py           ✅ CLI wrapper
```

**Key metric we don't currently track:** *verifier-flip rate* — how often
Cosmos overrides YOLOE. A healthy cascade has a modest, stable flip rate
(5–15 %); trending up means YOLOE drift, trending down means Cosmos is
rubber-stamping.

**Why first:** makes every subsequent change measurable.

**Remaining work:** the three dataset adapters (D-Fire especially needs a
separate detector-only eval path with IoU matching, since it's image-
labeled rather than video-event-labeled). Everything else — scoring,
matching, aggregation, flip-rate tracking, CLI, and the CI regression
gate — is in and tested.

### 2. Closed-loop threshold self-calibration  ⏳ Stage A landed (2026-04-22)

Per camera per class, track a rolling precision estimate from verifier
flips and auto-adjust YOLOE's per-class `min_score`:

```python
# pseudocode, run every N minutes per (stream_id, class)
flip_rate = recent_verifier_flips / max(recent_alerts, 1)
if flip_rate > 0.30:
    min_score += 0.02    # tighten — too many false alarms
elif flip_rate < 0.05 and recent_alerts_per_hour < target_recall:
    min_score -= 0.02    # loosen — we may be missing cases
```

Each camera ends up with its own tuned thresholds after ~24 h of
operation — no human curation. This is how we recover what a carefully
tuned prompt bank would give you, **learned from the verifier's
ground-truth-ish signal** instead of from offline prompt scoring.

Ship in two stages:

- **Stage A (safe):** log flip rates and *suggest* threshold deltas in
  `runs/*/calibration_suggestions.jsonl`; a human applies them.
- **Stage B (autonomous):** apply deltas directly with caps
  (`min_score ∈ [0.15, 0.80]`) and a cool-down between adjustments.

**Landed — Stage A:** `vrs/calibration/` ships a stateless
`suggest(stream_id, class, current_min_score, window, …)` function and a
stateful `Calibrator` that wraps it. Every `VerifiedAlert` the verifier
produces is fed to the calibrator; when the rolling flip rate crosses
thresholds, a `Suggestion` is written to
`<out>/calibration_suggestions.jsonl`. The loosen arm is gated on an
operator-supplied `target_alerts_per_hour` (default null → tighten-only,
which is the safe behavior when we don't know what the site's real
alert rate should be). After each emission the window is cleared, so
one sustained regime → one suggestion, not one-per-alert. Multi-stream
shares a single calibrator and keys internal state on
`(stream_id, class)` so cameras don't contaminate each other's flip
statistics. Opt-in via `calibration.enabled: true` in `default.yaml`.

### 3. Verifier latency attack  ⏳ backend abstraction + vLLM skeleton landed

At 10–15 s per alert (BF16), the verifier is the capacity ceiling. Three
stacking wins:

| Change | Expected speedup | Effort |
|---|---|---|
| **TensorRT-LLM** for Cosmos-Reason2-2B instead of `transformers.generate` | ~2–3× gen throughput | medium |
| **Speculative decoding** with a tiny Qwen3-0.5B draft model | ~1.6–1.8× on top of TRT-LLM | medium |
| **Constrained decoding** (XGrammar / lm-format-enforcer) to force JSON | ~5–10 %, plus eliminates parser-fallback path | low |

Stacking these: roughly 10 s → 3–4 s per verify. That directly raises the
sustainable alert rate by ~3×, which matters more than any fast-path
optimization.

**Landed:** constrained decoding shipped earlier (item 6 above). The
verifier has been abstracted behind a `CosmosBackend` Protocol
(`vrs/runtime/backends.py`) with a `build_cosmos_backend(cfg, backend)`
factory picking `transformers` / `vllm` / `trtllm`. Each backend handles
its own constraint surface internally, so new engines plug in without
touching `AlertVerifier`. A vLLM backend skeleton
(`vrs/runtime/vllm_cosmos.py`) is structurally complete — builds a
Qwen3-VL chat message list, attaches `GuidedDecodingParams(json=schema)`
for constrained output, and is Protocol-conformant per
`tests/test_backends.py` — but has not been smoke-run against a live
vLLM instance in this repo's CI. Validate end-to-end and pin the vLLM
version before flipping a deployment.

**Remaining:** TRT-LLM backend (plus its speculative-decoding config)
and per-verify latency instrumentation to actually measure the gains.

### 4. Object tracking + incident grouping  ⏳ in-stream landed (2026-04-22)

Right now a single fire raises an alert per cooldown window per camera.
In reality we want:

- A **ByteTrack / OC-SORT** tracker on YOLOE's boxes → every event gets
  a `track_id`.
- Alerts group by `(stream_id, class, track_id)` → **one event per
  physical occurrence**, with a duration, not a stream of duplicates.
- When two cameras see the same thing (overlapping FOV), a lightweight
  spatio-temporal correlator collapses them into one **incident** with
  multiple camera views attached.

This is what matters to an operator watching a wall of monitors — they
don't want 40 log lines for one fire, they want one incident with a
clip timeline.

**Landed — in-stream:** `vrs/triage/tracking.py` ships a pure-Python
`SimpleIoUTracker` (greedy IoU association, class-segregated, stale
track expiry) + a `NullTracker` pass-through. `Detection.track_id` and
`CandidateAlert.track_id` now carry through the cascade into
`alerts.jsonl`. `EventStateQueue` fires one candidate per distinct
track_id per firing class and keys cooldown on `(class, track_id)`. A
persistent fire with a stable track → one alert. Two simultaneous fires
with different tracks → two alerts. Untracked runs reproduce the old
behavior unchanged.

**Remaining — cross-camera correlator:** the overlapping-FOV dedupe that
produces one "incident" from N cameras is a larger piece and is deferred
until after the eval harness is on real data.

### 5. Temporal-context verifier for slow events

A 30-second smoke event barely changes between adjacent keyframes; the
current ±3 s window is too narrow. Cosmos-Reason2-2B has a **256 K
context window** and we're using a tiny fraction of it.

- For `smoke` / `falldown-duration` / anything where **persistence is
  the evidence**, hand the verifier a 30–60 s clip at FPS=4 (120–240
  frames) instead of 6 keyframes.
- For `fire` / `weapon` / instantaneous events, keep the short window —
  token budget is the only reason to shrink.
- Encode per-class via a new `verifier_window_s` field in the watch
  policy.

### 6. Structured output via constrained decoding  ✅ done (2026-04-22)

Our JSON parser has a regex fallback and a single-quote-repair path
because VLMs sometimes emit malformed output. With **XGrammar** +
transformers (or TRT-LLM guided decoding), the model **cannot** emit
non-conformant JSON — delete the parser fallback entirely, smaller
surface, zero silent failures.

**Landed:** `vrs/verifier/constrained.py` builds the response JSON schema
(policy-aware `false_negative_class` enum included) and, when `xgrammar`
is available, compiles it to a transformers `LogitsProcessor` that
Cosmos's `generate()` consumes. The balanced-brace parser stays in place
as a strict safety net for hosts without xgrammar, and the import is
guarded so the rest of the cascade runs unchanged on machines where the
extra isn't installed. Install with `uv sync --extra constrained`.

## Worth doing, but second-tier

- **DINOv2 novelty gate before YOLOE.** In a 24/7 CCTV system with ~95 %
  quiet frames, a cheap DINOv2-S rolling-baseline skip could cut
  fast-path compute ~50 %. Real win only when scaling past ~30 streams
  per GPU.
- **Visual-prompt YOLOE.** YOLOE accepts reference images as prompts. A
  "that specific handgun model" or "this uniform" use case becomes
  image-upload instead of text prompt. Operationally valuable for
  site-specific assets.
- **Hot-reload of watch policy** without pipeline restart.
- **Prometheus metrics endpoint** reading `queue_stats()`. One
  afternoon's work; huge ops value once you're running more than one
  GPU.
- **TRT-exported YOLOE + DeepStream 8.0 decoder.** End-to-end zero-copy
  GPU path (`nvurisrcbin → nvstreammux → nvinfer`). Biggest latency win
  for dense multi-stream sites. Big investment (both models need TRT
  export), so only worth doing *after* the accuracy harness tells you
  the policy is correct.
  _Status (2026-04-22):_ the **detector-side** TRT path has landed as a
  `Detector` Protocol + `TensorRTYOLOEDetector` backend — load a
  pre-exported `.engine` (produced either by Ultralytics'
  `model.export(format="engine")` or NVIDIA TAO's `tao-deploy`) and
  point `detector.backend: tensorrt` at it. `scripts/export_yoloe_trt.py`
  wraps the common case. Awaiting a GPU smoke run; the full DeepStream
  zero-copy pipeline is still future work.
- **Face blurring before writing annotated mp4s.** Privacy by default;
  legally required in some jurisdictions (EU/KR).
  _Status (2026-04-23):_ landed. `vrs/privacy/` ships YuNet face
  detection + Gaussian blur, wired into both pipelines' annotated mp4
  writers. Opt-in via `privacy.enabled: true` in `default.yaml`.
- **Audit-log signing** — HMAC each `alerts.jsonl` line with a
  stream-specific key, for tamper-evident replay.

## Things we should not change

- **The cascade structure itself.** Fast-path localization + slow-path
  reasoning is the right pattern and the 2025–2026 VAD literature
  converges on it.
- **Cosmos-Reason2-2B as the reasoning model.** 2B fits anywhere, the
  physical-world priors fit the domain, and the native bbox/trajectory
  outputs are genuinely differentiated.
- **Watch Policy as the only operator surface.** Plain-English YAML is
  the right abstraction.
- **Bounded queues with `drop_oldest` as the default.** Live-stream
  correctness > completeness, always.

## If we pick three to do first

1. **Accuracy harness on D-Fire + Le2i** (item 1) — unlocks everything
   else; without it we're optimizing blind.
   _Status: core harness + CI regression gate landed 2026-04-22; D-Fire
   and Le2i dataset adapters are the only remaining pieces and are parked
   pending dataset acquisition._
2. **Constrained decoding for the verifier** (item 6) — quick win,
   eliminates a whole class of silent failures.
   _Status: landed 2026-04-22 via the optional `xgrammar` dep._
3. **Closed-loop self-calibration prototype, Stage A** (item 2) — most
   novel, highest long-term value. Even a version that only *logs*
   suggested threshold deltas (without applying them) is an enormous
   operator-facing debug tool.
   _Status: landed 2026-04-22; Stage B (autonomous apply with caps +
   cooldowns) still open._

Everything else falls out of those three.
