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

### 1. Real-dataset evaluation harness

Until we can say "F1 on D-Fire is X, precision on Le2i Falldown is Y,
verifier-flip rate is Z on UCF-Crime," every other change is guesswork.

Proposed structure:

```
vrs/eval/
├── datasets/
│   ├── dfire.py          # fire/smoke bboxes
│   ├── le2i.py           # falldown clip labels
│   └── ucf_crime.py      # anomaly timestamps
├── metrics.py            # per-class P/R/F1 + verifier-flip rate
├── harness.py            # iterate dataset → run cascade → score
└── ci.py                 # regression gate: fail on F1 delta < -0.02
```

**Key metric we don't currently track:** *verifier-flip rate* — how often
Cosmos overrides YOLOE. A healthy cascade has a modest, stable flip rate
(5–15 %); trending up means YOLOE drift, trending down means Cosmos is
rubber-stamping.

**Why first:** makes every subsequent change measurable.

### 2. Closed-loop threshold self-calibration

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

### 3. Verifier latency attack

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

### 4. Object tracking + incident grouping

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

### 6. Structured output via constrained decoding

Our JSON parser has a regex fallback and a single-quote-repair path
because VLMs sometimes emit malformed output. With **XGrammar** +
transformers (or TRT-LLM guided decoding), the model **cannot** emit
non-conformant JSON — delete the parser fallback entirely, smaller
surface, zero silent failures.

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
- **Face blurring before writing annotated mp4s.** Privacy by default;
  legally required in some jurisdictions (EU/KR).
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
2. **Constrained decoding for the verifier** (item 6) — quick win,
   eliminates a whole class of silent failures.
3. **Closed-loop self-calibration prototype, Stage A** (item 2) — most
   novel, highest long-term value. Even a version that only *logs*
   suggested threshold deltas (without applying them) is an enormous
   operator-facing debug tool.

Everything else falls out of those three.
