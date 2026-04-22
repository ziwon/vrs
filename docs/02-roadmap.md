# VRS Roadmap

Status date: 2026-04-23.

This is the forward plan. Keep historical rationale in `docs/archive/`; keep
this file short, ranked, and tied to measurable outcomes.

## P0 — Make Quality Measurable

### 1. Real-Dataset Adapters

Implement and verify dataset adapters for:

- D-Fire or equivalent fire/smoke data.
- Le2i or UP-Fall fall-detection data.
- UCF-Crime or internal CCTV anomaly clips.

Acceptance criteria:

- `scripts/eval.py` can run each dataset without custom one-off scripts.
- The report includes per-class precision, recall, F1, verifier flip rate, and
  false-negative flag rate.
- A committed baseline report exists for CI comparison.

### 2. Detector-Only And BBox Scoring

The current eval harness is event-level. D-Fire-like datasets often need
image-level or bbox-level scoring.

Acceptance criteria:

- Detector-only scoring path that bypasses Cosmos when labels are frame/image
  level.
- IoU-based TP/FP/FN matching for bbox datasets.
- Separate report sections for detector quality and full-cascade quality.

### 3. Detector Model Refresh Evaluation

The code defaults to `yoloe-11l-seg.pt`. Newer YOLOE-26 variants should be
tested, not assumed better for the watch policy.

Acceptance criteria:

- Side-by-side eval of current YOLOE-L and candidate YOLOE-26 model.
- Same prompts, same thresholds, same labeled clips.
- Decision recorded with precision/recall/F1 and latency deltas.

## P1 — Raise Throughput Safely

### 4. Qwen Verifier Backend And Model Bake-Off

Internal benchmarks indicate that Cosmos-Reason2-2B is not strong enough to
lock in as the production verifier. Add a Qwen-compatible verifier path and
compare it against Cosmos on the same CCTV labels.

Priority candidates:

- Qwen3.6-class served VLM, via vLLM/SGLang/OpenAI-compatible API.
- Qwen3.5-class served VLM when Qwen3.6 deployment cost is too high.
- Smaller local Qwen VL candidate only if the deployment must stay on one
  16 GB GPU.

Acceptance criteria:

- Generalize `CosmosBackend` naming or add a parallel `VLMBackend` abstraction.
- Add a Qwen/OpenAI-compatible backend that accepts the same verifier prompt and
  response JSON schema.
- Run Cosmos and Qwen candidates on identical labeled clips.
- Record per-class precision, recall, F1, verifier flip rate, malformed JSON
  rate, p50/p95 latency, and GPU memory.
- Do not change the default production model until eval data shows a clear win.

### 5. Verifier Latency Instrumentation

Before changing runtimes, measure where time is spent.

Acceptance criteria:

- Per-verify latency p50/p95/p99.
- Model backend error counter.
- Token generation rate when the backend exposes it.
- Queue wait time separate from model execution time.

### 6. Validate vLLM Backend

The vLLM backend is implemented as a structural skeleton. It needs live GPU
validation before deployment.

Acceptance criteria:

- Smoke test with `nvidia/Cosmos-Reason2-2B` and the selected Qwen candidate
  when vLLM supports that model.
- Valid structured JSON under guided decoding.
- Version-pinned `vllm` extra after validation.
- Throughput comparison against transformers backend on identical clips.

### 7. TRT-LLM Backend

TRT-LLM is the next large verifier-speed path.

Acceptance criteria:

- Backend implementation behind `CosmosBackend`.
- JSON-constrained output support.
- Optional speculative decoding configuration.
- GPU benchmark showing material latency reduction versus transformers.

## P1 — Reduce Operator Noise

### 8. Cross-Camera Incident Correlation

Current tracking reduces duplicates only within one stream. Operators need one
incident when overlapping cameras see the same event.

Acceptance criteria:

- Incident IDs that group alerts across streams.
- Configurable camera adjacency/overlap map.
- Time-window correlation with class, severity, and optional geometry.
- JSONL schema extension that preserves original per-camera alerts.

### 9. Per-Class Verifier Windows

Slow events need longer context than instantaneous events.

Acceptance criteria:

- Add `verifier_window_s` or equivalent to each watch-policy item.
- Smoke/fire/falldown policies can use different verifier clip windows.
- Backward-compatible default from `verifier.context_window_s`.

### 10. Calibration Stage B

Stage A writes suggestions. Stage B should apply them under tight controls.

Acceptance criteria:

- Per-stream/per-class min-score updates with caps.
- Cooldown between changes.
- Audit log with old value, new value, reason, and verifier statistics.
- Rollback mechanism or config export for operator review.

## P2 — Production Hardening

### 11. Metrics Endpoint

Expose runtime health without scraping logs.

Acceptance criteria:

- Prometheus or OpenTelemetry-compatible metrics.
- Queue depth/drop counters.
- Detector/verifier latency.
- Alert counts by stream, class, verdict, and severity.
- Privacy detector setup failures.

### 12. Signed Audit Log

Alerts should be tamper-evident when used for compliance or incident review.

Acceptance criteria:

- HMAC or chained hash per JSONL line.
- Key-id support for rotation.
- Verification CLI for replay/audit.

### 13. Hot Policy Reload

Operators should not restart the process for threshold-only policy changes.

Acceptance criteria:

- Atomic policy reload on file change or SIGHUP.
- Clear boundary for changes that still require model reload, such as detector
  prompt vocabulary changes.
- Rollback on invalid policy file.

## Non-Goals For Now

- Replacing the two-stage cascade. The fast-localize/slow-reason design remains
  the right architecture for this project.
- Replacing the watch policy with a prompt bank. The operator surface should
  stay YAML event definitions.
- DeepStream zero-copy rewrite before accuracy and verifier throughput are
  measured. It is a valid optimization, but not the next constraint to solve.
