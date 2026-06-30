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

Status: adapter coverage implemented. `scripts/eval.py` supports
`--dataset-format dfire` for D-Fire image/bbox labels, `--dataset-format le2i`
for Le2i fall frame-range labels, and `--dataset-format ucf_crime` for
UCF-Crime/UCA temporal annotations. Remaining work is running these against
local full datasets and committing baseline reports.

Acceptance criteria:

- `scripts/eval.py` can run each dataset without custom one-off scripts.
- The report includes per-class precision, recall, F1, verifier flip rate, and
  false-negative flag rate.
- A committed baseline report exists for CI comparison.

### 2. Detector-Only And BBox Scoring

The current eval harness is event-level. D-Fire-like datasets often need
image-level or bbox-level scoring.

Status: implemented. `scripts/eval.py --mode detector_only` bypasses the
verifier for still-image datasets, D-Fire labels carry normalized boxes, scoring
can require `--bbox-iou-threshold`, and `report.json` separates
`detector_quality` from `full_cascade_quality`.

Acceptance criteria:

- Detector-only scoring path that bypasses Cosmos when labels are frame/image
  level.
- IoU-based TP/FP/FN matching for bbox datasets.
- Separate report sections for detector quality and full-cascade quality.

### 3. Detector Model Refresh Evaluation

The code defaults to `yoloe-11l-seg.pt`. Newer YOLOE-26 variants should be
tested, not assumed better for the watch policy.

Status: implemented. `scripts/eval_detector_models.py` runs fixed-policy
side-by-side detector comparisons, records precision/recall/F1 and latency
deltas against a baseline model, and writes an auditable `model_refresh.json`
decision report. The RTX 5080 D-Fire YOLOE-26 comparison is recorded in
`docs/benchmarks/dfire-yoloe26-model-refresh-2026-06-21.md`.

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

Status: implementation ready; live candidate bake-off still required.
`VLMBackend` supports `transformers`, `vllm`, and `openai_compatible` verifier
backends; `configs/qwen-openai-compatible.yaml` defines the served Qwen-style
profile; and `scripts/eval_verifier_backends.py` runs identical full-cascade
evals and writes `verifier_bakeoff.json` with metrics, quality signals,
latency, and runtime memory fields. Remaining work is to point the served
profile at a selected Qwen3.5/Qwen3.6-class endpoint and commit the resulting
comparison report.

Acceptance criteria:

- Use the generalized `VLMBackend` abstraction to plug in candidate backends.
- Add a Qwen/OpenAI-compatible backend that accepts the same verifier prompt and
  response JSON schema.
- Run Cosmos and Qwen candidates on identical labeled clips.
- Record per-class precision, recall, F1, verifier flip rate, malformed JSON
  rate, p50/p95 latency, and GPU memory.
- Do not change the default production model until eval data shows a clear win.

### 5. Verifier Latency Instrumentation

Before changing runtimes, measure where time is spent.

Status: implemented. Eval reports include verifier latency p50/p95/p99 and
backend token generation rate when exposed. Runtime metrics export verifier
latency, verifier backend errors, token generation throughput, and separate
queue wait histograms for worker queues.

Acceptance criteria:

- Per-verify latency p50/p95/p99.
- Model backend error counter.
- Token generation rate when the backend exposes it.
- Queue wait time separate from model execution time.

### 6. Validate vLLM Backend

The vLLM backend is implemented as a structural skeleton. It needs live GPU
validation before deployment.

Status: validated for the Cosmos baseline on RTX 5080. Live smoke/eval
completed on 2026-06-22 with `vllm==0.19.1`; see
`docs/benchmarks/rtx5080-vllm-cosmos-validation-2026-06-22.md`. The validation
confirmed structured JSON and pinned the optional `vllm` extra. Remaining work
is target-host deployment sizing and a comparable Qwen-class candidate when a
selected model is available through vLLM or a served backend.

Acceptance criteria:

- Smoke test with `nvidia/Cosmos-Reason2-2B` and the selected Qwen candidate
  when vLLM supports that model.
- Valid structured JSON under guided decoding.
- Version-pinned `vllm` extra after validation.
- Throughput comparison against transformers backend on identical clips.

### 7. TRT-LLM Backend

TRT-LLM is the next large verifier-speed path.

Status: structural backend implemented; live GPU benchmark still required.
`verifier.backend: trtllm` routes through `VLMBackend`, uses TensorRT-LLM's
Python `LLM` API, passes the verifier JSON schema through guided decoding, and
accepts optional draft-model speculative decoding config. Remaining work is a
target-GPU smoke/eval benchmark proving latency reduction versus transformers
before any production promotion.

Acceptance criteria:

- Backend implementation behind `VLMBackend`.
- JSON-constrained output support.
- Optional speculative decoding configuration.
- GPU benchmark showing material latency reduction versus transformers.

## P1 — Reduce Operator Noise

### 8. Cross-Camera Incident Correlation

Current tracking reduces duplicates only within one stream. Operators need one
incident when overlapping cameras see the same event.

Status: implemented for the multistream pipeline as an opt-in
`multistream.incident_correlation` stage. Verified alerts keep their original
per-camera JSONL records and receive `incident_id`, `incident_stream_ids`, and
`incident_primary_stream_id` fields when correlation is enabled.

Acceptance criteria:

- Incident IDs that group alerts across streams.
- Configurable camera adjacency/overlap map.
- Time-window correlation with class, severity, and optional geometry.
- JSONL schema extension that preserves original per-camera alerts.

### 9. Per-Class Verifier Windows

Slow events need longer context than instantaneous events.

Status: implemented. Watch-policy items accept optional `verifier_window_s`
overrides, `EventStateQueue` falls back to `verifier.context_window_s` when the
override is absent, and policy reload treats window-only edits as runtime-safe.
The default safety policy uses a longer smoke verifier window while fire keeps
the global default.

Acceptance criteria:

- Add `verifier_window_s` or equivalent to each watch-policy item.
- Smoke/fire/falldown policies can use different verifier clip windows.
- Backward-compatible default from `verifier.context_window_s`.

### 10. Calibration Stage B

Stage A writes suggestions. Stage B should apply them under tight controls.

Status: implemented as an opt-in calibration apply/export stage. It keeps
per-stream/per-class applied scores, enforces caps and cooldowns, writes
`calibration_applied.jsonl`, and atomically exports
`calibration_overrides.yaml` for operator review or rollback.

Acceptance criteria:

- Per-stream/per-class min-score updates with caps.
- Cooldown between changes.
- Audit log with old value, new value, reason, and verifier statistics.
- Rollback mechanism or config export for operator review.

## P2 — Production Hardening

### 11. Canonical Runtime Contracts

Status: identity-hardened runtime slice implemented. Versioned JSON Schema files live under
`contracts/schemas/`, and `vrs.contracts` adapts current `Detection`,
`CandidateAlert`, and `VerifiedAlert` dataclasses to `detection.v1`,
`candidate_alert.v1`, and `verified_alert.v1` while preserving the existing
local JSONL shape. Canonical contracts carry deterministic IDs and idempotency
keys derived from stream/frame/event fields. Runtime sinks write one per-alert
`object_manifest.v1` under `manifests/` plus `object_manifest.index.jsonl`,
with `verified_alert.v1` and thumbnail `evidence_ref.v1` records by default.
`stream.v1` defines the stream-source boundary for the upcoming transport and
DeepStream work.

Acceptance criteria:

- Keep legacy Python dataclass compatibility and existing `alerts.jsonl`
  consumers.
- Publish canonical contract JSON at bus, object-store, and DeepStream service
  boundaries.
- Treat object storage as canonical for evidence assets and metadata manifests.
- Keep relational stores as optional rebuildable query projections.

### 12. Event Transport And Object Storage Interfaces

Status: initial interfaces implemented. `vrs.transport` defines
`EventTransport`, `EventMessage`, a deterministic `InMemoryEventTransport` for
tests, and Redis Streams / Kafka config naming shapes without requiring either
service. `vrs.storage` defines `ObjectStore`, `StoredObject`,
`LocalObjectStore`, and an S3/SeaweedFS-compatible URI config scaffold. Runtime
manifest writing uses the local object-store implementation.

Acceptance criteria:

- Keep unit tests free of Redis, Kafka, S3, and SeaweedFS service dependencies.
- Use Redis Streams as the edge-mode transport shape.
- Use Kafka as the production transport shape.
- Keep JSONL as audit/export fallback.
- Treat local filesystem object storage as the first concrete implementation.

### 13. DeepStream Detection Metadata Adapter

Status: native DS 8.0 worker scaffold implemented. `vrs.deepstream.adapter`
still defines the dependency-free Python metadata conversion path for tests and
fallback JSON/JSONL conversion. The C++ worker under `native/deepstream` builds
inside the DeepStream 8.0 container, runs a GStreamer pipeline, reads
`NvDsFrameMeta` and `NvDsObjectMeta` at a pad probe, and emits canonical
`detection.v1` JSONL with `source_runtime: deepstream`. GPU smoke, real PGIE
config, TensorRT engine validation, and transport publishing remain pending.

Acceptance criteria:

- Keep VLM verification outside DeepStream.
- Keep the Python detector path functional for tests, eval, and development.
- Publish DeepStream detections using the same `detection.v1` contract as the
  Python runtime.
- Use the native GStreamer/DeepStream worker to map `NvDsFrameMeta` and
  `NvDsObjectMeta` into canonical `detection.v1`.

### 14. Detector Runtime Parity Hook

Status: initial report hook implemented. `scripts/export_python_detections.py`
emits Python detector `detection.v1` JSONL for comparison clips, and
`scripts/compare_detector_parity.py` compares it with DeepStream/TensorRT
`detection.v1` output from the same clips. The comparison writes
`vrs.eval.detector_parity.v1` with class mapping, bbox IoU and pixel deltas,
confidence deltas, match/unmatched counts, and runtime capacity fields for
latency, throughput, queue drops, and GPU memory when provided.

Acceptance criteria:

- Compare outputs from the same clips and frame indexes.
- Record class mapping and per-class unmatched records.
- Record bbox and confidence deltas for matched detections.
- Carry latency, throughput, queue-drop, and GPU-memory fields without making
  production capacity claims.
- Keep `docs/runtime-matrix.md` status rows evidence-backed.

### 15. Helm Edge Profile Scaffold

Status: executable DeepStream worker baseline implemented under `charts/vrs`.
The chart has explicit `values-dev.yaml`, `values-edge.yaml`, `values-kind.yaml`,
and `values-prod.yaml` profiles, plus templates for API/runtime, DeepStream
worker, disabled verifier worker template, Redis edge bus, local PVC or
SeaweedFS object storage, metrics service, and optional ServiceMonitor. The
CPU-only kind chart mounts `/data`, provides sample metadata by ConfigMap,
removes GPU resource requests, and runs the importable Python metadata adapter
command inside the DeepStream worker deployment. GPU roles are labeled as
`vrs.ai/gpu-role: deepstream` and `vrs.ai/gpu-role: verifier`.

Acceptance criteria:

- Keep dev, single-node edge, and production cluster values explicit.
- Keep Redis as the edge bus shape and object storage as local PVC or SeaweedFS.
- Do not introduce a Kubernetes operator yet.
- Keep DeepStream and verifier workers as separate workload classes.

### 16. Control-Plane Primitives

Status: initial primitives implemented. `vrs.control.static_assignment`
converts stream manifest-style inputs to `stream.v1`, assigns streams to
DeepStream worker IDs with optional capacity limits, and renders worker config
payloads with transport and object-store settings. `vrs.control.registry` adds
an in-memory stream registry, worker health records, and queue-pressure
summaries from the existing multistream `queue_stats()` shape.

Acceptance criteria:

- Start with static stream assignment.
- Render worker configs from existing stream manifest shapes.
- Add stream registry and health reporting primitives.
- Surface queue pressure as control-plane health data.
- Keep active queue-pressure scheduling, GPU role scheduling, and Go operator
  behavior deferred until contracts and Helm profiles are stable.

### 17. Metrics Endpoint

Expose runtime health without scraping logs.

Status: implemented. Runtime configs can enable a Prometheus-compatible
`/metrics` endpoint. The endpoint exports queue depth/drop counters,
detector/verifier latency histograms, verified-alert counters labeled by stream,
class, verdict, and severity, verifier/backend error counters, privacy setup
failure counters, and sink write error counters.

Acceptance criteria:

- Prometheus or OpenTelemetry-compatible metrics.
- Queue depth/drop counters.
- Detector/verifier latency.
- Alert counts by stream, class, verdict, and severity.
- Privacy detector setup failures.

### 18. Signed Audit Log

Alerts should be tamper-evident when used for compliance or incident review.

Acceptance criteria:

- HMAC or chained hash per JSONL line.
- Key-id support for rotation.
- Verification CLI for replay/audit.

### 19. Hot Policy Reload

Operators should not restart the process for threshold-only policy changes.

Status: implemented for the single-stream pipeline as opt-in
`policy_reload`. Runtime-safe policy edits reload by file polling or SIGHUP;
invalid files and detector-vocabulary changes are rejected while the current
policy stays active.

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
