# Operations Notes

This page keeps operational details out of the README while preserving the
commands needed to run, validate, and extend VRS.

## Tamper-Evident Alert Logs

`alerts.jsonl` can optionally include a hash chain so later review can detect
modified, deleted, or reordered records. This is not encryption; alert contents
remain plaintext JSON.

```yaml
audit:
  enabled: true
  mode: hmac_sha256
  key_id: local-dev-key
  key_env: VRS_AUDIT_HMAC_KEY
```

Set the HMAC key in the runtime environment and keep it out of the config file:

```bash
export VRS_AUDIT_HMAC_KEY='replace-with-a-secret-key'
```

Verify a signed log:

```bash
python -m vrs.audit --log runs/demo/alerts.jsonl \
  --mode hmac_sha256 \
  --key-env VRS_AUDIT_HMAC_KEY
```

## Served OpenAI-Compatible Verifier

To compare the local Cosmos baseline against a served Qwen/vLLM/SGLang-style
VLM, point the verifier at an OpenAI-compatible chat-completions endpoint:

```yaml
verifier:
  enabled: true
  backend: openai_compatible
  model_id: qwen-vl-served
  base_url: http://localhost:5445/v1
  api_key_env: VRS_VLM_API_KEY
  max_new_tokens: 512
  temperature: 0.0
```

The backend sends the same system prompt, user prompt, keyframes, and JSON
schema used by the local verifier. If a server ignores `response_format`, VRS
still applies the same verifier JSON parsing and failure policy.

Smoke-test a served verifier before running the labeled bake-off:

```bash
just smoke-verifier
```

Then prepare a small local labeled-dir set and run the comparison:

```bash
just prepare-verifier-eval
just eval-verifier-bakeoff
```

## vLLM Verifier Validation

The `vllm` verifier backend is optional and must be validated on the target
CUDA host before deployment:

```bash
uv sync --extra cu128 --extra vllm
just smoke-vllm
```

The RTX 5080 validation path is pinned to `vllm==0.19.1`. That version uses
vLLM structured outputs for JSON schema enforcement and is incompatible with
the repository's `cu121` extra because it requires a newer torch stack.

If the model is gated on Hugging Face, put an authorized token in `.env`:

```bash
HF_TOKEN=hf_...
```

The `smoke-vllm` and `eval-verifier-vllm-bakeoff` recipes source `.env` before
starting Python. The token must belong to an account that has accepted access
to the gated model page; otherwise Hugging Face will still return an
authorization error.

If vLLM fails at startup with a free-memory error, lower these fields in
`configs/vllm-cosmos.yaml`:

```yaml
verifier:
  gpu_memory_utilization: 0.70
  max_model_len: 4096
```

`gpu_memory_utilization` controls how much total GPU memory vLLM reserves for
weights and KV cache. `max_model_len` controls the maximum context length and
therefore KV-cache pressure.

The smoke writes `runs/vllm-smoke/result.json` with Python, torch, CUDA, vLLM,
GPU metadata, verifier JSON validity, and backend generation stats when
available. Record that output in a benchmark note before switching production
configs to `verifier.backend: vllm`.

On vLLM 0.19.1, structured-output runs may print `xgrammar` nanobind leak
warnings at process exit. Treat those as a validation caveat to record in the
benchmark note; they did not cause the RTX 5080 smoke or eval command to fail.

Compare the transformers baseline and vLLM backend on identical prepared clips:

```bash
just prepare-verifier-eval
just eval-verifier-vllm-bakeoff
```

## Calibration Stage B

Calibration is disabled by default. Stage A writes suggestions to
`calibration_suggestions.jsonl`; Stage B can also promote those suggestions into
an operator-reviewable override export and append-only audit log:

```yaml
calibration:
  enabled: true
  apply_enabled: true
  apply_cooldown_s: 3600
  apply_audit: calibration_applied.jsonl
  apply_export: calibration_overrides.yaml
```

Stage B keeps separate scores per `(stream_id, class_name)`, enforces caps and a
per-key cooldown, writes old/new values plus verifier statistics to the audit
log, and atomically rewrites `calibration_overrides.yaml`. The running detector
policy is not mutated; use the export as the review/rollback artifact before
copying selected thresholds into the watch policy or a deployment-specific
override.

On 16 GB validation hosts, the bakeoff recipe defaults the transformers
candidate to `configs/tiny.yaml`; the BF16 `configs/default.yaml` baseline can
OOM before the vLLM candidate runs. Override `vllm_bakeoff_baseline_config` if
you are validating on a larger GPU.

## Prometheus Metrics

Metrics are disabled by default. Enable the lightweight scrape endpoint in your
runtime config:

```yaml
observability:
  metrics:
    enabled: true
    host: "0.0.0.0"
    port: 9108
```

Prometheus can then scrape `http://<host>:9108/metrics`. The endpoint exports
queue depth and drop counters for multi-stream runs, detector/verifier latency
histograms, candidate counters, verified-alert counters, verifier error
counters, and sink write error counters.

## GPU Smoke Tests

Generate synthetic plumbing-test clips:

```bash
uv run scripts/make_test_clips.py --out runs/test_clips
```

Benchmark on the local GPU:

```bash
uv run scripts/bench.py --clips runs/test_clips --out runs/bench
```

Synthetic clips validate pipeline plumbing, latency, FPS, and VRAM. They do not
prove model accuracy. For accuracy testing, use real datasets and add clips to
`configs/multistream.yaml` as `file://` sources.

## Watch Policy

Watch policy is the primary operator-maintained surface:

```yaml
watch:
  - name: falldown
    detector: ["person lying on the ground", "fallen person", "collapsed person"]
    verifier: "A person has collapsed and is lying motionless"
    severity: high
    min_score: 0.35
    min_persist_frames: 3
```

Adding a custom event is a single block. No embedding bank needs to be rescored.

Longer term, policy optimization should be offline and eval-gated: collect
verified alerts and operator feedback, mine false positives and misses, generate
structured policy diffs, evaluate against known cases, then promote only if
regression gates pass.

## Multi-Stream Notes

The multi-stream path uses one shared YOLOE instance and one shared VLM verifier
instance. Sustainable stream count depends on alert rate, verifier backend, clip
length, and GPU memory. Measure it on the target host with `scripts/bench.py`
instead of treating a stream count as portable.

Decoder backends implemented today:

- `opencv` — FFmpeg CPU decode. Works everywhere.
- `nvdec` — `cv2.cudacodec.VideoReader` for hardware-accelerated NVDEC decode;
  falls back to `opencv` when unavailable.

DeepStream is tracked as a future runtime adapter for high-density RTSP ingest,
NVDEC/NVMM pipelines, TensorRT detector execution, tracker metadata, and broker
publishing. See [VSS + SAM3 blueprint](03-vss-sam3-blueprint.md).

## Evaluation Reports

`scripts/eval.py` writes `<out>/report.json` using the stable schema version
`vrs.eval.report.v1`.

Regenerate the committed mini baseline only after an intentional fixture or
metric-contract update:

```bash
uv run python scripts/write_eval_baseline.py --out baselines/eval/report.json
```

Compare a candidate report against the baseline:

```bash
uv run python -m vrs.eval.ci \
  --baseline baselines/eval/report.json \
  --current runs/eval/report.json \
  --max-f1-drop 0.02
```
