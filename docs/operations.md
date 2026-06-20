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
