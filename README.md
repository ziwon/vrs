# VRS — Video Reasoning System

[![CI](https://img.shields.io/github/actions/workflow/status/ziwon/vrs/ci.yml?branch=main&label=CI&logo=github)](https://github.com/ziwon/vrs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A modern, two-stage CCTV / video-understanding pipeline for a **single local GPU**.

VRS is inspired by the architecture patterns in
[NVIDIA Video Search and Summarization (VSS)](https://docs.nvidia.com/vss/latest/)
and the Public Safety Blueprint: perception first, VLM-based alert verification,
and optional higher-level incident reasoning. It is not a VSS clone. The goal is
a smaller, hackable, local-GPU-oriented Video Reasoning System that can be
evaluated, customized, and embedded into CCTV / VMS / edge-appliance
environments.

The deployment target is 16 GB cards when the verifier is quantized or otherwise
capacity-tested. For BF16 Cosmos-Reason2-2B, validate on the target host first:
NVIDIA's 2026 model card lists a 24 GB minimum for the reference inference path.

- **Fast path:** open-vocabulary detection with **YOLOE-L** (~6 ms / frame on a T4).
  Add or change event classes by editing one YAML line — no prompt-bank curation,
  no per-class fine-tuning.
- **Slow path:** a pluggable VLM verifier. The current baseline is
  **NVIDIA Cosmos-Reason2-2B**, but 2026 research and internal benchmarks show
  it should not be treated as the final verifier. Qwen3.5/Qwen3.6-class VLMs
  are priority candidates for side-by-side evaluation.

## Why this design

Classical CLIP-style classifiers (the "encode each frame and cosine-match a
prompt bank" approach) work, but they require continuous **prompt
maintenance**: every new event type, every new camera, every site-specific
quirk needs new prompts and re-scoring. They also produce only frame-level
scores, not localizations.

VRS uses two practical 2026-era baselines, with the verifier intentionally
kept swappable:

| Stage | Model | Why |
|-------|-------|-----|
| Detect | **Ultralytics YOLOE-L** (`yoloe-11l-seg.pt` by default) | Open-vocabulary text prompts, returns bounding boxes, no per-site retraining loop |
| Reason | **nvidia/Cosmos-Reason2-2B** baseline | Physical-reasoning specialization, FPS=4 video path, bbox/point/trajectory-oriented prompting. Treat as a baseline, not the expected winner. |

Sources:
- NVIDIA VSS — [Video Search and Summarization documentation](https://docs.nvidia.com/vss/latest/), used as architectural inspiration for perception-first video AI, alert verification, and optional incident reasoning patterns.
- YOLOE — Ultralytics docs and CVPR'25 paper. Ultralytics also publishes newer
  YOLOE-26 models; migrate only after eval confirms a gain for the active policy.
- Cosmos-Reason2 — NVIDIA docs and `nvidia/Cosmos-Reason2-2B` model card.
- Qwen — Qwen3.5/Qwen3.6 official releases report stronger multimodal
  foundation-model capability than earlier Qwen3-VL-era models. Because
  Cosmos-Reason2-2B is derived from Qwen3-VL-2B, Qwen3.5/Qwen3.6-class models
  should be evaluated as verifier backends before production model lock-in.

## Architecture

```
                ┌─── FAST PATH (every frame, ~6 ms) ─────────────┐
RTSP/mp4 ─► Reader ─► YOLOE-L ─► per-class score+bbox ─► EventState
                       │                                          │
                       │       (VRAM ~0.5 GB)                     │ candidate
                       └──────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                ┌─── SLOW PATH (only on candidate) ──────────────┐
                │ VLM verifier backend                           │
                │  • default: Cosmos-Reason2-2B baseline         │
                │  • candidates: Qwen3.5/Qwen3.6 served VLMs     │
                │  • returns true_alert / false_alarm / FN-class │
                └────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                              alerts.jsonl + thumbnails/*.jpg
```

## Install

```bash
uv python install 3.11

# Pick the right torch build for your GPU architecture:
#   * Blackwell (RTX 5080/5090, B100, GB100) → CUDA 12.8+ / torch 2.6+
#   * Hopper   (H100, H200)                 → CUDA 12.4+
#   * Ada      (RTX 4080/4090, L4, L40)     → CUDA 12.1+
uv sync --python 3.11 --extra cu128     # Blackwell
# or:
# uv sync --python 3.11 --extra cu121   # Ada/Ampere
```

Always pick one CUDA extra on GPU hosts; a bare `uv sync` can satisfy
transitive `torch` requirements without selecting the deployment-specific
PyTorch wheel.

### RTX 5080 / Blackwell note

RTX 5080 (GB205, Blackwell, SM 12.0) needs **CUDA 12.8+ and torch 2.6+** for
native kernel support. If you get `no kernel image is available for execution
on the device` when YOLOE or Cosmos starts, your torch is too old — sync with
the `cu128` extra above.

Cosmos-Reason2-2B and YOLOE should be validated on the exact Blackwell host and
runtime you intend to ship. The BF16 model weights are small, but video tokens,
KV cache, and framework overhead can dominate memory; use the W4A16 profile or
a served backend if BF16 does not fit.

### W4A16 profile (≤8 GB cards / Jetson)

```bash
uv sync --extra cu128 --extra quant
```

## Run

**Single mp4:**
```bash
uv run scripts/run_mp4.py \
  --video /path/to/cctv.mp4 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/demo
```

**Single RTSP:**
```bash
uv run scripts/run_rtsp.py \
  --rtsp rtsp://user:pass@cam.local:554/stream1 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/live
```

**Multi-stream (N cameras on one GPU):**
```bash
uv run scripts/run_multistream.py \
  --config  configs/default.yaml \
  --policy  configs/policies/safety.yaml \
  --streams configs/multistream.yaml \
  --out     runs/live
```

List your cameras in `configs/multistream.yaml`. One shared YOLOE and one
shared verifier backend serve every stream; per-stream outputs land under
`runs/live/<stream_id>/{alerts.jsonl, thumbnails/*.jpg}`. Annotated MP4 output
is still available as an opt-in debug/demo artifact.

Outputs:

- `runs/<name>/alerts.jsonl` — one JSON per verified alert (verdict, confidence, bbox, trajectory, rationale, thumbnail path)
- `runs/<name>/thumbnails/*.jpg` — one event image per alert, with detector/verifier overlays
- `runs/<name>/annotated.mp4` — optional debug/demo overlay video when `sink.write_annotated: true`

### Served OpenAI-compatible verifier

To compare the local Cosmos baseline against a served Qwen/vLLM/SGLang-style
VLM, point the verifier at an OpenAI-compatible chat-completions endpoint:

```yaml
verifier:
  enabled: true
  backend: openai_compatible
  model_id: qwen-vl-served
  base_url: http://localhost:8000/v1
  api_key_env: VRS_VLM_API_KEY
  max_new_tokens: 512
  temperature: 0.0
```

The backend posts to `${base_url}/chat/completions` with the same system prompt,
user prompt, and keyframe list used by the local verifier. Each BGR keyframe is
JPEG-encoded and sent as a `data:image/jpeg;base64,...` `image_url` content item
after the text prompt. When the verifier supplies its JSON schema, the backend
passes it as `response_format: {type: "json_schema", ...}`; if a server ignores
that field, VRS still applies the same verifier JSON parsing and failure policy
to the returned text.

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
histograms, candidate and verified-alert counters, verifier error counters, and
sink write error counters.

Example scrape config:

```yaml
scrape_configs:
  - job_name: vrs
    static_configs:
      - targets: ["vrs-appliance.local:9108"]
```

## Smoke-test on your GPU (e.g. RTX 5080)

1. **Generate synthetic plumbing-test clips** (no network, no datasets):

   ```bash
   uv run scripts/make_test_clips.py --out runs/test_clips
   ```

   This writes 4 FHD mp4s — `fire_test.mp4`, `smoke_test.mp4`,
   `falldown_test.mp4`, `weapon_test.mp4`. They use programmatic OpenCV
   drawing (noise patches, Gaussian blobs, stick figures, silhouettes).

   **They test the pipeline, not the models.** YOLOE's real-world
   training means it will usually *not* raise detections on these clips —
   that's expected. The bench script still gives you real latency, FPS,
   and VRAM numbers.

2. **Benchmark on the local GPU**:

   ```bash
   uv run scripts/bench.py --clips runs/test_clips --out runs/bench
   ```

   You'll see per-clip single-stream throughput, then a multi-stream run
   (4 clips × 4 replicas = 16 concurrent streams by default) with peak
   VRAM and queue drop counts. A JSON report is saved to
   `runs/bench/bench_report.json`.

3. **For accuracy testing**, use real datasets:
   - Fire/smoke: [FireNet](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection), [D-Fire](https://github.com/gaiasd/DFireDataset)
   - Falldown: [Le2i Fall Detection](http://le2i.cnrs.fr/Fall-detection-Dataset), [UP-Fall Detection](https://sites.google.com/up.edu.mx/har-up/)
   - Multi-class CCTV anomaly: [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)

   Drop the clips into a directory, add them to `configs/multistream.yaml`
   as `file://` sources, and re-run `uv run scripts/bench.py`.

   For D-Fire detector-only evaluation, see [docs/dfire-dataset.md](docs/dfire-dataset.md).

## Watch Policy — the only place you maintain

```yaml
# configs/policies/safety.yaml
watch:
  - name: fire
    detector: ["fire", "open flame", "burning object"]   # YOLOE text prompts
    verifier: "Open flames or active fire indoors"        # VLM verifier prompt
    severity: critical
    min_score: 0.30
    min_persist_frames: 2
  - name: smoke
    detector: ["smoke", "smoke cloud", "billowing smoke"]
    verifier: "Thick smoke filling a space"
    severity: high
    min_score: 0.25
    min_persist_frames: 2
  - name: falldown
    detector: ["person lying on the ground", "fallen person", "collapsed person"]
    verifier: "A person has collapsed and is lying motionless"
    severity: high
    min_score: 0.35
    min_persist_frames: 3
  - name: weapon            # custom — added in 5 seconds
    detector: ["handgun", "knife", "rifle"]
    verifier: "A person holding or brandishing a weapon"
    severity: critical
    min_score: 0.40
    min_persist_frames: 1
```

Adding a custom event is a single block. No bank to score, no embeddings to
recompute, no thresholds to grid-search.

## Policy Optimization Roadmap

Today, VRS uses a hand-authored watch policy to decide which detector-side
events should become verification candidates. Longer term, the verifier policy
should become more structured and data-driven instead of growing into an
unbounded collection of customer-specific prompt files.

The planned direction is:

```text
VerifiedAlert + thumbnails/clips + operator feedback
  -> CaseStore
  -> false-positive / missed-event mining
  -> stronger LLM proposes structured PolicyDiff candidates
  -> scenario evaluation compares baseline vs candidate
  -> promote only if regression gates pass
  -> versioned rollout / rollback
```

This is **not** runtime self-modification. VRS should not change prompts live
just because one alert was wrong. The intended loop is offline and eval-gated:
collect evidence, generate a candidate policy update, evaluate it against known
cases, then promote it only if false positives improve without unacceptable
recall or uncertainty regressions.

This roadmap also reframes verifier prompts as rendered artifacts from
structured, versioned scenario policies. A future policy pack may define fields
such as `normal_conditions`, `abnormal_conditions`, `false_positive_hints`,
`required_evidence`, and `uncertain_when`; prompt templates can then render those
fields into the VLM verifier instructions. The goal is to reduce manual prompt
drift while still allowing customer/site-specific semantics to be captured and
evaluated.

## VRAM profiles

See [Runtime validation matrix](docs/runtime-matrix.md) for the current
validated, unvalidated, and planned GPU/runtime combinations. Treat the table
below as configuration intent until a profile has a linked benchmark note.

| Profile | Detector | Verifier | Notes |
|---------|----------|----------|-------|
| `default.yaml` | YOLOE-L FP16 | Cosmos-Reason2-2B BF16 | Accuracy-oriented local profile; validate memory on target GPU. NVIDIA's reference model card lists 24 GB minimum. |
| `tiny.yaml` | YOLOE-S FP16 | Cosmos-Reason2-2B W4A16 | Intended for 8-16 GB cards / Jetson-class deployments after quantized-runtime validation. |

## Multi-stream architecture

The multistream path is designed around one shared YOLOE instance and one
shared VLM verifier instance. Sustainable stream count depends on alert
rate, verifier backend, clip length, and GPU memory; measure it with
`uv run scripts/bench.py` on the target host instead of treating a stream count as
portable. See `vrs/multistream/` for the topology:

```
RTSP[i] ─► DecoderThread[i] ──┐
                              ▼ bounded FrameQueue (drop-oldest)
                    DetectorWorker  (batched YOLOE, per-stream EventStateQueue)
                              │
                              ▼ bounded CandidateQueue
                    VerifierWorker  (VLM verifier, drop-oldest on overflow)
                              │
                              ▼
                    SinkWorker[i]  ─► runs/<out>/<stream_id>/{alerts.jsonl, thumbnails/*.jpg}
```

Decoder backends (`multistream.decoder_backend`):
- `opencv` (default) — FFmpeg CPU decode. Works everywhere.
- `nvdec` — `cv2.cudacodec.VideoReader` for hardware-accelerated NVDEC decode.
  Requires OpenCV built with CUDA; gracefully falls back to `opencv` otherwise.
- `deepstream` — reserved. A full DeepStream 8.0 path (pyds → `nvstreammux` →
  `nvinfer` with TRT-exported YOLOE → custom verifier serving path) would
  keep frames in NVMM (zero-copy) end-to-end; it also requires TRT export of
  both models. Out of scope for this release — the Reader interface is small
  enough that adding it is additive.

## Layout

```
vrs/
├── ingest/        RTSP/mp4 frame iterator (single-stream, OpenCV)
├── triage/        YOLOE detector (+ batched inference), per-stream event-state queue
├── verifier/      VLM verifier prompts + structured-output parsing
├── policy/        Plain-English watch policy parser
├── runtime/       VLM runtime backends (Cosmos baseline today)
├── sinks/         JSONL writer, event thumbnails, optional annotated-video writer
├── multistream/   N-stream cascade: decoders, workers, queues, pipeline
├── pipeline.py    Single-stream cascade orchestration
└── schemas.py     Frame / Detection / CandidateAlert / VerifiedAlert
```

## Evaluation Reports

`scripts/eval.py` writes `<out>/report.json` using the stable schema version
`vrs.eval.report.v1`.

Top-level sections:
- `schema_version` — explicit report-contract identifier for compatibility checks.
- `run` — dataset name, run ID, timestamp, mode, and config/policy paths.
- `models` — detector and verifier backend/model identifiers from the active config.
- `metrics` — overall and per-class precision, recall, F1, TP, FP, and FN.
- `latency` — reserved stable slots for detector/verifier latency percentiles.
- `runtime` — lightweight environment metadata such as Python version.
- `quality_signals` — verifier flip rate, false-negative flag rate, and reserved diagnostic slots.
- `per_video` — optional per-clip breakdown using the same metrics/quality shape.

Today the CLI emits `mode: "full_cascade"` for the current detector+verifier
path. A future detector-only scoring mode can reuse the same report schema with
`mode: "detector_only"` instead of changing the JSON contract again.

`run.created_at` and `runtime.*` are diagnostic — they vary across runs and
machines and are not part of the regression contract. The CI gate
(`uv run python -m vrs.eval.ci`) only compares `metrics` and `quality_signals`, so
committed baselines are immune to clock and Python-version drift.

The committed mini baseline lives at `baselines/eval/report.json`. Regenerate it
only after an intentional fixture or metric-contract update:

```bash
uv run python scripts/write_eval_baseline.py --out baselines/eval/report.json
```

Compare a candidate report against the baseline with:

```bash
uv run python -m vrs.eval.ci \
  --baseline baselines/eval/report.json \
  --current runs/eval/report.json \
  --max-f1-drop 0.02
```
