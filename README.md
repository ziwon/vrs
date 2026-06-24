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
and [VAST Data VSS Blueprint](https://github.com/vast-data/vss-blueprint/):
perception first, VLM-based alert verification, and optional higher-level
incident reasoning. It is not a VSS clone. The goal is a smaller, hackable,
local-GPU-oriented Video Reasoning System that can be evaluated, customized,
and embedded into CCTV / VMS / edge-appliance environments.

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

## Dashboard

![VRS Console live alerts view showing detector candidates, verifier verdicts, confidence, and falldown thumbnails.](docs/assets/vrs-console-live-alerts.png)

The local console reads `runs/live/alerts.jsonl` and thumbnails through a
CPU-light FastAPI backend. It shows detector candidates, VLM verdicts,
confidence, rationale, live RTSP state, and the two-stage cascade.

The same console can inspect offline validation runs under `runs/`, including
MIVIA fire/smoke checks. Those runs exercise the full two-stage result: YOLOE
proposes fire/smoke boxes from video frames, event-state promotes stable
candidates, and the VLM verifier records the final verdict, confidence,
rationale, and missed-event signals in the alert drawer.

![VRS Console streams view showing the local falldown RTSP source and latest live keyframe.](docs/assets/vrs-console-streams.png)

![VRS Console cascade view showing the decode, YOLOE, event-state, VLM verifier, and sink stages.](docs/assets/vrs-console-cascade.png)

## Quick Start

![Animated diagram of the local Docker Compose runtime: falldown MP4 to RTSP, GPU inference, runs/live artifacts, FastAPI, and VRS Console.](docs/assets/vrs-local-runtime.svg)

Start the local RTSP/API/UI stack:

```bash
docker compose up --build
```

Open <http://127.0.0.1:5173>. The Compose stack publishes
`runs/pr-integration/clips/falldown_test.mp4` as:

```text
rtsp://127.0.0.1:8554/falldown
```

Run the full inference profile with a Hugging Face token:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN / HUGGING_FACE_HUB_TOKEN
docker compose -f docker-compose.yaml -f docker-compose.hf-local.yaml \
  --profile inference up --build
```

Convenience commands:

```bash
just local-up
just local-logs -f inference
just local-down
```

For a local multi-stream RTSP demo, run each long-lived process in its own
terminal:

```bash
just local-rtsp-publish-all   # falldown, fire, smoke, weapon, optional MIVIA fire
just local-multistream-run    # consumes configs/local-rtsp-streams.yaml
just web-api
just web-ui
```

Stop the local stack with:

```bash
just stop-local
```

See [docs/local-web-ui.md](docs/local-web-ui.md) for the full local workflow,
RTSP checks, API checks, token setup, and troubleshooting.

## Why this design

Classical CLIP-style classifiers (the "encode each frame and cosine-match a
prompt bank" approach) work, but they require continuous **prompt
maintenance**: every new event type, every new camera, every site-specific
quirk needs new prompts and re-scoring. They also produce only frame-level
scores, not localizations.

### Why not embedding-only business logic

VRS also avoids making precomputed embeddings the primary home for business logic. Embedding lookup can be very fast, but operational semantics often end up hidden in manually maintained vectors, thresholds, and category-specific matching rules. As sites, camera angles, lighting conditions, and false-positive patterns change, engineers must continuously update those assets.

VRS keeps business intent explicit and policy-driven. Event definitions live in human-readable watch policies, while models remain execution components: the detector proposes candidates, event-state checks temporal stability, optional gates estimate risk, and the VLM verifier makes the final decision using the policy definition and video context.

```text
Business intent       -> watch policies
Fast perception       -> open-vocabulary detector
Risk / uncertainty    -> event-state and optional gates
Final decision        -> VLM verifier
```

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

![Animated diagram of the VRS two-stage architecture: RTSP input, reader, YOLOE fast path, event-state promotion, VLM verifier slow path, and alert sinks.](docs/assets/vrs-architecture-flow.svg)

The detector runs on every sampled frame, while the verifier only runs after
event-state promotes a stable candidate. This keeps the local GPU budget focused
on real alert decisions instead of spending VLM time on quiet frames.

## Policy flow

Watch policies live under [`configs/policies`](configs/policies/README.md).
They define the user-editable event registry: detector prompts, verifier
definitions, severity, confidence floors, persistence, and optional verifier
context windows.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#0f172a", "mainBkg": "#243044", "secondBkg": "#1f2937", "primaryColor": "#243044", "primaryBorderColor": "#475569", "primaryTextColor": "#f8fafc", "lineColor": "#38bdf8", "textColor": "#f8fafc", "fontFamily": "sans-serif"}}}%%
flowchart LR
    A[Policy YAML] --> B[WatchPolicy]
    B --> C[YOLOE vocabulary]
    B --> D[Event thresholds]
    B --> E[VLM definitions]
    C --> F[Fast-path detections]
    D --> G[CandidateAlert]
    F --> G
    E --> H[VLM verifier]
    G --> H
    H --> I[VerifiedAlert]

    classDef source fill:#111827,stroke:#334155,color:#f8fafc,stroke-width:1px
    classDef policy fill:#243044,stroke:#475569,color:#f8fafc,stroke-width:1.5px
    classDef fast fill:#1f3a36,stroke:#84cc16,color:#f8fafc,stroke-width:1.5px
    classDef gate fill:#3b3420,stroke:#f59e0b,color:#f8fafc,stroke-width:1.5px
    classDef verifier fill:#2f2945,stroke:#a78bfa,color:#f8fafc,stroke-width:1.5px
    classDef output fill:#193244,stroke:#38bdf8,color:#f8fafc,stroke-width:1.5px

    class A source
    class B policy
    class C,F fast
    class D,G gate
    class E,H verifier
    class I output

    linkStyle 0 stroke:#84cc16,stroke-width:3px
    linkStyle 1 stroke:#84cc16,stroke-width:3px
    linkStyle 2 stroke:#f59e0b,stroke-width:3px
    linkStyle 3 stroke:#a78bfa,stroke-width:3px
    linkStyle 4 stroke:#84cc16,stroke-width:3px
    linkStyle 5 stroke:#f59e0b,stroke-width:3px
    linkStyle 6 stroke:#f59e0b,stroke-width:3px
    linkStyle 7 stroke:#a78bfa,stroke-width:3px
    linkStyle 8 stroke:#38bdf8,stroke-width:3px
```

At runtime, policy entries such as `fire`, `smoke`, and `falldown` are expanded
into YOLOE open-vocabulary classes, mapped back to stable event names, promoted
only after temporal persistence, and finally verified by the VLM before being
written to JSONL, thumbnails, and the local console.

## Documentation

- [System review](docs/01-system-review.md) — current implementation status,
  known gaps, and engineering review.
- [Roadmap](docs/02-roadmap.md) — near-term prioritized work.
- [Local web UI workflow](docs/local-web-ui.md) — Docker Compose, RTSP,
  `.env`, Hugging Face cache, API checks, and UI troubleshooting.
- [Policy model](configs/policies/README.md) — watch-policy schema, runtime
  flow, UI-driven editing, validation, reload strategy, and scenario-policy
  direction.
- [Operations notes](docs/operations.md) — audit signing, served verifier,
  metrics, GPU smoke tests, policies, and evaluation reports.
- [VSS + SAM3 blueprint](docs/03-vss-sam3-blueprint.md) — long-term
  vendor-neutral platform direction, including optional SAM3 workers,
  DeepStream runtime adapter planning, platform contracts, semantic search, and
  enterprise storage.
- [Runtime validation matrix](docs/runtime-matrix.md) — validated,
  unvalidated, and planned GPU/runtime profiles.

## GPU Setup

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

For the W4A16 verifier profile used by `configs/tiny.yaml`, include the quant
extra:

```bash
uv sync --python 3.11 --extra cu128 --extra quant
```

## Pipeline Commands

Single MP4:
```bash
uv run scripts/run_mp4.py \
  --video /path/to/cctv.mp4 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/demo
```

Single RTSP:
```bash
uv run scripts/run_rtsp.py \
  --rtsp rtsp://user:pass@cam.local:554/stream1 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/live
```

Multi-stream:
```bash
uv run scripts/run_multistream.py \
  --config  configs/default.yaml \
  --policy  configs/policies/safety.yaml \
  --streams configs/multistream.yaml \
  --out     runs/live
```

Outputs:

- `runs/<name>/alerts.jsonl` — one JSON per verified alert (verdict, confidence, bbox, trajectory, rationale, thumbnail path)
- `runs/<name>/thumbnails/*.jpg` — one event image per alert, with detector/verifier overlays
- `runs/<name>/annotated.mp4` — optional debug/demo overlay video when `sink.write_annotated: true`

## Operations

- Watch-policy changes live in `configs/policies/safety.yaml`; adding a custom
  event is one detector prompt list plus one verifier sentence.
- Metrics, audit signing, served VLM verifier setup, GPU smoke tests, and eval
  report commands are covered in [docs/operations.md](docs/operations.md).
- Runtime and GPU profile status is tracked in
  [docs/runtime-matrix.md](docs/runtime-matrix.md).

| Profile | Detector | Verifier | Notes |
|---------|----------|----------|-------|
| `default.yaml` | YOLOE-L FP16 | Cosmos-Reason2-2B BF16 | Accuracy-oriented local profile; validate memory on target GPU. NVIDIA's reference model card lists 24 GB minimum. |
| `tiny.yaml` | YOLOE-S FP16 | Cosmos-Reason2-2B W4A16 | Intended for 8-16 GB cards / Jetson-class deployments after quantized-runtime validation. |

## Layout

```
.
├── web/                    Static VRS Console frontend
├── docker/                 nginx config for frontend/API proxying
├── docs/                   Current docs, operations notes, benchmarks, archive
├── configs/                Runtime configs, stream manifests, watch policies
├── scripts/                CLI runners, fixtures, benchmarks, eval helpers
├── docker-compose.yaml     RTSP, backend, frontend, and inference workflow
├── Dockerfile.*            Backend, frontend, and GPU inference images
└── vrs/
    ├── web/                FastAPI artifact browser for dashboard data
    ├── ingest/             RTSP/mp4 frame iterator
    ├── triage/             YOLOE detector, tracking, event-state queue
    ├── verifier/           VLM prompts and structured-output parsing
    ├── runtime/            transformers, vLLM, OpenAI-compatible backends
    ├── sinks/              JSONL, thumbnails, optional annotated video
    ├── multistream/        N-stream cascade workers and queues
    ├── pipeline.py         Single-stream cascade orchestration
    └── schemas.py          Frame, detection, candidate, verified alert models
```
