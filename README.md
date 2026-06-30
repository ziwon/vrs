# VRS — Video Reasoning System

[![CI](https://img.shields.io/github/actions/workflow/status/ziwon/vrs/ci.yml?branch=main&label=CI&logo=github)](https://github.com/ziwon/vrs/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit) [![DeepStream](https://img.shields.io/badge/DeepStream-8.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/deepstream-sdk)

VRS is an educational open-source Video Reasoning System inspired by VSS-style architectures.

It uses a two-stage pipeline: a fast perception worker first detects candidate events, then a slower VLM verifier reviews visual evidence and policy context before producing verified alerts. The goal is to make modern video reasoning systems understandable, hackable, and portable across local development, single-GPU edge nodes, and Kubernetes clusters.

VRS is learning-oriented, but it is not only a local demo. The repository includes production-shaped boundaries for DeepStream 8 data-plane experiments, Redis-style transport, S3-compatible evidence storage, FastAPI control APIs, a static operator console, and Helm-based deployment. You can start locally with Docker Compose, validate Kubernetes wiring with kind or microk8s, and run a single-GPU edge-style deployment with k3s.

## Why VRS

Modern video AI systems are easiest to reason about when perception, event state, verification, evidence, and operator APIs are kept as explicit boundaries. VRS packages those boundaries into a small OSS system that can be studied locally and then mapped onto Kubernetes without changing the core runtime contracts.

VRS is useful for:

- learning VSS-inspired video reasoning architecture in a compact codebase
- prototyping open-vocabulary video alert policies
- inspecting detector candidates, verifier decisions, thumbnails, manifests, and run artifacts
- testing the migration path from Python iteration to a DeepStream-oriented data plane
- experimenting with production-shaped Kubernetes components on a single GPU machine

## Two-Stage Video Reasoning Pipeline

The core runtime is a perception-first, verification-second pipeline:

```text
video source
  -> perception worker
  -> event-state
  -> verifier
  -> evidence
  -> API
  -> console
```

![Animated diagram of the VRS two-stage architecture: RTSP input, reader, YOLOE fast path, event-state promotion, VLM verifier slow path, and alert sinks.](docs/assets/vrs-architecture-flow.svg)

The first stage is optimized for fast candidate generation from video streams. The second stage uses policy text and visual context to accept, reject, or refine those candidates into higher-quality alerts. The same conceptual pipeline can run as Python workers for fast iteration or as a DeepStream 8 data plane for production-oriented experiments.

## Run Shapes

VRS is organized around stable runtime boundaries rather than one specific machine layout. The expected progression is:

```text
Compose for learning
  -> kind or microk8s for portable Kubernetes wiring
  -> k3s or single-node edge Kubernetes with GPU workers
  -> multi-node Kubernetes using platform-native routing, storage, and observability
```

The same runtime model can run in several forms:

| Target | Runtime Shape | Purpose |
|--------|---------------|---------|
| Local workstation | Docker Compose plus local files | Learn the pipeline, inspect artifacts, iterate on policies |
| Kubernetes smoke test | kind or microk8s | Validate Helm wiring without requiring a GPU node |
| Single-GPU edge node | k3s or appliance Kubernetes with GPU workers | Run a compact production-shaped deployment |
| GPU data plane | DeepStream worker plus API/console/control plane | Move ingest and detection into the native NVIDIA data plane |
| Multi-node Kubernetes | GPU workers plus API, console, transport, storage, metrics | Scale data-plane and control/API workloads independently |

The runtime is intentionally split so that components can be replaced without changing the whole system. For example, a Python detector can be replaced by a DeepStream worker as long as both emit the same detection contract.

## What Is Implemented

| Area | Status |
|------|--------|
| Python local pipeline | Implemented for MP4, RTSP, multi-stream, alerts, thumbnails, manifests |
| Policy model | Implemented with YAML watch policies and verifier prompt rendering |
| API and console | Implemented and packaged for Docker Compose and Helm |
| Storage boundary | Local filesystem and S3-compatible abstraction implemented |
| Transport boundary | In-memory and Redis Streams adapter shape implemented |
| Helm chart | API, console, DeepStream worker, Redis, local PVC, SeaweedFS, metrics |
| DeepStream 8 worker | Native C++ worker, `vrsmeta`, YOLOE parser/configs, JSONL bridge |
| DeepStream detector acceptance | In progress; parity and dataset validation still gate production approval |

VRS is useful today for learning, prototyping policies, inspecting artifacts, running local evaluation, and validating deployment wiring. The native DeepStream path is the intended production data plane, but it should not be treated as fully accepted until detector parity and end-to-end validation gates are satisfied.

## Kubernetes And Helm

VRS is designed to run as ordinary Kubernetes workloads rather than as a cloud-specific appliance. The Helm chart separates data-plane workers, API, console, transport, storage, and metrics so that a single-node k3s or microk8s deployment can grow into a larger Kubernetes environment without changing the runtime contracts.

Platform-specific choices such as Gateway/API routing, managed Redis, external object storage, secret management, GPU node pools, and observability are intentionally left to the target Kubernetes platform. This repository provides portable workload boundaries and contracts; platform teams can map those boundaries onto their own cluster standards.

## Quick Start

Start the local RTSP/API/console stack:

```bash
docker compose up --build
```

Open <http://127.0.0.1:5173>. The default stack publishes a local falldown clip as RTSP, exposes the browser-playable HLS preview through MediaMTX, serves matching demo run artifacts through the FastAPI backend, and opens the static VRS Console through the same-origin `/api/*` proxy.

The default `local-rtsp-demo` run is a fixture aligned with the sample stream. It is meant for UI and plumbing validation; fresh model-backed alerts are written to the `live` run when the inference profile or a manual pipeline is running.

To try another MP4, put it under [`clips`](clips/README.md) and select it with `VRS_SAMPLE_CLIP`:

```bash
cp /path/to/site-camera-sample.mp4 clips/site-camera-sample.mp4
VRS_SAMPLE_CLIP=site-camera-sample.mp4 docker compose up --build
```

You can also download a licensed test video into `clips/` with `curl` or `yt-dlp`; see [`clips/README.md`](clips/README.md) for the exact commands and the inference-profile follow-up.

The Compose stack publishes the selected MP4 with `ffmpeg` into MediaMTX. The equivalent command is documented in [`clips/README.md`](clips/README.md) for debugging or manual RTSP tests.

For GPU inference, k3s validation, and Helm profiles, follow the deployment guides below.

## Deployment Guides

VRS keeps deployment-specific mechanics in separate documents. The README describes the portable shape; the linked guides provide exact commands.

| Target | Guide |
|--------|-------|
| Local Docker Compose | [docs/local-console.md](docs/local-console.md) |
| kind / microk8s smoke test | [docs/kind-validation.md](docs/kind-validation.md) |
| Single-node GPU k3s validation | [docs/k3s-gpu.md](docs/k3s-gpu.md) |
| Helm profiles | [docs/helm.md](docs/helm.md) |
| DeepStream worker | [docs/deepstream-worker.md](docs/deepstream-worker.md) |
| Operations and validation | [docs/operations.md](docs/operations.md) |

## Console

![VRS Console live alerts view showing detector candidates, verifier verdicts, confidence, and falldown thumbnails.](docs/assets/vrs-console-live-alerts.png)

The VRS Console is a static operator surface. It does not own business logic. It reads the API through same-origin `/api/*` proxying and lets users inspect runs, alerts, streams, policy definitions, thumbnails, and verifier rationale.

![VRS Console streams view showing the local falldown RTSP source and latest live keyframe.](docs/assets/vrs-console-streams.png)

## Policy Model

Watch policies are the user-editable event registry. A policy defines:

- stable event name
- detector vocabulary
- verifier definition
- severity
- confidence and persistence thresholds
- optional context-window behavior

```text
business intent
  -> watch policy
  -> detector vocabulary
  -> event-state thresholds
  -> verifier prompt
  -> verified alert
```

Policies live under [`configs/policies`](configs/policies/README.md).

## Interfaces And Contracts

The most important portability surface in VRS is the interface set. These contracts are shared across Docker Compose, k3s, microk8s, kind, and larger Kubernetes deployments.

| Interface | Standard Surface | Current Implementations |
|-----------|------------------|-------------------------|
| Video input | RTSP, MP4/file source, stream manifest | Python readers, Compose RTSP fixture, DeepStream file-source examples |
| Detection output | `detection.v1` | Python detector export, DeepStream `vrsmeta`, JSONL bridge |
| Candidate event | `candidate_alert.v1` | Event-state promotion in Python pipeline |
| Verified event | `verified_alert.v1` | VLM verifier output and JSONL sinks |
| Evidence reference | `evidence_ref.v1` | Thumbnails, manifests, local files, S3-compatible refs |
| Object manifest | `object_manifest.v1` | Per-alert manifest writer and append index |
| Storage | local filesystem, S3-compatible object storage | Local store, SeaweedFS profile, external S3-compatible config |
| Transport | logical event stream | In-memory test transport, Redis Streams adapter shape |
| API | `/api/*` artifact/runtime API | FastAPI service |
| Console | same-origin `/api/*` operator surface | Static VRS Console served by nginx |
| Deployment | declarative runtime profile | Docker Compose, Helm values profiles |

The JSON schemas live under [`contracts/schemas`](contracts/schemas). Runtime adapters live under `vrs/storage`, `vrs/transport`, `vrs/api`, and `vrs/deepstream`.

## System Components

| Component | Responsibility |
|-----------|----------------|
| Detector / DeepStream worker | Decode video, run perception, emit `detection.v1` |
| Event-state | Convert noisy frame detections into stable candidate events |
| Verifier | Use policy text and visual context to accept, reject, or refine an alert |
| Sinks | Write JSONL alerts, thumbnails, manifests, and optional annotated video |
| Storage | Store evidence locally or through S3-compatible object storage |
| Transport | Move runtime events between data-plane and control/API services |
| API | Expose run artifacts, policy, streams, thumbnails, and tail polling |
| Console | Provide the operator-facing inspection surface |
| Helm chart | Package edge and production-shaped deployments |

## GPU And Model Strategy

VRS is not tied to one GPU SKU. The local Python path can run on different NVIDIA cards depending on detector and verifier choices. The DeepStream path targets NVIDIA GPU nodes with DeepStream 8.

| Runtime | Role |
|---------|------|
| Python detector path | Fast iteration, policy development, evaluation, fallback local runs |
| Served verifier path | Test Qwen/OpenAI-compatible/vLLM-style verifier backends |
| DeepStream 8 path | Production-oriented video ingest and detector metadata export |
| TensorRT / TensorRT-LLM work | Optimization path for accepted detector/verifier runtimes |

For CUDA/PyTorch setup details, see [docs/runtime-matrix.md](docs/runtime-matrix.md) and [docs/operations.md](docs/operations.md).

## Repository Layout

```text
.
├── charts/vrs/             Helm chart for dev, kind, edge, and production profiles
├── console/                Static VRS Console operator surface
├── configs/                Runtime configs, stream manifests, policies, DS configs
├── contracts/              Versioned JSON schemas for platform boundaries
├── docs/                   Architecture, operations, benchmarks, validation notes
├── native/deepstream/      DS8 C++ worker, vrsmeta plugin, YOLOE parser
├── scripts/                CLI runners, eval helpers, parity and TRT tooling
├── vrs/
│   ├── api/                FastAPI artifact and runtime-state API
│   ├── deepstream/         JSONL bridge and DeepStream adapter helpers
│   ├── multistream/        N-stream cascade workers and queues
│   ├── policy/             Watch-policy loading, rendering, hot reload
│   ├── runtime/            transformers, vLLM, OpenAI-compatible backends
│   ├── storage/            Local and S3-compatible object-store boundary
│   ├── transport/          In-memory and Redis Streams event-bus boundary
│   ├── triage/             YOLOE detector, tracking, event-state queue
│   └── verifier/           VLM prompts and structured-output parsing
├── Dockerfile.backend
├── Dockerfile.console      Static console image using console/nginx.conf
├── Dockerfile.deepstream
└── docker-compose.yaml
```

## Documentation

- [Local console workflow](docs/local-console.md)
- [kind / microk8s validation](docs/kind-validation.md)
- [Helm chart notes](docs/helm.md)
- [Documentation index](docs/README.md)

## Scope And Non-Goals

VRS is intended for education, learning, and research-oriented prototyping. It does not guarantee commercial accuracy, throughput, reliability, or regulatory fitness.

VRS is inspired by architecture patterns in [NVIDIA Video Search and Summarization (VSS)](https://docs.nvidia.com/vss/latest/) and [VAST Data VSS Blueprint](https://github.com/vast-data/vss-blueprint/): perception first, VLM-based alert verification, and optional higher-level incident reasoning. It is not a commercial VMS, a finished NVIDIA VSS replacement, or a VSS clone.

The project direction is a smaller, hackable, local-GPU-oriented Video Reasoning System that can be evaluated, customized, and embedded into CCTV / VMS / edge-appliance environments while preserving portable contracts across local and Kubernetes deployments.
