# VRS Documentation

This directory is the documentation portal for VRS. The root README explains
the portable runtime model and standard interfaces; this index groups the
detailed guides by purpose.

## Start Here

- [Local console workflow](local-console.md) — Docker Compose, RTSP fixture, API,
  console, optional GPU inference, and troubleshooting.
- [kind / microk8s validation](kind-validation.md) — single-node Kubernetes
  smoke test for Helm wiring, API, console, Redis, PVCs, and sample metadata.
- [k3s GPU validation](k3s-gpu.md) — single-node GPU Kubernetes validation for
  native DeepStream worker scheduling and service wiring.
- [Helm chart notes](helm.md) — chart profiles, workloads, storage modes, and
  DeepStream production-profile shape.

## Architecture

- [DeepStream architecture](architecture/deepstream.md) — architecture-layer
  proposal for the DeepStream-based runtime.
- [DeepStream plugin runtime](architecture/deepstream-plugin-runtime.md) —
  `vrsmeta`, metadata export, and plugin/runtime milestone plan.
- [Contracts](contracts.md) — event, evidence, transport, and storage contract
  boundaries.
- [VSS + SAM3 blueprint](03-vss-sam3-blueprint.md) — application-layer blueprint
  and longer-term product direction.

## Operations

- [Operations notes](operations.md) — metrics, audit signing, GPU smoke tests,
  verifier setup, and runbook-style commands.
- [Runtime validation matrix](runtime-matrix.md) — validated, unvalidated, and
  planned runtime/GPU profiles.
- [Policy model](../configs/policies/README.md) — watch-policy schema and
  policy-driven event definitions.

## DeepStream

- [DeepStream worker](deepstream-worker.md) — DS8 C++ worker build/run notes and
  contract boundary.
- [DeepStream YOLOE validation](benchmarks/deepstream-ds8-yoloe-validation-2026-06-30.md)
  — current parser, preprocessing, parity evidence, and remaining blockers.

## Evaluation And Data

- [D-Fire dataset notes](dfire-dataset.md)
- [Benchmark template](benchmarks/template.md)
- [DFire YOLOE model refresh](benchmarks/dfire-yoloe26-model-refresh-2026-06-21.md)
- [RTX 5080 vLLM / Cosmos validation](benchmarks/rtx5080-vllm-cosmos-validation-2026-06-22.md)
- [Jetson Orin verifier plans and measurements](benchmarks/jetson-orin-qwen25vl7b-vlm-plan-2026-05-14.md)

## Planning And Review

- [System review](01-system-review.md)
- [Roadmap](02-roadmap.md)
- [Archive](archive/)
