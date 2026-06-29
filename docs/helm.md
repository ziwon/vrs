# Helm Edge Profile Scaffold

The initial chart lives under `charts/vrs`. It packages an executable metadata
adapter baseline plus storage/Redis scaffolding; it is not a full Kubernetes
operator or real DeepStream deployment.

Profiles:

- `values-dev.yaml` disables GPU workers for local chart iteration.
- `values-kind.yaml` enables the metadata adapter sample path without GPU
  resource requests so a plain kind cluster can validate Kubernetes wiring.
- `values-edge.yaml` targets a single-node edge host with Redis, local PVC
  object storage, one metadata adapter worker, verifier worker disabled until a
  service entrypoint exists, and metrics.
- `values-prod.yaml` sketches a production cluster shape with multiple API,
  metadata adapter replicas plus SeaweedFS and ServiceMonitor enabled. It
  disables sample metadata and injects SeaweedFS S3-compatible endpoint,
  bucket, and credential environment variables into workloads. The verifier
  worker is disabled until a real service entrypoint exists.

Workload components:

- API/runtime deployment.
- Metadata adapter worker deployment with `vrs.ai/gpu-role: deepstream`.
- Verifier worker deployment template with `vrs.ai/gpu-role: verifier`, disabled
  by default until `vrs.verifier.worker` exists.
- Redis edge bus.
- Local PVC or SeaweedFS object storage.
- Metrics service and optional ServiceMonitor.

The metadata adapter worker command points at
`python -m vrs.deepstream.worker`, which converts DeepStream-style metadata
JSON/JSONL into canonical `detection.v1` JSONL. It does not run DeepStream or
GStreamer. The default chart mounts `/data`, creates a local PVC when
`objectStorage.mode: local-pvc`, mounts sample metadata through a ConfigMap, and
writes adapter output under `/data/runs`.

When `objectStorage.mode: seaweedfs`, SeaweedFS is exposed through its S3 API
and workloads receive `VRS_OBJECT_STORE_ENDPOINT`,
`VRS_OBJECT_STORE_BUCKET`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`. Application code should use the S3-compatible object
store adapter as the canonical evidence/manifest storage path; any container
filesystem output is scratch only.

For a kind smoke path, see `docs/kind-validation.md`.
