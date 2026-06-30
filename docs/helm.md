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
  worker is disabled until a real service entrypoint exists. The DeepStream
  worker uses the DS8 image-baked YOLOE PGIE/label configs and expects the
  deployment to provide `vrs-deepstream-input` and `vrs-deepstream-models`
  PVCs for input media/RTSP config and TensorRT engines.

Workload components:

- API/runtime deployment.
- Metadata adapter worker deployment with `vrs.ai/gpu-role: deepstream`.
  The default and kind profiles use the Python metadata adapter. The production
  profile uses the DS 8.0 native C++ worker image/command and should be pointed
  at deployment-specific DeepStream pipeline/model configuration. In production,
  a `detection-publisher` sidecar tails the worker's `detection.v1` JSONL output
  from a shared `emptyDir` and publishes it to Redis Streams.
- Verifier worker deployment template with `vrs.ai/gpu-role: verifier`, disabled
  by default until `vrs.verifier.worker` exists.
- Redis edge bus.
- Local PVC or SeaweedFS object storage.
- Metrics service and optional ServiceMonitor.

The default metadata adapter worker command points at
`python -m vrs.deepstream.worker`, which converts DeepStream-style metadata
JSON/JSONL into canonical `detection.v1` JSONL for CPU-only smoke tests. The
production profile points at `/opt/vrs/bin/vrs-deepstream-worker`, the native
DS 8.0 C++ worker. Its reference pipeline uses a square `nvstreammux`
(`width=640 height=640 enable-padding=1`) and
`/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt` to avoid the
aspect-ratio distortion documented in the DS8 YOLOE validation note. The
default chart mounts `/data`, creates a local PVC when
`objectStorage.mode: local-pvc`, mounts sample metadata through a ConfigMap, and
writes adapter output under `/data/runs`.

The Redis bridge is intentionally a sidecar rather than C++ worker logic. It
keeps the current worker small while providing a real platform handoff:

```text
vrs-deepstream-worker -> /tmp/vrs/deepstream_detections.jsonl
  -> python -m vrs.deepstream.jsonl_bridge
  -> Redis Streams: vrs.detections
```

When `objectStorage.mode: seaweedfs`, SeaweedFS is exposed through its S3 API
and workloads receive `VRS_OBJECT_STORE_ENDPOINT`,
`VRS_OBJECT_STORE_BUCKET`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`. Application code should use the S3-compatible object
store adapter as the canonical evidence/manifest storage path. Runtime
`object_manifest.v1` documents are written through that configured store; any
container filesystem output is scratch or a local append index only.

For a kind smoke path, see `docs/kind-validation.md`.
