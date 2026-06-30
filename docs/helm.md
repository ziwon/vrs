# Helm Edge Profile Scaffold

The chart lives under `charts/vrs`. It packages the API, console, DeepStream
worker, storage, Redis, and metrics scaffolding needed for single-node smoke
tests and production-profile rendering. It is still a Helm chart, not a custom
Kubernetes operator.

Profiles:

- `values-dev.yaml` disables GPU workers for local chart iteration.
- `values-kind.yaml` enables the Python DeepStream adapter sample path without GPU
  resource requests so a plain kind cluster can validate Kubernetes wiring.
- `values-edge.yaml` targets a single-node edge host with Redis, local PVC
  object storage, one DeepStream worker, verifier worker disabled until a
  service entrypoint exists, and metrics.
- `values-prod.yaml` sketches a production cluster shape with multiple API,
  DeepStream worker replicas plus SeaweedFS and ServiceMonitor enabled. It
  disables sample metadata and injects SeaweedFS S3-compatible endpoint,
  bucket, and credential environment variables into workloads. The verifier
  worker is disabled until a real service entrypoint exists. The DeepStream
  worker uses the DS8 image-baked YOLOE preprocess/PGIE/label configs and
  expects the deployment to provide `vrs-deepstream-input` and
  `vrs-deepstream-models` PVCs for input media/RTSP config and TensorRT engines.

Workload components:

- API/runtime deployment.
- Console deployment and service. The console is the operator-facing static UI
  served by nginx; its chart ConfigMap injects `config.js` and proxies `/api/*`
  to the in-cluster API service.
- DeepStream worker deployment with `vrs.ai/gpu-role: deepstream`.
  The default and kind profiles use the Python metadata adapter. The production
  profile uses the DS 8.0 native C++ worker image/command and should be pointed
  at deployment-specific DeepStream pipeline/model configuration. In production,
  the pipeline writes `detection.v1` JSONL through `vrsmeta`, and a
  `detection-publisher` sidecar tails that file from a shared `emptyDir` and
  publishes it to Redis Streams.
- Verifier worker deployment template with `vrs.ai/gpu-role: verifier`, disabled
  by default until `vrs.verifier.worker` exists.
- Redis edge bus.
- Local PVC or SeaweedFS object storage.
- Metrics service and optional ServiceMonitor.

The default DeepStream worker command points at
`python -m vrs.deepstream.worker`, which converts DeepStream-style metadata
JSON/JSONL into canonical `detection.v1` JSONL for CPU-only smoke tests. The
production profile points at `/opt/vrs/bin/vrs-deepstream-worker`, the native
DS 8.0 C++ worker. Its reference pipeline uses a square `nvstreammux`
(`width=640 height=640 enable-padding=1`), an explicit `nvdspreprocess` stage,
and `nvinfer input-tensor-meta=true` with
`/opt/vrs/share/deepstream/configs/pgie-yoloe-safety-preprocess.txt`. This keeps
the production profile aligned with the raw tensor parity evidence documented in
the DS8 YOLOE validation note. The default chart mounts `/data`, creates a
local PVC when `objectStorage.mode: local-pvc`, mounts sample metadata through a
ConfigMap, and writes worker output under `/data/runs`. API pods receive
`VRS_RUNS_ROOT=/data/runs` so local artifact reads align with the worker output
directory in kind and microk8s-style single-node smoke profiles.

The console image is separate from the API image because it serves static
assets through nginx rather than Python. For local clusters, build and load
`vrs-console:latest` from `Dockerfile.console` alongside the API image.

The Redis bridge is intentionally a sidecar rather than C++ worker logic. It
keeps the current worker small while providing a real platform handoff:

```text
vrs-deepstream-worker --disable-probe
  -> nvdspreprocess ! nvinfer input-tensor-meta=true ! nvtracker
  -> vrsmeta output-path=/tmp/vrs/deepstream_detections.jsonl
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
