# kind Validation For Helm Wiring And DeepStream Worker

This path validates Kubernetes wiring and the dependency-free Python
DeepStream adapter entrypoint inside the Helm DeepStream worker deployment. It
does not validate the native C++ DeepStream 8.0 worker, GStreamer, NVIDIA GPU
device plugins, RTSP ingest, `nvstreammux`, `nvinfer`, or `nvtracker`.

Observed local data under `/data/vrs` on this workstation:

- `/data/vrs/D-Fire/test/images/*.jpg`
- `/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi`
- `/data/vrs/kaggle-fire-detection/mivia_smoke/mivia_smoke/SmokeAVI/*.avi`
- `/data/vrs/verifier-eval/*.json`
- UCF/UCA annotation archives under `/data/vrs/uca`

The chart ships a tiny sample metadata ConfigMap by default. Use that first so
the worker can run without copying large datasets into the cluster.

## Create A Cluster

```bash
kind create cluster --name vrs-dev
```

For microk8s, use the same `values-kind.yaml` smoke profile first. Enable DNS,
storage, and Helm support, then run the same Helm commands through microk8s:

```bash
microk8s enable dns hostpath-storage helm3
alias kubectl='microk8s kubectl'
alias helm='microk8s helm3'
```

Use `values-edge.yaml` only after the single node has an NVIDIA runtime and
device plugin configured; the edge profile keeps GPU scheduling labels and
requests enabled.

## Build And Load The Image

```bash
docker build -t vrs:latest -f Dockerfile.backend .
docker build -t vrs-console:latest -f Dockerfile.console .
kind load docker-image vrs:latest --name vrs-dev
kind load docker-image vrs-console:latest --name vrs-dev
```

For microk8s containerd, import the same local image before installing:

```bash
docker save vrs:latest | microk8s ctr image import -
docker save vrs-console:latest | microk8s ctr image import -
```

## Render And Inspect The Chart

```bash
helm lint charts/vrs
helm template test charts/vrs -f charts/vrs/values-dev.yaml
helm template test charts/vrs -f charts/vrs/values-kind.yaml
```

`values-kind.yaml` enables the DeepStream worker with a sample metadata
ConfigMap, disables the placeholder verifier worker, and removes GPU resource
requests so a normal kind cluster can schedule the pod. `values-dev.yaml`
disables GPU workers entirely for chart iteration.

## Install The Single-Node Smoke Baseline

```bash
helm install vrs charts/vrs \
  -f charts/vrs/values-kind.yaml \
  --set image.repository=vrs \
  --set image.tag=latest \
  --set image.pullPolicy=IfNotPresent
```

Check pods and logs:

```bash
kubectl get pods
kubectl logs deploy/vrs-vrs-deepstream-worker
kubectl exec deploy/vrs-vrs-deepstream-worker -- \
  sh -lc 'ls -R /data && cat /data/runs/deepstream_detections.jsonl'
```

Expected result: `/data/runs/deepstream_detections.jsonl` contains
`detection.v1` records converted from the sample metadata. This validates chart
commands, `/data` mounts, ConfigMap sample metadata, and the Python metadata
adapter entrypoint used by the DeepStream worker deployment.

Check the API deployment against the same mounted runtime path:

```bash
kubectl port-forward svc/vrs-vrs-api 8000:8000
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/artifacts
```

The API pod receives `VRS_RUNS_ROOT=/data/runs`, so artifact reads point at the
same output directory that the worker writes in this local-PVC smoke profile.

Check the console through its service. The console nginx config proxies `/api/*`
to the in-cluster API service, so this validates the browser-facing route:

```bash
kubectl port-forward svc/vrs-vrs-console 5173:80
curl -fsS http://127.0.0.1:5173/api/health
```

Open <http://127.0.0.1:5173> to inspect the VRS Console.

## Use Local `/data/vrs` Assets For Non-GPU Checks

The workstation has useful test data under `/data/vrs`, but the kind smoke path
does not mount those large host datasets into the cluster. Use them outside
kind first to validate detector export/parity shapes:

```bash
uv run scripts/export_python_detections.py \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/parity/python_detections.jsonl \
  /data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi
```

To test the Python adapter entrypoint with custom DeepStream-like metadata:

```bash
mkdir -p runs/parity
cat > runs/parity/deepstream_metadata.jsonl <<'JSONL'
{"stream_id":"kind-cam-01","clip_id":"fire1","frame_index":1,"pts_s":0.25,"class_name":"fire","confidence":0.91,"bbox_xyxy":[10,20,80,120],"detector_id":"sample-deepstream-export"}
JSONL

uv run python -m vrs.deepstream.worker \
  --input runs/parity/deepstream_metadata.jsonl \
  --out runs/parity/deepstream_detections.jsonl
```

Then compare parity files:

```bash
uv run scripts/compare_detector_parity.py \
  --python-detections runs/parity/python_detections.jsonl \
  --candidate-detections runs/parity/deepstream_detections.jsonl \
  --out runs/parity/detector_parity.json
```

This produces a parity report shape. It does not prove model parity unless both
files were generated from the same clips and frame indexes.

## Native DeepStream 8.0 Worker Validation Later

The C++ worker lives under `native/deepstream` and is packaged by
`Dockerfile.deepstream`. Real validation needs an NVIDIA runtime-enabled node
with DeepStream 8.0, GStreamer, TensorRT engines, NVIDIA device plugin, and
representative RTSP or video sources. The target path remains:

```text
RTSP/video -> DeepStream/GStreamer -> nvinfer/nvtracker -> metadata export
  -> detection.v1 -> VRS event-state/policy -> candidate_alert.v1
  -> verifier -> verified_alert.v1
```

See `docs/deepstream-worker.md` for the DS 8.0 build and smoke-test commands.

## Cleanup

```bash
helm uninstall vrs
kind delete cluster --name vrs-dev
```
