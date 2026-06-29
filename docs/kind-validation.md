# kind Validation For Helm Wiring And Metadata Adapter

This path validates Kubernetes wiring and the dependency-free DeepStream
metadata adapter. It does not validate real DeepStream, GStreamer, NVIDIA GPU
device plugins, RTSP ingest, `nvstreammux`, `nvinfer`, or `nvtracker`.

Observed local data under `/data/vrs` on this workstation:

- `/data/vrs/D-Fire/test/images/*.jpg`
- `/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi`
- `/data/vrs/kaggle-fire-detection/mivia_smoke/mivia_smoke/SmokeAVI/*.avi`
- `/data/vrs/verifier-eval/*.json`
- UCF/UCA annotation archives under `/data/vrs/uca`

The chart ships a tiny sample metadata ConfigMap by default. Use that first so
the metadata adapter can run without copying large datasets into the cluster.

## Create A Cluster

```bash
kind create cluster --name vrs-dev
```

## Build And Load The Image

```bash
docker build -t vrs:latest -f Dockerfile.backend .
kind load docker-image vrs:latest --name vrs-dev
```

## Render And Inspect The Chart

```bash
helm lint charts/vrs
helm template test charts/vrs -f charts/vrs/values-dev.yaml
helm template test charts/vrs -f charts/vrs/values-kind.yaml
```

`values-kind.yaml` enables the metadata adapter worker with a sample metadata
ConfigMap, disables the placeholder verifier worker, and removes GPU resource
requests so a normal kind cluster can schedule the pod. `values-dev.yaml`
disables GPU workers entirely for chart iteration.

## Install The Metadata Adapter Baseline

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
kubectl logs deploy/vrs-vrs-metadata-adapter
kubectl exec deploy/vrs-vrs-metadata-adapter -- \
  sh -lc 'ls -R /data && cat /data/runs/deepstream_detections.jsonl'
```

Expected result: `/data/runs/deepstream_detections.jsonl` contains
`detection.v1` records converted from the sample metadata. This validates chart
commands, `/data` mounts, ConfigMap sample metadata, and the Python metadata
adapter entrypoint.

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

To test the metadata adapter with custom DeepStream-like metadata:

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

## Real GPU/k3s Validation Later

Real DeepStream validation needs an NVIDIA runtime-enabled node with DeepStream,
GStreamer, TensorRT engines, NVIDIA device plugin, and representative RTSP or
video sources. The target path remains:

```text
RTSP/video -> DeepStream/GStreamer -> nvinfer/nvtracker -> metadata export
  -> detection.v1 -> VRS event-state/policy -> candidate_alert.v1
  -> verifier -> verified_alert.v1
```

## Cleanup

```bash
helm uninstall vrs
kind delete cluster --name vrs-dev
```
