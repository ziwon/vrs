# VRS DeepStream 8 C++ Worker

This directory contains the native DeepStream data-plane worker. It runs a real
GStreamer/DeepStream pipeline, attaches a pad probe to a named element, reads
`NvDsBatchMeta` / `NvDsFrameMeta` / `NvDsObjectMeta`, and emits canonical
`detection.v1` JSONL.

The worker intentionally does not promote detections to candidate or verified
alerts. That remains VRS event-state, policy, and verifier worker ownership.

## Build In DeepStream 8.0

Build inside an NVIDIA DeepStream 8.0 devel container:

```bash
cmake -S native/deepstream -B build/deepstream \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream
cmake --build build/deepstream -j
```

Local repository CI does not compile this target because DeepStream headers and
libraries are not present outside the NVIDIA container.

## Smoke Run

```bash
build/deepstream/vrs-deepstream-worker \
  --pipeline "$(cat native/deepstream/configs/file-source-example.pipeline)" \
  --probe-element sink \
  --probe-pad sink \
  --stream-id cam-01 \
  --detector-id ds8-nvinfer-yoloe \
  --labels /models/labels.txt \
  --out runs/deepstream/detections.jsonl
```

The example pipeline is a template. Mount a real video, PGIE config, TensorRT
engine, labels file, and tracker config before using it for validation.
