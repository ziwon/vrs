# DeepStream 8 C++ Worker

VRS now includes a native C++ DeepStream worker under `native/deepstream`.
It is the first real DeepStream/GStreamer data-plane implementation: it runs a
pipeline, attaches a pad probe, reads DeepStream object metadata, and emits
canonical `detection.v1` JSONL.

The existing Python `vrs.deepstream.worker` remains useful for kind and contract
smoke tests because it does not require NVIDIA runtime libraries.

## Target Runtime

- DeepStream 8.0
- NVIDIA DeepStream container: `nvcr.io/nvidia/deepstream:8.0-triton-multiarch`
- Host OS/driver aligned with the DeepStream 8.0 support matrix
- NVIDIA container runtime with video driver capabilities

## Build

```bash
docker build -t vrs-deepstream:ds8 -f Dockerfile.deepstream .
```

## Run A File Smoke

Mount input video, model config, TensorRT engine, labels, and output directory:

```bash
mkdir -p runs/deepstream

docker run --rm --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v "$PWD/runs:/runs" \
  -v /data/vrs:/data/vrs:ro \
  -v "$PWD/configs/deepstream:/etc/vrs/deepstream:ro" \
  vrs-deepstream:ds8 \
  --pipeline "$(cat configs/deepstream/ds8-file-example.pipeline)" \
  --probe-element sink \
  --probe-pad sink \
  --stream-id file-fire1 \
  --detector-id ds8-nvinfer \
  --labels /etc/vrs/deepstream/labels.txt \
  --out /runs/deepstream/detections.jsonl
```

The checked-in pipeline is a template. It expects `/etc/vrs/deepstream/pgie.txt`
and the referenced engine/model files to exist. Update the demux/parser section
to match the actual codec of the mounted test clip.

## Contract Boundary

The C++ worker emits `detection.v1` only. Promotion into
`candidate_alert.v1` and `verified_alert.v1` remains in VRS policy/event-state
and verifier workers.

Each output record includes:

- `schema_version: detection.v1`
- stable `detection_id` and `idempotency_key`
- `stream_id`, optional `source_id`, `frame_index`, `pts_s`
- `class_name`, `score`, `bbox_xyxy`, `track_id`
- `source_runtime: deepstream`
- `detector_id`

## Validation

After producing DeepStream JSONL, compare against the existing Python detector
export path:

```bash
uv run scripts/export_python_detections.py \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/parity/python_detections.jsonl \
  /data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi

uv run scripts/compare_detector_parity.py \
  --python-detections runs/parity/python_detections.jsonl \
  --candidate-detections runs/deepstream/detections.jsonl \
  --out runs/parity/deepstream_parity.json
```
