# DeepStream 8 C++ Worker

VRS now includes a native C++ DeepStream worker under `native/deepstream`.
It is the first real DeepStream/GStreamer data-plane implementation: it runs a
pipeline, attaches a pad probe, reads DeepStream object metadata, and emits
canonical `detection.v1` JSONL.

The next production step is to move the metadata export logic from an
application pad probe into a reusable GStreamer element. See
`docs/architecture/deepstream-plugin-runtime.md` for the `gst-vrsmeta`
milestones and zero-copy boundary.

The first `vrsmeta` plugin is now built into the DS8 image. It is installed under
`/opt/vrs/lib/gstreamer-1.0` and exposed through `GST_PLUGIN_PATH`. It can export
DeepStream object metadata to `detection.v1` JSONL when `output-path` is set. The
worker pad probe remains available as the fallback/bootstrap path until Helm
switches production pipelines to include `vrsmeta`.

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
  --labels /opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt \
  --out /runs/deepstream/detections.jsonl
```

The checked-in pipeline is a template. It uses the image-baked
`/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt` and expects the
referenced TensorRT engine under `/models` to exist. Update the source,
demux/parser section, and bbox transform options to match the actual clip or RTSP
source.

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

Check the plugin skeleton inside the DeepStream image:

```bash
docker run --rm --entrypoint gst-inspect-1.0 vrs-deepstream:ds8 vrsmeta
```

Pass-through smoke:

```bash
docker run --rm --entrypoint gst-launch-1.0 vrs-deepstream:ds8 \
  videotestsrc num-buffers=1 ! vrsmeta ! fakesink
```

Plugin-owned metadata export path:

```text
... ! nvinfer ! nvtracker \
  ! vrsmeta stream-id=cam-01 detector-id=ds8-yoloe \
      labels=/opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt \
      output-mode=jsonl output-path=/runs/deepstream/detections.jsonl \
  ! fakesink sync=false
```

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

The first DS 8 YOLOE validation is recorded in
`docs/benchmarks/deepstream-ds8-yoloe-validation-2026-06-30.md`. The worker and
custom parser load correctly, but YOLOE TensorRT parity is not accepted yet:
the MIVIA Python baseline emits five detections while the DeepStream `nvinfer`
path emits zero at production thresholds. Treat the checked-in YOLOE PGIE config
as a validation target, not a production detector profile, until that parity gap
is closed.
