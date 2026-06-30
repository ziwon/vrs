# DeepStream Plugin Runtime Roadmap

This document turns the current DeepStream worker into a concrete plugin
roadmap. It is based on the DeepStream custom GStreamer plugin pattern
(`GstBaseTransform`, in-place metadata processing, and DeepStream metadata
access) and on the validation work already captured in
`docs/benchmarks/deepstream-ds8-yoloe-validation-2026-06-30.md`.

The goal is not to move all VRS logic into DeepStream. The goal is to keep the
video data plane inside DeepStream while publishing contract-compatible VRS
metadata to the rest of the platform.

## Current State

Implemented today:

- `native/deepstream` builds a DS 8.0 C++ worker.
- The worker runs a real GStreamer/DeepStream pipeline with `gst_parse_launch`.
- It attaches a pad probe to a named element/pad.
- It reads `NvDsBatchMeta`, `NvDsFrameMeta`, and `NvDsObjectMeta`.
- It writes canonical `detection.v1` JSONL.
- It includes a custom YOLOE `nvinfer` parser library.
- It can apply bbox scale/offset transforms before writing source-frame boxes.

What this gives us:

- DeepStream owns decode, mux, inference, tracking, and object metadata.
- VRS avoids pulling full video frames into Python for detection export.
- The worker proves that the contract boundary is feasible.

What is still missing:

- The metadata exporter is a pad probe inside an application, not a reusable
  GStreamer element.
- Output is still file-oriented JSONL by default.
- The verifier/evidence path does not keep `NvBufSurface` objects on GPU.
- Runtime configuration is still mostly a pipeline string plus process args.
- Detector parity for YOLOE `nvinfer` is not production-accepted yet.

## Target Direction

The next production data-plane unit should be a native C++ GStreamer plugin:

```text
gst-vrsmeta
```

It should sit after `nvinfer` and usually after `nvtracker`:

```text
nvmultiurisrcbin or nvv4l2decoder
  -> nvstreammux
  -> nvinfer
  -> nvtracker
  -> vrsmeta
  -> fakesink or downstream display/tee
```

`vrsmeta` should be a `GstBaseTransform` element that operates in-place. It
should not copy frame pixels. Its first responsibility is metadata export:

```text
NvDsBatchMeta / NvDsObjectMeta
  -> detection.v1
  -> local JSONL, Redis Streams, or another EventTransport
```

Frame pixels should stay in NVMM/GPU memory unless an explicit evidence
extraction stage requests a crop or clip.

## Why A C++ Plugin Instead Of A Python Plugin

A Python GStreamer plugin can be useful when a team already has a mature Python
or PyTorch inference stack and wants DeepStream to call into it. That is not the
main VRS production goal.

For VRS, the production goal is:

- DeepStream and TensorRT in the hot path.
- Minimal host-memory transfer.
- Reproducible DS 8 container builds.
- Stable deployment inside the Helm chart.
- Metadata contracts emitted with bounded latency.

Therefore the hot-path metadata plugin should be C++. Python should remain in
the reasoning plane: event-state, policy evaluation, verifier orchestration,
calibration, evaluation, and non-DeepStream portability.

## Zero-Copy Boundary

The practical zero-copy target is:

```text
decode -> mux -> infer -> track -> metadata export
```

This means:

- Use NVIDIA elements such as `nvv4l2decoder`, `nvstreammux`, `nvinfer`, and
  `nvtracker`.
- Keep buffers in NVMM/GPU memory through the detector/tracker path.
- Read DeepStream metadata directly from the `GstBuffer`.
- Publish metadata and object/evidence references, not raw video frames.
- Avoid production hot-path `appsink`, OpenCV frame downloads, CPU
  `videoconvert`, and Python frame loops.

This does not mean the entire VRS platform is zero-copy. The API, event-state,
policy, verifier, and storage layers operate on metadata, references, thumbnails,
or clips. Any verifier crop/clip extraction should be a separate GPU-aware
evidence stage, not hidden inside the metadata exporter.

## Plugin Responsibilities

### `gst-vrsmeta`

First production plugin. Required.

Responsibilities:

- Read `NvDsFrameMeta` and `NvDsObjectMeta`.
- Map class ids to labels.
- Preserve `track_id` when `nvtracker` is present.
- Apply configured bbox scale/offset or deletterbox transforms.
- Emit `detection.v1`.
- Support JSONL output first.
- Add Redis Streams or `EventTransport` output next.
- Expose metrics counters for frames, detections, publish failures, and dropped
  records.

Non-goals:

- VLM verification.
- Event-state promotion.
- Policy threshold decisions beyond nvinfer/parser thresholds.
- CPU frame extraction.
- Object crop generation.

Expected element properties:

```text
stream-id
source-id
detector-id
labels
output-mode        jsonl | redis
output-path
redis-url
redis-stream
bbox-scale-x
bbox-scale-y
bbox-offset-x
bbox-offset-y
debug
```

Example pipeline:

```text
filesrc location=/data/vrs/input.mp4
  ! qtdemux
  ! h264parse
  ! nvv4l2decoder
  ! m.sink_0 nvstreammux name=m batch-size=1 width=640 height=640 enable-padding=1
  ! nvinfer config-file-path=/etc/vrs/deepstream/pgie-yoloe-safety.txt
  ! nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
  ! vrsmeta stream-id=cam-01 detector-id=ds8-yoloe output-mode=jsonl output-path=/runs/detections.jsonl bbox-offset-y=-80 bbox-scale-x=0.5 bbox-scale-y=0.5
  ! fakesink sync=false
```

### `gst-vrsevidence`

Second plugin. Optional and later.

Responsibilities:

- Extract keyframes, object crops, or short clips from selected detections.
- Use `NvBufSurface` and `NvBufSurfTransform` or equivalent GPU-aware paths.
- Publish `evidence_ref.v1` records that point to object storage.
- Avoid blocking the detector path. Use bounded queues or async handoff.

Non-goals:

- Detector inference.
- VLM execution.
- Policy semantics.

This plugin should not be started until `gst-vrsmeta` is stable.

## Configuration Boundary

The worker process should eventually become a pipeline runner and lifecycle
manager. It should not own metadata conversion logic long term.

Current:

```text
vrs-deepstream-worker
  -> runs pipeline
  -> attaches pad probe
  -> converts metadata
  -> writes JSONL
```

Target:

```text
vrs-deepstream-worker
  -> runs pipeline
  -> handles signals, bus errors, lifecycle, health

gst-vrsmeta
  -> converts metadata
  -> publishes detection.v1
```

The Helm chart should configure the pipeline with a `vrsmeta` element instead of
depending on worker CLI output flags.

## Milestones

### M0 - Documentation And Acceptance Criteria

Status: this document.

Deliverables:

- Plugin runtime roadmap.
- Clear zero-copy boundary.
- Milestone list with acceptance tests.

Acceptance:

- README and architecture docs point to the plugin runtime plan.
- Current worker remains documented as the bootstrap implementation, not the
  final plugin boundary.

### M1 - Extract Metadata Mapping Core

Status: implemented as `vrs_deepstream_metadata_core`.

Goal: make metadata conversion reusable before creating a plugin.

Deliverables:

- Move detection serialization, label lookup, stable id generation, bbox
  transform, and JSONL writing out of `main.cpp` into small C++ helpers.
- Keep the existing pad-probe worker behavior unchanged.
- Add local tests where possible for pure helper logic that does not require
  DeepStream headers.

Implemented:

- `native/deepstream/src/metadata_core.hpp`
- `native/deepstream/src/metadata_core.cpp`
- `native/deepstream/tests/test_metadata_core.cpp`
- CMake target `vrs_deepstream_metadata_core`
- CMake executable `vrs-deepstream-metadata-core-test`

Acceptance:

- Existing DS 8 worker smoke still builds in `Dockerfile.deepstream`.
- `vrs-deepstream-worker` still emits the same `detection.v1` fields.
- CPU-only tests still pass.

### M2 - Add `gst-vrsmeta` Plugin Skeleton

Status: implemented.

Goal: create a loadable GStreamer element.

Deliverables:

- Add a CMake target for `libgstvrsmeta.so`.
- Register a `vrsmeta` element.
- Implement pass-through `GstBaseTransform` behavior.
- Install the plugin into the DS 8 image.

Implemented:

- `native/deepstream/src/gst_vrsmeta.cpp`
- CMake target `gstvrsmeta`, producing `libgstvrsmeta.so`
- Plugin install path `/opt/vrs/lib/gstreamer-1.0`
- Runtime `GST_PLUGIN_PATH=/opt/vrs/lib/gstreamer-1.0:${GST_PLUGIN_PATH}`

Acceptance:

- `gst-inspect-1.0 vrsmeta` works inside `vrs-deepstream:ds8`.
- A simple pipeline can pass buffers through `vrsmeta` to `fakesink`.
- Existing worker target still builds.

### M3 - Move Detection Export Into `vrsmeta`

Status: initial JSONL export implemented.

Goal: replace pad-probe export with plugin-owned export.

Deliverables:

- `vrsmeta` reads `NvDsBatchMeta` from each `GstBuffer`.
- It emits `detection.v1` JSONL.
- It supports labels, stream id, detector id, bbox scale, and bbox offset
  properties.
- The worker no longer needs to attach a metadata pad probe for this path.

Implemented:

- `vrsmeta` reads `NvDsBatchMeta`, `NvDsFrameMeta`, and `NvDsObjectMeta` from
  each `GstBuffer`.
- It uses `vrs_deepstream_metadata_core` to serialize `detection.v1`.
- It supports `stream-id`, `source-id`, `detector-id`, `labels`,
  `output-mode=jsonl`, `output-path`, `append`, bbox scale/offset, and `debug`
  properties.
- The existing worker pad probe remains available as the fallback/bootstrap path
  until Helm switches production pipelines to include `vrsmeta`.

Acceptance:

- The sample DeepStream PGIE smoke produces detections through `vrsmeta`.
- MIVIA and 120.mp4 validation pipelines can run with:

```text
... ! nvinfer ! nvtracker ! vrsmeta ... ! fakesink
```

- Output matches the current worker JSONL shape.

### M4 - Preserve Source-Frame Coordinates

Goal: make square muxer and source coordinate handling explicit and reliable.

Deliverables:

- Document and test `bbox-offset` and `bbox-scale` properties.
- Add deletterbox helper logic for common muxer/source combinations.
- Keep manual transform properties as an override.

Acceptance:

- 120.mp4 square-muxer parity still matches 18/18 smoke detections.
- MIVIA falldown strict parity still matches with IoU around 0.94.
- `detection.v1` boxes are source-frame boxes, not padded muxer-space boxes.

### M5 - Add EventTransport Output

Goal: move beyond JSONL as the only production output.

Deliverables:

- Add Redis Streams output mode to `vrsmeta`, or a small sidecar bridge if that
  is safer.
- Define stream names and idempotency behavior.
- Keep JSONL as debug and fallback output.

Acceptance:

- `vrsmeta output-mode=redis` publishes `detection.v1` records.
- Python event-state can consume the same records through the existing transport
  boundary.
- Publish failures are counted and bounded, not silently blocking the pipeline.

Current interim implementation: the production Helm profile uses a
`detection-publisher` sidecar running `python -m vrs.deepstream.jsonl_bridge`.
This tails the C++ worker JSONL output from a shared `emptyDir` and publishes
records to Redis Streams. `vrsmeta output-mode=redis` remains the target plugin
implementation.

### M6 - Helm Integration

Goal: make the chart use the plugin path by default for production.

Deliverables:

- Update `charts/vrs/values-prod.yaml` pipeline to include `vrsmeta`.
- Mount labels and PGIE config consistently.
- Expose output mode, stream id, detector id, bbox transform, and Redis settings
  through values.

Acceptance:

- `helm template` renders a DS 8 worker command with `vrsmeta`.
- kind profile still uses the Python metadata adapter path.
- production profile still requests GPU resources for DeepStream workers.

### M7 - Raw Tensor And Detector Parity Gate

Goal: close the remaining YOLOE DeepStream parity gap.

Deliverables:

- Add a same-frame comparison tool for PyTorch, ONNX Runtime, TensorRT, and
  DeepStream raw outputs.
- Use it to investigate 120.mp4 fire false positives and MIVIA smoke weakness.
- Record a benchmark note before promoting the detector profile.

Acceptance:

- A parity report explains class-score divergence or validates a fix.
- The production PGIE config is not promoted until this report passes agreed
  thresholds.

### M8 - Optional `gst-vrsevidence`

Goal: add GPU-aware evidence extraction after metadata export is stable.

Deliverables:

- Prototype keyframe/object crop extraction from `NvBufSurface`.
- Publish `evidence_ref.v1` to object storage.
- Keep extraction asynchronous or bounded so detection throughput is protected.

Acceptance:

- Evidence generation does not force detector-path CPU frame downloads.
- Object storage manifests reference generated evidence.
- VLM verifier can consume evidence references without needing DeepStream in its
  process.

## Implementation Order

Start with M1. It reduces risk because the current worker remains functional
while conversion logic becomes reusable. Then implement M2 and M3 to introduce
the real GStreamer element. M4 should happen before any production Helm switch,
because bbox coordinate correctness is part of the `detection.v1` contract. M5
and M6 make the plugin useful in the distributed chart. M7 is the detector
accuracy gate. M8 is deliberately last.

Recommended next task:

```text
M1 - Extract Metadata Mapping Core
```

Do not start with `gst-vrsevidence` or custom preprocessing. The current highest
leverage step is to make `detection.v1` metadata export a reusable native
component, then place it behind a real DeepStream plugin boundary.
