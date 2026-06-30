# DeepStream 8 YOLOE Validation

Date: 2026-06-30

Host/runtime:

- NVIDIA DeepStream container: `nvcr.io/nvidia/deepstream:8.0-triton-multiarch`
- Worker image: `vrs-deepstream:ds8`
- TensorRT engine: `runs/engines/yoloe-11s-safety-rx5080-ds8-trtexec.engine`
- Test clip: `/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi`
- Policy: `configs/policies/safety.yaml`

## What Passed

The native C++ worker builds inside the DeepStream 8 container and installs:

- `/opt/vrs/bin/vrs-deepstream-worker`
- `/opt/vrs/lib/libnvdsinfer_custom_yoloe.so`

The worker can run GStreamer/DeepStream pipelines, attach a pad probe, read
`NvDsBatchMeta`, `NvDsFrameMeta`, and `NvDsObjectMeta`, and write canonical
`detection.v1` JSONL.

The DeepStream sample PGIE smoke on
`/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264` produced
28,312 detections across `bicycle`, `car`, and `person`, with non-null tracker
ids. This validates the worker metadata path independent of YOLOE.

The MIVIA clip decode path also completed through DeepStream:

```bash
filesrc location=/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi \
  ! avidemux ! decodebin ! videoconvert ! nvvideoconvert \
  ! video/x-raw(memory:NVMM),format=NV12 \
  ! m.sink_0 nvstreammux name=m batch-size=1 width=320 height=240 live-source=0 \
  ! fakesink name=sink sync=false
```

## TensorRT Engine Build

Ultralytics `model.export(format="engine")` did not complete in the local
Python environment because the installed TensorRT bindings do not match the
Ultralytics exporter API:

```text
AttributeError: ... NetworkDefinitionCreati has no attribute EXPLICIT_BATCH
```

The intermediate ONNX export succeeded, then a DeepStream 8 container build
used `trtexec` to create the TensorRT engine:

```bash
docker run --rm --gpus all -v "$PWD:/workspace/vrs" \
  --entrypoint trtexec vrs-deepstream:ds8 \
  --onnx=/workspace/vrs/yoloe-11s-seg.onnx \
  --saveEngine=/workspace/vrs/runs/engines/yoloe-11s-safety-rx5080-ds8-trtexec.engine \
  --fp16 --skipInference
```

The engine loaded under `trtexec` in the same DeepStream 8 container. Random
input smoke measured about 792 qps, mean latency about 1.684 ms, and mean GPU
compute about 1.258 ms. Treat this as engine-load evidence only, not detector
accuracy evidence.

## Python Baseline

The Python YOLOE baseline on the same MIVIA clip produced five canonical
`detection.v1` records:

| Frame | PTS | Class | Score |
| ---: | ---: | --- | ---: |
| 92 | 24.5333 | fire | 0.4092 |
| 147 | 39.2000 | fire | 0.3416 |
| 168 | 44.8000 | falldown | 0.4187 |
| 168 | 44.8000 | smoke | 0.2612 |
| 175 | 46.6667 | fire | 0.3025 |

Command:

```bash
uv run scripts/export_python_detections.py \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/parity/python_mivia_fire1.jsonl \
  /data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi
```

## DeepStream YOLOE Result

The custom parser is loaded and receives the expected YOLOE tensors:

```text
NvDsInferParseCustomYoloE: layers=2 classes=12
  layer[0] name=output0 dims=48x8400
  layer[1] name=output1 dims=32x160x160
NvDsInferParseCustomYoloE: selected=output0 channels=48 anchors=8400 global_best_score=0.0460353 pre_nms_candidates=0
```

With the production threshold in `configs/deepstream/pgie-yoloe-safety.txt`,
DeepStream produced zero MIVIA detections:

```text
0 runs/deepstream-yoloe-mivia-latest/detections.jsonl
```

The detector parity report therefore fails acceptance:

```text
written runs/parity/deepstream_yoloe_mivia_latest_parity.json: matched=0 unmatched_python=5 unmatched_candidate=0
```

Lowering only `pre-cluster-threshold` to `0.01` in a temporary run config
produced 2,039 low-confidence detections. The top score was 0.215351 and many
early detections were noisy full-width top-of-frame boxes. This proves the
parser and metadata path are active, but it does not satisfy detector parity.

## Current Assessment

The DS 8 C++ worker is a real DeepStream/GStreamer data-plane worker now. The
YOLOE TensorRT detector path is not production-ready yet.

The blocking gap is parity between the Python YOLOE baseline and the DeepStream
`nvinfer` path. The current evidence makes preprocessing parity the leading
cause. Parser/export issues are less likely than they were after the first
MIVIA failure, but they are not fully ruled out until a raw tensor comparison
tool compares PyTorch, ONNX Runtime, TensorRT, and DeepStream on the same decoded
frame.

Evidence that narrows the problem:

- The parser debug shows `output0 dims=48x8400`, i.e. `48 = 4 (bbox) + 12
  classes + 32 mask coeffs`. That is strong evidence that the exported head has
  the expected safety-policy class count. It does not by itself prove every
  prompt channel is semantically aligned.
- The custom parser produces plausible boxes and high-IoU matches on the 120.mp4
  smoke case and the MIVIA falldown case. This makes the basic
  `[x, y, w, h, class scores, mask coefficients]` decoding likely correct.
- The original first-frame `global_best_score=0.046` was not sufficient evidence
  of a bad engine: parser debug prints once, and the first frame may not contain
  the target event.

The concrete preprocessing risks are:

1. **`nvstreammux` aspect-ratio distortion.** A non-square or non-source-aspect
   muxer stretches sources before `nvinfer`. The original 1280x720 reference
   pipeline distorted 4:3 MIVIA input.
2. **Letterbox coordinate handling.** A square `nvstreammux width=640
   height=640 enable-padding=1` preserves detection geometry, but DeepStream
   object metadata remains in padded muxer coordinates. The worker must
   deletterbox before writing `detection.v1` source-frame boxes.
3. **Padding color and resampling.** DeepStream padding is black while
   Ultralytics pads with 114-gray. This appears secondary on the tested clips,
   but it remains a possible contributor to residual class-score divergence.

TensorRT engine serialization is still runtime-version-specific (the DS 8 engine
would not load under the local Python TRT runtime), which is why per-frame
comparison must happen inside one consistent container, not a separate parity
cause.

Next validation should be done inside one consistent container/runtime:

1. Export ONNX and TensorRT from a DeepStream 8-compatible Python environment
   after `model.set_classes(prompts, model.get_text_pe(prompts))`.
2. Compare PyTorch, ONNX Runtime, TensorRT, and DeepStream outputs for the same
   decoded frame before running full-video parity.
3. Only promote `configs/deepstream/pgie-yoloe-safety.txt` after per-frame
   class, score, and bbox parity is within an agreed tolerance.

## Reproducible Recheck Artifacts

The recheck artifacts are isolated under:

```text
runs/parity/ds8-yoloe-recheck-20260630/
```

Two helper scripts were added so the experiment is reproducible instead of
manual:

- `scripts/export_yoloe_raw_detections.py` exports unfiltered YOLOE prompt
  detections without policy `min_score` filtering.
- `scripts/compare_aligned_detector_parity.py` compares Python and DeepStream
  outputs by event name, `pts_s` tolerance, IoU, and explicit DeepStream bbox
  coordinate transforms.

The native worker also now accepts explicit bbox transforms:

```text
--bbox-offset-x/--bbox-offset-y
--bbox-scale-x/--bbox-scale-y
```

These are required when a padded square muxer is used and the output contract
must be source-frame coordinates.

## Control Experiment: 16:9 Clip, Low Threshold

To test the preprocessing hypothesis, we re-ran on a clip whose aspect ratio
matches the muxer so the `nvstreammux` stretch is removed as a variable:

- Clip: `/data/vrs/fire_dataset/extracted_sample/fire/131.mp4`, **640x360 (16:9)**,
  171 frames. With the muxer at `1280x720` this is a clean 2x upscale, **no
  aspect-ratio distortion**. The `nvinfer` 16:9 -> 640x640 letterbox (black pad)
  vs Ultralytics 114-gray pad difference is the only major preprocessing delta
  left.

Both pipelines were run at a low threshold to compare raw score distributions
(this clip's zero-shot signal is below policy thresholds, so it is a
raw-distribution probe, not an end-to-end parity clip):

| Source | global max | smoke cloud | fallen person | fire | person lying |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python raw (`conf=0.001`) | 0.122 | 0.122 | 0.114 | 0.106 | 0.091 |
| DeepStream (`pre-cluster-threshold=0.001`) | 0.102 | 0.094 | 0.102 | 0.045 | 0.055 |

Recheck files:

- `python_131_raw.jsonl`: 4,166 detections.
- `deepstream_131_low_1280.jsonl`: 7,539 detections.

DeepStream command (custom parser debug on):

```bash
docker run --rm --gpus all -e VRS_YOLOE_PARSER_DEBUG=1 \
  -v "$PWD/runs/engines:/models:ro" \
  -v "$PWD/configs/deepstream:/etc/vrs/deepstream:ro" \
  -v "/data/vrs/fire_dataset/extracted_sample/fire:/clips:ro" \
  -v "$PWD/runs/parity:/out" \
  vrs-deepstream:ds8 \
  --pipeline "filesrc location=/clips/131.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
    ! m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 live-source=0 \
    ! nvinfer config-file-path=/etc/vrs/deepstream/pgie-yoloe-low.txt \
    ! fakesink name=sink sync=false" \
  --out /out/deepstream_131_low.jsonl --stream-id 131 --source-id 131 \
  --detector-id deepstream-yoloe-low \
  --labels /etc/vrs/deepstream/yoloe-safety-labels.txt
```

Findings:

- On a same-aspect clip the two pipelines' raw score magnitudes **converge**
  (global max 0.10 vs 0.12), and the DeepStream top detections are now plausibly
  localized boxes spread across the frame rather than the MIVIA "full-width
  top-of-frame" band artifacts.
- This supports the hypothesis that `nvstreammux` aspect handling was a major
  driver of the original MIVIA failure. It does not prove it is the only cause.
- 131.mp4 cannot serve as an end-to-end parity clip: zero-shot YOLOE-11s raw
  confidence peaks at ~0.12, below the policy thresholds (fire 0.30, smoke 0.25),
  so both pipelines emit zero production detections. A proper end-to-end parity
  clip must be 16:9 (to keep the stretch removed) **and** produce above-threshold
  Python detections.

Action items in priority order:

1. Remove the muxer aspect-ratio distortion: preserve aspect at `nvstreammux`
   (new-streammux config) or match the muxer resolution to the source aspect.
2. Align letterbox padding (114-gray vs black) between Ultralytics and `nvinfer`.
3. Re-run end-to-end parity on a strong 16:9 fire clip and only then promote
   `configs/deepstream/pgie-yoloe-safety.txt` to production thresholds.

## End-to-End Parity: 16:9 Clip With Above-Threshold Signal (120.mp4)

`131.mp4` had no above-threshold signal, so we re-tested on a 16:9 clip that
does fire at production thresholds:

- Clip: `/data/vrs/fire_dataset/extracted_sample/fire/120.mp4`, **640x360 (16:9)**,
  132 frames. Same 2x clean upscale to the `1280x720` muxer (no stretch).
- Python raw "billowing smoke" peaks at 0.626, above the smoke threshold (0.25).

Production-threshold runs (`pgie-yoloe-safety.txt`, `pre-cluster-threshold=0.25`):

- Python baseline (`configs/tiny.yaml`, `target_fps=4`): **18 smoke detections**,
  conf 0.26-0.55, boxes ~`[240,70,410,210]` in 640x360.
- DeepStream (all frames, no subsampling): 190 detections — billowing smoke 132
  (max 0.725), fire 45 (max 0.65), smoke cloud 13 (max 0.486).
- DeepStream square muxer with source-coordinate bbox transform:
  184 detections — billowing smoke 132 (max 0.726), fire 39 (max 0.619), smoke
  cloud 13 (max 0.482).

The aligned parity command maps raw prompt labels to policy events, matches on
`pts_s`, and applies the needed coordinate transform:

```bash
uv run scripts/compare_aligned_detector_parity.py \
  --python-detections runs/parity/ds8-yoloe-recheck-20260630/python_120_policy.jsonl \
  --candidate-detections runs/parity/ds8-yoloe-recheck-20260630/deepstream_120_prod_square_sourcecoords.jsonl \
  --time-tolerance-s 0.06 \
  --iou-threshold 0.5 \
  --out runs/parity/ds8-yoloe-recheck-20260630/parity_120_square_sourcecoords.json
```

Aligned parity result:

- **Smoke parity is essentially exact: 18/18 Python smoke detections matched,
  mean IoU 0.943 (min 0.853)**. DeepStream confidence is higher by about 0.204
  on average.

Remaining per-class divergence (fire):

- DeepStream still emits fire detections on 120.mp4 after the square muxer fix:
  39 fire detections, max 0.619. Python raw YOLOE scores fire at only 0.061.
- This means the square/aspect fix resolves geometry and smoke localization, but
  it does **not** eliminate all class-score divergence. Fire false positives
  remain an open production blocker.

## Aspect-Fix Re-run on MIVIA (4:3)

This directly tests the fix on the clip that originally failed. MIVIA
`fire1.avi` is 320x240 (4:3, FMP4, 15 fps). The fix sets the muxer to a 4:3
resolution (`width=640 height=480`) so the source is a clean 2x upscale with
**no stretch and no muxer padding**; `nvinfer` then letterboxes 640x480 ->
640x640 with the existing `symmetric-padding=1`, geometrically identical to the
Ultralytics letterbox (only the band color, black vs 114-gray, still differs —
stock `nvinfer` exposes no pad-value knob).

Padding knobs investigated:

- `nvstreammux enable-padding` exists but pads with **black** bands; with an
  exact-aspect muxer (640x480) no muxer padding occurs at all, which is cleaner.
- `nvinfer` already uses `maintain-aspect-ratio=1` + `symmetric-padding=1`, so
  padding *geometry* already matches Ultralytics; the 114-gray *color* is not
  configurable and would require an `nvdspreprocess` letterbox stage.

Result at the production threshold (`pre-cluster-threshold=0.25`), versus the
1280x720 stretched run:

| Event | Python (baseline) | DeepStream stretched (1280x720) | DeepStream aspect-fix (640x480) |
| --- | --- | --- | --- |
| fire | 3 dets, max 0.409 | 20 dets, max 0.432 | 33 dets, max 0.458 |
| falldown | 1 det, max 0.419 | 40 dets, max 0.581 | 37 dets, max 0.584 |
| smoke | 1 det, max 0.261 | 4 dets, max 0.371 | 1 det, max 0.351 |

DeepStream command (only the muxer `width/height` changed from the original):

```bash
docker run --rm --gpus all -e VRS_YOLOE_PARSER_DEBUG=1 \
  -v "$PWD/runs/engines:/models:ro" \
  -v "$PWD/configs/deepstream:/etc/vrs/deepstream:ro" \
  -v "/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire:/clips:ro" \
  -v "$PWD/runs/parity:/out" \
  vrs-deepstream:ds8 \
  --pipeline "filesrc location=/clips/fire1.avi ! avidemux ! decodebin ! videoconvert \
    ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 \
    ! m.sink_0 nvstreammux name=m batch-size=1 width=640 height=480 live-source=0 \
    ! nvinfer config-file-path=/etc/vrs/deepstream/pgie-yoloe-safety.txt \
    ! fakesink name=sink sync=false" \
  --out /out/deepstream_mivia_aspectfix.jsonl --stream-id mivia --source-id mivia \
  --detector-id deepstream-yoloe-aspectfix \
  --labels /etc/vrs/deepstream/yoloe-safety-labels.txt
```

Strict aligned parity, using the square muxer with worker bbox transform
(`--bbox-offset-y -80 --bbox-scale-x 0.5 --bbox-scale-y 0.5`):

- `time_tolerance_s=0.08`, IoU >= 0.5: 1/5 Python detections matched.
- The stable match is **falldown @ 44.8s**, IoU 0.944, DS conf 0.490 vs Python
  0.419.
- With a looser 1.0s temporal tolerance, 4/5 match: three fire detections plus
  falldown. Fire IoU ranges 0.517-0.926. This supports spatial agreement for
  fire, but the timing jitter is too large to call strict parity.
- Smoke remains weak: Python emits one smoke detection; DeepStream emits one
  smoke-cloud detection, but it does not match under the strict criteria.

Conclusion: the aspect-preserving muxer and bbox transform materially improve
MIVIA versus the original failure mode, and falldown parity is strong. Fire is
spatially plausible under looser temporal alignment, but strict end-to-end
parity is not yet accepted. Smoke remains open.

## Implemented Fix: Source-Agnostic Aspect-Preserving Muxer

Rather than hardcoding a 4:3 muxer per clip, the reference pipeline now uses a
**square muxer that matches the detector input with padding enabled**:

```text
nvstreammux name=m batch-size=1 width=640 height=640 enable-padding=1 live-source=0
```

This letterboxes any source aspect ratio into the 640x640 detector space with no
stretch, regardless of source resolution. Because object metadata remains in
padded muxer coordinates, the worker must be given the corresponding bbox
transform when source-frame `detection.v1` boxes are required.

On MIVIA, square muxer plus worker transform reproduces the exact-aspect
detector counts:

| Config | fire | falldown | smoke |
| --- | --- | --- | --- |
| 1280x720 (stretch) | 20 / 0.432 | 40 / 0.581 | 4 / 0.371 |
| 640x480 (exact 4:3) | 33 / 0.458 | 37 / 0.584 | 1 / 0.351 |
| 640x640 `enable-padding=1` (source-agnostic) | 33 / 0.458 | 37 / 0.584 | 1 / 0.351 |

Applied in:

- `native/deepstream/src/main.cpp` `example_pipeline()` (reference pipeline)
- `configs/deepstream/ds8-file-example.pipeline`
- header comment in `configs/deepstream/pgie-yoloe-safety.txt`
- worker bbox transform CLI options for source-coordinate output

(The worker consumes the pipeline string via `--pipeline`/`gst_parse_launch`, so
it does not hardcode muxer dims itself; the fix is the reference pipeline that
operators and the Helm command supply. Rebuilding the image only refreshes the
baked `--print-example-pipeline` text.)

## Padding-Color Parity (114-gray): Measured and Deprioritized

DeepStream `nvstreammux`/`nvinfer` pad with **black (0)**; Ultralytics letterbox
pads with **114-gray**. Stock DeepStream exposes no pad-value knob (it would
require a custom `nvdspreprocess` CUDA library). Before building that, we
measured the effect directly at the model level — same MIVIA fire frames,
letterboxed to 640x640 with 114 vs 0, through the Ultralytics torch model:

| Event | pad=114 (Ultralytics) | pad=0 (black/DeepStream) |
| --- | ---: | ---: |
| fire | 0.413 | 0.381 |
| falldown | 0.184 | 0.229 |
| smoke | 0.044 | 0.038 |

The difference is ~8% on fire and is not consistently signed (falldown is higher
with black). With the aspect ratio corrected, padding color does not appear to
be the first thing to fix. **Decision: do not build a custom 114-gray
preprocessing library yet.** The remaining fire/smoke class-score divergence
should be investigated with a per-frame raw-tensor tool (PyTorch/ONNX/TRT/
DeepStream on one decoded frame) before adding custom preprocessing.

## VRSMeta Production Path Recheck

After adding the `vrsmeta` GStreamer element, the production path was rechecked
with the worker running as a launcher only:

```text
vrs-deepstream-worker --disable-probe
  -> nvinfer
  -> nvtracker
  -> vrsmeta output-path=/out/fire120-ds8-disable-probe-vrsmeta.jsonl
```

Input:

- Clip: `/data/vrs/fire_dataset/extracted_sample/fire/120.mp4`
- Source geometry: 640x360, 24 fps, 132 frames
- Muxer: `width=640 height=640 enable-padding=1`
- Engine: `runs/engines/yoloe-11s-safety-rx5080-ds8-trtexec.engine`
- PGIE config: `configs/deepstream/pgie-yoloe-safety.txt`

Result:

- `vrsmeta` wrote 133 valid `detection.v1` records.
- Classes: `billowing smoke=127`, `smoke cloud=6`.
- Score range: 0.278257 to 0.726425.
- Redis bridge smoke published all 133 records into `vrs.detections`.

Python baseline:

```bash
uv run scripts/export_python_detections.py \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/parity/ds8-vrsmeta-20260630/python_fire120_policy.jsonl \
  --stream-id fire120 \
  --detector-id python-yoloe \
  /data/vrs/fire_dataset/extracted_sample/fire/120.mp4
```

The full Python -> DS8 `vrsmeta` -> strict/loose parity flow is now wrapped as:

```bash
uv run scripts/run_deepstream_vrsmeta_parity.py \
  --source /data/vrs/fire_dataset/extracted_sample/fire/120.mp4 \
  --out-dir runs/parity/ds8-vrsmeta-runner-20260630 \
  --stream-id fire120
```

This command probes the source geometry with `ffprobe`, exports the Python
baseline, runs the DS8 container with `vrs-deepstream-worker --disable-probe`
and `vrsmeta`, then writes strict and loose parity reports plus a summary JSON.

The baseline produced 18 `smoke` records. Because the source is 640x360 and the
DeepStream muxer is 640x640 with padding, candidate boxes must be de-padded by
140 px on Y. `scripts/compare_aligned_detector_parity.py` now supports this via
`--auto-candidate-letterbox`:

```bash
uv run scripts/compare_aligned_detector_parity.py \
  --python-detections runs/parity/ds8-vrsmeta-20260630/python_fire120_policy.jsonl \
  --candidate-detections runs/deepstream-verify/fire120-ds8-disable-probe-vrsmeta.jsonl \
  --policy configs/policies/safety.yaml \
  --time-tolerance-s 0.08 \
  --iou-threshold 0.5 \
  --auto-candidate-letterbox \
  --source-width 640 \
  --source-height 360 \
  --candidate-mux-width 640 \
  --candidate-mux-height 640 \
  --out runs/parity/ds8-vrsmeta-20260630/parity_fire120_strict_auto.json
```

Parity:

| Tolerance | Matched | Python unmatched | Candidate unmatched | Mean IoU | Mean dt | Mean DS-Python score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08s | 17/18 | 1 | 116 | 0.941 | 0.022s | +0.210 |
| 1.0s | 18/18 | 0 | 115 | 0.946 | 0.176s | +0.199 |

This recheck validates the `vrsmeta` production export path and confirms that
letterbox de-padding is required for source-coordinate parity. It does not close
the broader detector parity gate for all classes and clips; it strengthens the
smoke path on `120.mp4` and leaves MIVIA fire/falldown/smoke as the broader
acceptance set.
