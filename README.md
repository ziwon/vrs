# VRS — Video Reasoning System

A modern, two-stage CCTV / video-understanding pipeline for a **single local GPU**.

The deployment target is 16 GB cards when the verifier is quantized or otherwise
capacity-tested. For BF16 Cosmos-Reason2-2B, validate on the target host first:
NVIDIA's 2026 model card lists a 24 GB minimum for the reference inference path.

- **Fast path:** open-vocabulary detection with **YOLOE-L** (~6 ms / frame on a T4).
  Add or change event classes by editing one YAML line — no prompt-bank curation,
  no per-class fine-tuning.
- **Slow path:** a pluggable VLM verifier. The current baseline is
  **NVIDIA Cosmos-Reason2-2B**, but 2026 research and internal benchmarks show
  it should not be treated as the final verifier. Qwen3.5/Qwen3.6-class VLMs
  are priority candidates for side-by-side evaluation.

## Why this design

Classical CLIP-style classifiers (the "encode each frame and cosine-match a
prompt bank" approach) work, but they require continuous **prompt
maintenance**: every new event type, every new camera, every site-specific
quirk needs new prompts and re-scoring. They also produce only frame-level
scores, not localizations.

VRS uses two practical 2026-era baselines, with the verifier intentionally
kept swappable:

| Stage | Model | Why |
|-------|-------|-----|
| Detect | **Ultralytics YOLOE-L** (`yoloe-11l-seg.pt` by default) | Open-vocabulary text prompts, returns bounding boxes, no per-site retraining loop |
| Reason | **nvidia/Cosmos-Reason2-2B** baseline | Physical-reasoning specialization, FPS=4 video path, bbox/point/trajectory-oriented prompting. Treat as a baseline, not the expected winner. |

Sources:
- YOLOE — Ultralytics docs and CVPR'25 paper. Ultralytics also publishes newer
  YOLOE-26 models; migrate only after eval confirms a gain for the active policy.
- Cosmos-Reason2 — NVIDIA docs and `nvidia/Cosmos-Reason2-2B` model card.
- Qwen — Qwen3.5/Qwen3.6 official releases report stronger multimodal
  foundation-model capability than earlier Qwen3-VL-era models. Because
  Cosmos-Reason2-2B is derived from Qwen3-VL-2B, Qwen3.5/Qwen3.6-class models
  should be evaluated as verifier backends before production model lock-in.

## Architecture

```
                ┌─── FAST PATH (every frame, ~6 ms) ─────────────┐
RTSP/mp4 ─► Reader ─► YOLOE-L ─► per-class score+bbox ─► EventState
                       │                                          │
                       │       (VRAM ~0.5 GB)                     │ candidate
                       └──────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                ┌─── SLOW PATH (only on candidate) ──────────────┐
                │ VLM verifier backend                           │
                │  • default: Cosmos-Reason2-2B baseline         │
                │  • candidates: Qwen3.5/Qwen3.6 served VLMs     │
                │  • returns true_alert / false_alarm / FN-class │
                └────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                              alerts.jsonl + thumbnails/*.jpg
```

## Install

```bash
conda create -n vrs python=3.11 -y
conda activate vrs

# Pick the right torch build for your GPU architecture:
#   * Blackwell (RTX 5080/5090, B100, GB100) → CUDA 12.8+ / torch 2.6+
#   * Hopper   (H100, H200)                 → CUDA 12.4+
#   * Ada      (RTX 4080/4090, L4, L40)     → CUDA 12.1+
pip install torch --index-url https://download.pytorch.org/whl/cu128     # Blackwell
# or:
# pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121  # Ada/Ampere

pip install -r requirements.txt
```

### RTX 5080 / Blackwell note

RTX 5080 (GB205, Blackwell, SM 12.0) needs **CUDA 12.8+ and torch 2.6+** for
native kernel support. If you get `no kernel image is available for execution
on the device` when YOLOE or Cosmos starts, your torch is too old — install
the cu128 wheel above.

Cosmos-Reason2-2B and YOLOE should be validated on the exact Blackwell host and
runtime you intend to ship. The BF16 model weights are small, but video tokens,
KV cache, and framework overhead can dominate memory; use the W4A16 profile or
a served backend if BF16 does not fit.

### W4A16 profile (≤8 GB cards / Jetson)

```bash
pip install bitsandbytes>=0.43
```

## Run

**Single mp4:**
```bash
python scripts/run_mp4.py \
  --video /path/to/cctv.mp4 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/demo
```

**Single RTSP:**
```bash
python scripts/run_rtsp.py \
  --rtsp rtsp://user:pass@cam.local:554/stream1 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/live
```

**Multi-stream (N cameras on one GPU):**
```bash
python scripts/run_multistream.py \
  --config  configs/default.yaml \
  --policy  configs/policies/safety.yaml \
  --streams configs/multistream.yaml \
  --out     runs/live
```

List your cameras in `configs/multistream.yaml`. One shared YOLOE and one
shared verifier backend serve every stream; per-stream outputs land under
`runs/live/<stream_id>/{alerts.jsonl, thumbnails/*.jpg}`. Annotated MP4 output
is still available as an opt-in debug/demo artifact.

Outputs:

- `runs/<name>/alerts.jsonl` — one JSON per verified alert (verdict, confidence, bbox, trajectory, rationale, thumbnail path)
- `runs/<name>/thumbnails/*.jpg` — one event image per alert, with detector/verifier overlays
- `runs/<name>/annotated.mp4` — optional debug/demo overlay video when `sink.write_annotated: true`

## Smoke-test on your GPU (e.g. RTX 5080)

1. **Generate synthetic plumbing-test clips** (no network, no datasets):

   ```bash
   python scripts/make_test_clips.py --out runs/test_clips
   ```

   This writes 4 FHD mp4s — `fire_test.mp4`, `smoke_test.mp4`,
   `falldown_test.mp4`, `weapon_test.mp4`. They use programmatic OpenCV
   drawing (noise patches, Gaussian blobs, stick figures, silhouettes).

   **They test the pipeline, not the models.** YOLOE's real-world
   training means it will usually *not* raise detections on these clips —
   that's expected. The bench script still gives you real latency, FPS,
   and VRAM numbers.

2. **Benchmark on the local GPU**:

   ```bash
   python scripts/bench.py --clips runs/test_clips --out runs/bench
   ```

   You'll see per-clip single-stream throughput, then a multi-stream run
   (4 clips × 4 replicas = 16 concurrent streams by default) with peak
   VRAM and queue drop counts. A JSON report is saved to
   `runs/bench/bench_report.json`.

3. **For accuracy testing**, use real datasets:
   - Fire/smoke: [FireNet](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection), [D-Fire](https://github.com/gaiasd/DFireDataset)
   - Falldown: [Le2i Fall Detection](http://le2i.cnrs.fr/Fall-detection-Dataset), [UP-Fall Detection](https://sites.google.com/up.edu.mx/har-up/)
   - Multi-class CCTV anomaly: [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)

   Drop the clips into a directory, add them to `configs/multistream.yaml`
   as `file://` sources, and re-run `scripts/bench.py`.

## Watch Policy — the only place you maintain

```yaml
# configs/policies/safety.yaml
watch:
  - name: fire
    detector: ["fire", "open flame", "burning object"]   # YOLOE text prompts
    verifier: "Open flames or active fire indoors"        # VLM verifier prompt
    severity: critical
    min_score: 0.30
    min_persist_frames: 2
  - name: smoke
    detector: ["smoke", "smoke cloud", "billowing smoke"]
    verifier: "Thick smoke filling a space"
    severity: high
    min_score: 0.25
    min_persist_frames: 2
  - name: falldown
    detector: ["person lying on the ground", "fallen person", "collapsed person"]
    verifier: "A person has collapsed and is lying motionless"
    severity: high
    min_score: 0.35
    min_persist_frames: 3
  - name: weapon            # custom — added in 5 seconds
    detector: ["handgun", "knife", "rifle"]
    verifier: "A person holding or brandishing a weapon"
    severity: critical
    min_score: 0.40
    min_persist_frames: 1
```

Adding a custom event is a single block. No bank to score, no embeddings to
recompute, no thresholds to grid-search.

## VRAM profiles

| Profile | Detector | Verifier | Notes |
|---------|----------|----------|-------|
| `default.yaml` | YOLOE-L FP16 | Cosmos-Reason2-2B BF16 | Accuracy-oriented local profile; validate memory on target GPU. NVIDIA's reference model card lists 24 GB minimum. |
| `tiny.yaml` | YOLOE-S FP16 | Cosmos-Reason2-2B W4A16 | Intended for 8-16 GB cards / Jetson-class deployments after quantized-runtime validation. |

## Multi-stream architecture

The multistream path is designed around one shared YOLOE instance and one
shared VLM verifier instance. Sustainable stream count depends on alert
rate, verifier backend, clip length, and GPU memory; measure it with
`scripts/bench.py` on the target host instead of treating a stream count as
portable. See `vrs/multistream/` for the topology:

```
RTSP[i] ─► DecoderThread[i] ──┐
                              ▼ bounded FrameQueue (drop-oldest)
                    DetectorWorker  (batched YOLOE, per-stream EventStateQueue)
                              │
                              ▼ bounded CandidateQueue
                    VerifierWorker  (VLM verifier, drop-oldest on overflow)
                              │
                              ▼
                    SinkWorker[i]  ─► runs/<out>/<stream_id>/{alerts.jsonl, thumbnails/*.jpg}
```

Decoder backends (`multistream.decoder_backend`):
- `opencv` (default) — FFmpeg CPU decode. Works everywhere.
- `nvdec` — `cv2.cudacodec.VideoReader` for hardware-accelerated NVDEC decode.
  Requires OpenCV built with CUDA; gracefully falls back to `opencv` otherwise.
- `deepstream` — reserved. A full DeepStream 8.0 path (pyds → `nvstreammux` →
  `nvinfer` with TRT-exported YOLOE → custom verifier serving path) would
  keep frames in NVMM (zero-copy) end-to-end; it also requires TRT export of
  both models. Out of scope for this release — the Reader interface is small
  enough that adding it is additive.

## Layout

```
vrs/
├── ingest/        RTSP/mp4 frame iterator (single-stream, OpenCV)
├── triage/        YOLOE detector (+ batched inference), per-stream event-state queue
├── verifier/      VLM verifier prompts + structured-output parsing
├── policy/        Plain-English watch policy parser
├── runtime/       VLM runtime backends (Cosmos baseline today)
├── sinks/         JSONL writer, event thumbnails, optional annotated-video writer
├── multistream/   N-stream cascade: decoders, workers, queues, pipeline
├── pipeline.py    Single-stream cascade orchestration
└── schemas.py     Frame / Detection / CandidateAlert / VerifiedAlert
```
