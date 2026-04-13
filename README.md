# VRS — Video Reasoning System

A modern, two-stage CCTV / video-understanding pipeline that runs on **a single 16 GB GPU**.

- **Fast path:** open-vocabulary detection with **YOLOE-L** (~6 ms / frame on a T4).
  Add or change event classes by editing one YAML line — no prompt-bank curation,
  no per-class fine-tuning.
- **Slow path:** physical-world reasoning with **NVIDIA Cosmos-Reason2-2B** to
  verify each candidate alert (true alert vs. false alarm), surface false
  negatives ("the detector missed something"), and return bounding boxes /
  trajectories with a short rationale.

## Why this design

Classical CLIP-style classifiers (the "encode each frame and cosine-match a
prompt bank" approach) work, but they require continuous **prompt
maintenance**: every new event type, every new camera, every site-specific
quirk needs new prompts and re-scoring. They also produce only frame-level
scores, not localizations.

VRS replaces both stages with their 2026-current best-of-class:

| Stage | Model | Why |
|-------|-------|-----|
| Detect | **Ultralytics YOLOE-L** (open-vocab YOLOv10) | 161 FPS @ 36.8 mAP on LVIS, takes class names as text, returns bounding boxes |
| Reason | **nvidia/Cosmos-Reason2-2B** (Qwen3-VL-2B post-trained for physical reasoning) | 256K context, native multi-frame video at FPS=4, native bbox/point/trajectory output, embodied-AI common sense |

Sources:
- YOLOE — Ultralytics docs and CVPR'25 paper
- Cosmos-Reason2 — `nvidia/Cosmos-Reason2-2B`, December 2025 release

## Architecture

```
                ┌─── FAST PATH (every frame, ~6 ms) ─────────────┐
RTSP/mp4 ─► Reader ─► YOLOE-L ─► per-class score+bbox ─► EventState
                       │                                          │
                       │       (VRAM ~0.5 GB)                     │ candidate
                       └──────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                ┌─── SLOW PATH (only on candidate, ~1-2 s) ──────┐
                │ Cosmos-Reason2-2B (BF16 ~5 GB / W4A16 ~1.7 GB) │
                │  • multi-frame video at FPS=4                  │
                │  • CoT verify + bbox + trajectory              │
                │  • returns true_alert / false_alarm / FN-class │
                └────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                              alerts.jsonl + annotated.mp4
```

## Install

```bash
conda create -n vrs python=3.12 -y
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

Cosmos-Reason2-2B (BF16) and YOLOE both run cleanly on Blackwell; the 16 GB
of GDDR7 is comfortable for the default cascade with room to spare.

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
shared Cosmos-Reason2-2B serve every stream; per-stream outputs land under
`runs/live/<stream_id>/{alerts.jsonl, annotated.mp4}`.

Outputs:

- `runs/<name>/alerts.jsonl` — one JSON per verified alert (verdict, confidence, bbox, trajectory, rationale)
- `runs/<name>/annotated.mp4` — overlay of detector boxes + verifier verdict + Cosmos bboxes/trajectories

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
    verifier: "Open flames or active fire indoors"        # Cosmos CoT prompt
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

| Profile | Detector | Verifier | Total VRAM |
|---------|----------|----------|------------|
| `default.yaml` (recommended for 16 GB) | YOLOE-L FP16 | Cosmos-Reason2-2B BF16 | ~8.5 GB |
| `tiny.yaml` (8 GB cards / Jetson) | YOLOE-S FP16 | Cosmos-Reason2-2B W4A16 | ~3.5 GB |

## Multi-stream architecture

One 16 GB GPU handles ~20–25 FHD streams at 4 FPS under a quiet-site alert rate,
using a single YOLOE instance and a single Cosmos-Reason2-2B instance shared
across all streams. See `vrs/multistream/` for the topology:

```
RTSP[i] ─► DecoderThread[i] ──┐
                              ▼ bounded FrameQueue (drop-oldest)
                    DetectorWorker  (batched YOLOE, per-stream EventStateQueue)
                              │
                              ▼ bounded CandidateQueue
                    VerifierWorker  (Cosmos-Reason2-2B, drop-oldest on overflow)
                              │
                              ▼
                    SinkWorker[i]  ─► runs/<out>/<stream_id>/{alerts.jsonl, annotated.mp4}
```

Decoder backends (`multistream.decoder_backend`):
- `opencv` (default) — FFmpeg CPU decode. Works everywhere.
- `nvdec` — `cv2.cudacodec.VideoReader` for hardware-accelerated NVDEC decode.
  Requires OpenCV built with CUDA; gracefully falls back to `opencv` otherwise.
- `deepstream` — reserved. A full DeepStream 8.0 path (pyds → `nvstreammux` →
  `nvinfer` with TRT-exported YOLOE → custom `nvinferserver` for Cosmos) would
  keep frames in NVMM (zero-copy) end-to-end; it also requires TRT export of
  both models. Out of scope for this release — the Reader interface is small
  enough that adding it is additive.

## Layout

```
vrs/
├── ingest/        RTSP/mp4 frame iterator (single-stream, OpenCV)
├── triage/        YOLOE detector (+ batched inference), per-stream event-state queue
├── verifier/      Cosmos-Reason2-2B reasoning + CoT prompts
├── policy/        Plain-English watch policy parser
├── runtime/       Cosmos-Reason2-2B BF16 / W4A16 loader
├── sinks/         JSONL writer, annotated-video writer (with bboxes/trajectories)
├── multistream/   N-stream cascade: decoders, workers, queues, pipeline
├── pipeline.py    Single-stream cascade orchestration
└── schemas.py     Frame / Detection / CandidateAlert / VerifiedAlert
```
