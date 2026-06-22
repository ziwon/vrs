# Runtime validation matrix

VRS runtime claims must be tied to a concrete GPU, CUDA/PyTorch stack, detector,
verifier backend, dtype/quantization profile, and benchmark artifact. A row is
validated only when it links to a benchmark note under `docs/benchmarks/` and the
commands below can reproduce the same profile on equivalent hardware.

The repository may include exploratory GPU smoke artifacts. Until a row is
promoted into the validated section, treat it as an implementation target or
smoke-test candidate, not a capacity guarantee.

## Status definitions

| Status | Meaning |
|--------|---------|
| Validated | Run on the named GPU class with the listed driver, CUDA, Python, torch, transformers, ultralytics, detector, verifier backend, dtype/quantization, peak VRAM, throughput, queue drops, and verifier latency recorded. |
| Smoke-tested | End-to-end runtime path has been exercised on target hardware, but the repo does not yet include the complete throughput, queue, and p50/p95/p99 evidence required for deployment sizing. |
| Unvalidated | Config exists or is expected to run, but the repo does not yet include a complete benchmark note for the exact runtime combination. |
| Planned | Design target or reserved backend that still needs implementation or an end-to-end benchmark. |

## Validated profiles

No validated GPU runtime profiles are committed yet.

When adding one, copy `docs/benchmarks/template.md`, fill every required field,
commit the benchmark note, and add a row here. Do not mark a profile validated
from model size estimates alone.

| GPU model | VRAM | Driver | CUDA | Python | torch | transformers | ultralytics | Detector | Verifier backend/model | dtype/quantization | Peak VRAM | Single-stream FPS | Multi-stream queue drops | Verifier p50/p95/p99 | Evidence |
|-----------|------|--------|------|--------|-------|--------------|-------------|----------|------------------------|--------------------|-----------|-------------------|--------------------------|--------------------|----------|
| _None committed_ | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Smoke-tested profiles

These profiles have a concrete hardware smoke result, but they are not yet
validated for deployment sizing.

| Target class | Hardware | Config / install | Detector | Verifier backend/model | dtype/quantization | Status notes |
|--------------|----------|------------------|----------|------------------------|--------------------|--------------|
| RTX 5080 local W4A16 Compose smoke | RTX 5080 16GB, driver 580.159.03, CUDA 13.0 host / CUDA 12.8 inference image | `docker compose -f docker-compose.yaml -f docker-compose.hf-local.yaml --profile inference up --build` with `.env` Hugging Face token and `configs/tiny.yaml` | YOLOE-S FP16 (`yoloe-11s-seg.pt`) | `transformers`, `embedl/Cosmos-Reason2-2B-W4A16` | W4A16 / `compressed-tensors` | Compose falldown RTSP smoke completed. The worker loaded the gated model, produced `runs/live/alerts.jsonl` and thumbnails, and the API exposed live alerts. This is not yet a deployment-sizing benchmark because peak VRAM and verifier latency percentiles were not recorded. |
| RTX 5080 local vLLM Cosmos smoke/eval | RTX 5080 16GB, driver 580.159.03, Linux 6.17.0-35, torch CUDA 12.8 | `uv sync --extra cu128 --extra vllm` with `configs/vllm-cosmos.yaml` and `vllm==0.19.1` | YOLOE-L FP16 (`yoloe-11l-seg.pt`) | `vllm`, `nvidia/Cosmos-Reason2-2B` | BF16 | Live smoke produced valid structured JSON; six-clip verifier eval produced malformed JSON rate 0.0, verifier p50/p95/p99 721.3/7698.5/11026.8 ms, and weak F1 0.0357. Treat this as backend validation, not deployment sizing or model adoption evidence. See `docs/benchmarks/rtx5080-vllm-cosmos-validation-2026-06-22.md`. |
| Jetson Orin local VLM verifier | Jetson Orin 64GB, L4T R36.4.3, CUDA 12.6, torch 2.7.0 | `scripts/setup_jetson_venv.sh` with `configs/jetson-qwen3vl2b.yaml` | YOLOE-S FP16 (`yoloe-11s-seg.pt`) | `transformers`, `Qwen/Qwen3-VL-2B-Instruct` | BF16 | Exploratory RTSP falldown smoke only. Memory headroom is comfortable, but verifier latency is still slow-path only. See `docs/benchmarks/jetson-orin-qwen3vl2b-vlm-2026-05-12.md`. |
| Jetson Orin 7B local VLM verifier | Jetson Orin 64GB, Python 3.10.12, torch 2.7.0, CUDA device `Orin` visible with 62841 MiB | `scripts/setup_jetson_venv.sh` with `configs/jetson-qwen25vl7b.yaml` | YOLOE-S FP16 (`yoloe-11s-seg.pt`), Ultralytics 8.3.153 | `transformers`, `Qwen/Qwen2.5-VL-7B-Instruct` | BF16 | RTSP falldown smoke completed on Orin. The run produced 73 verifier alerts, all parsed with structured JSON and thumbnails, with a 41 MiB artifact directory. This is not yet a deployment-sizing benchmark because verifier latency percentiles and peak VRAM were not recorded. See `docs/benchmarks/jetson-orin-qwen25vl7b-vlm-plan-2026-05-14.md`. |

## Unvalidated profiles

These profiles are available as configs or documented install paths. They still
need complete benchmark notes before they can be used in deployment sizing.

| Target class | Expected GPU examples | Config / install | Detector | Verifier backend/model | dtype/quantization | Status notes |
|--------------|-----------------------|------------------|----------|------------------------|--------------------|--------------|
| 24 GB+ local GPU | RTX 4090, L4, L40, H100-class hosts | `uv sync --extra cu121` or target-specific CUDA extra with `configs/default.yaml` | YOLOE-L FP16 (`yoloe-11l-seg.pt`) | `transformers`, `nvidia/Cosmos-Reason2-2B` | BF16 | Accuracy-oriented local profile. Validate peak VRAM and verifier latency on the target host before claiming capacity. |
| 16 GB Blackwell local GPU | RTX 5080 / GB205 | `uv sync --extra cu128 --extra quant` with `configs/tiny.yaml`, or a target-specific served verifier profile | YOLOE-S FP16 (`yoloe-11s-seg.pt`) unless benchmark proves YOLOE-L fits | `transformers`, `embedl/Cosmos-Reason2-2B-W4A16`, or a served VLM backend | W4A16 or backend-specific quantization | Local Compose plumbing has been smoke-tested on one RTX 5080. This remains unvalidated for deployment sizing until a benchmark note records peak VRAM, verifier latency percentiles, queue drops, and throughput. |
| 8-16 GB tight local GPU | Jetson Orin, shared workstation GPUs, smaller edge cards | `uv sync --extra cu128 --extra quant` with `configs/tiny.yaml` adjusted for the host CUDA stack | YOLOE-S FP16 | `transformers`, `embedl/Cosmos-Reason2-2B-W4A16` | W4A16 | Intended for smoke and capacity validation. Record drops under realistic stream counts before production use. |

## Planned profiles

| Target class | Detector | Verifier backend/model | Runtime notes |
|--------------|----------|------------------------|---------------|
| Multi-stream production GPU | TensorRT-exported YOLOE engine | vLLM-served Cosmos or Qwen-class VLM | Requires sustained end-to-end benchmark under realistic alert rates. The RTX 5080 local vLLM Cosmos path is smoke/eval-tested, but not yet a multistream capacity claim. |
| DeepStream / zero-copy path | TensorRT YOLOE through DeepStream | Served verifier backend | Reserved for a future DeepStream 8.0 path; not implemented in this release. |
| TRT-LLM verifier | YOLOE-L or YOLOE-S | TRT-LLM | Backend name is reserved in code, but the implementation is not validated. |
| Qwen-class verifier comparison | YOLOE-L or YOLOE-S | Qwen3.5/Qwen3.6-class served VLM | Evaluation target before production model lock-in; requires accuracy and runtime reports. |

## Benchmark commands

Generate synthetic smoke clips and run both single-stream and multi-stream
benchmarks:

```bash
uv run scripts/make_test_clips.py --out runs/test_clips
uv run scripts/bench.py --clips runs/test_clips --out runs/bench
```

Run a specific config:

```bash
uv run scripts/bench.py \
  --clips runs/test_clips \
  --config configs/tiny.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/bench-tiny
```

Run a shorter multi-stream smoke during runtime bring-up:

```bash
uv run scripts/bench.py \
  --clips runs/test_clips \
  --out runs/bench-smoke \
  --multi-replicas 1 \
  --multi-runtime-s 15
```

The benchmark writes `bench_report.json` under the chosen output directory. Copy
the relevant values into a benchmark note and keep the raw JSON path or artifact
location in the note.

## Required benchmark note fields

Every promoted row must record:

- GPU model and VRAM
- Driver version and CUDA version
- Python version
- torch version and selected CUDA extra
- transformers version
- ultralytics version
- detector model and detector backend
- verifier model and verifier backend
- dtype or quantization mode
- peak VRAM
- single-stream FPS
- multi-stream queue drops
- verifier p50/p95/p99 latency, or `not measured` with the reason
- benchmark command, config, policy, and source clips
- link or path to `bench_report.json`

Synthetic clips only validate pipeline plumbing, throughput, memory behavior,
and queue pressure. They do not validate model quality. Accuracy claims need a
separate dataset evaluation report.
