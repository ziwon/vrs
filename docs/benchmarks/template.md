# Benchmark note template

Copy this file to `docs/benchmarks/<gpu>-<profile>-<date>.md` after running the
benchmark on the target host.

## Summary

| Field | Value |
|-------|-------|
| Date | YYYY-MM-DD |
| Status | validated / failed / exploratory |
| GPU model | |
| VRAM | |
| Driver version | |
| CUDA version | |
| OS / kernel | |
| Python version | |
| torch version | |
| torch CUDA extra | `cu128` / `cu121` / other |
| transformers version | |
| ultralytics version | |
| VRS commit | |
| Config | `configs/default.yaml` / `configs/tiny.yaml` / other |
| Policy | `configs/policies/safety.yaml` / other |
| Source clips | `runs/test_clips` / dataset path |

## Runtime profile

| Component | Value |
|-----------|-------|
| Detector backend | `ultralytics` / `tensorrt` |
| Detector model | |
| Detector dtype | FP16 / FP32 / other |
| Verifier backend | `transformers` / `vllm` / served / other |
| Verifier model | |
| Verifier dtype / quantization | BF16 / FP16 / W4A16 / other |
| Constrained decoding backend | xgrammar / backend-native / none |

## Commands

```bash
uv run scripts/make_test_clips.py --out runs/test_clips
uv run scripts/bench.py --clips runs/test_clips --out runs/bench
```

Environment capture:

```bash
nvidia-smi
uv run python - <<'PY'
import platform
import torch
import transformers
import ultralytics

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("ultralytics", ultralytics.__version__)
PY
```

## Results

| Metric | Value |
|--------|-------|
| Peak VRAM | |
| Single-stream FPS | |
| Multi-stream count | |
| Frame queue drops | |
| Candidate queue drops | |
| Alerts written | |
| Verifier latency p50 | |
| Verifier latency p95 | |
| Verifier latency p99 | |
| Raw report path | `runs/bench/bench_report.json` |

## Notes

- Record any first-run model downloads separately from benchmark timing.
- Note whether verifier latency was measured directly or inferred from the
  end-to-end benchmark.
- Synthetic clips validate runtime plumbing and memory behavior, not model
  accuracy.
