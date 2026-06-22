# RTX 5080 vLLM Cosmos validation

## Summary

| Field | Value |
|-------|-------|
| Date | 2026-06-22 |
| Status | smoke-tested |
| GPU model | NVIDIA GeForce RTX 5080 |
| VRAM | 16,303 MiB reported by `nvidia-smi`; 15,806 MiB reported by torch |
| Driver version | 580.159.03 |
| CUDA version | torch CUDA 12.8 |
| OS / kernel | Linux 6.17.0-35-generic |
| Python version | 3.11.13 |
| torch version | 2.10.0+cu128 |
| torch CUDA extra | `cu128` |
| transformers version | 5.6.2 |
| ultralytics version | 8.4.41 |
| vLLM version | 0.19.1 |
| VRS commit | `0ba1884` plus local Task 6 validation changes |
| Config | `configs/vllm-cosmos.yaml` |
| Policy | `configs/policies/safety.yaml` |
| Source clips | `/data/vrs/verifier-eval` |

This run validates the local vLLM backend path against
`nvidia/Cosmos-Reason2-2B` on an RTX 5080. It does not validate production
stream capacity. The eval CLI currently rebuilds the vLLM engine per video, so
model load and graph capture are not representative of a long-lived verifier
worker.

## Runtime profile

| Component | Value |
|-----------|-------|
| Detector backend | `ultralytics` |
| Detector model | `yoloe-11l-seg.pt` |
| Detector dtype | FP16 |
| Verifier backend | `vllm` |
| Verifier model | `nvidia/Cosmos-Reason2-2B` |
| Verifier dtype / quantization | BF16 |
| Constrained decoding backend | vLLM structured outputs / xgrammar |

## Commands

```bash
set -a; [ ! -f .env ] || source .env; set +a
uv run --frozen python scripts/validate_vllm_backend.py \
  --config configs/vllm-cosmos.yaml \
  --policy configs/policies/safety.yaml \
  --class-name fire \
  --out runs/vllm-smoke/result.json

uv run --frozen python scripts/eval_verifier_backends.py \
  --dataset /data/vrs/verifier-eval \
  --dataset-format labeled_dir \
  --policy configs/policies/safety.yaml \
  --out runs/verifier-vllm-bakeoff \
  --candidate vllm=configs/vllm-cosmos.yaml

uv run --frozen python scripts/eval_verifier_backends.py \
  --dataset /data/vrs/verifier-eval \
  --dataset-format labeled_dir \
  --policy configs/policies/safety.yaml \
  --out runs/verifier-vllm-bakeoff \
  --candidate transformers=configs/tiny.yaml \
  --candidate vllm=configs/vllm-cosmos.yaml \
  --skip-run
```

## Smoke result

| Metric | Value |
|--------|-------|
| Raw result path | `runs/vllm-smoke/result.json` |
| Passed | true |
| Structured JSON parsed | true |
| Generation elapsed | 12.3544 s |
| Completion tokens | 94 |
| Tokens per second | 7.6086 |
| Model load memory reported by vLLM | 4.31 GiB |
| Available KV cache memory during smoke | 4.45 GiB |
| GPU KV cache size during smoke | 41,648 tokens |
| Max concurrency at 4,096 tokens/request | 10.17x |

## Eval result

| Metric | vLLM BF16 Cosmos |
|--------|------------------|
| Raw report path | `runs/verifier-vllm-bakeoff/vllm/report.json` |
| Comparison path | `runs/verifier-vllm-bakeoff/verifier_bakeoff.json` |
| Precision / recall / F1 | 0.0189 / 0.3333 / 0.0357 |
| TP / FP / FN | 1 / 52 / 2 |
| Malformed JSON rate | 0.0 |
| Verifier flip rate | 0.274 |
| Detector latency p50 / p95 / p99 | 12.2663 / 19.1324 / 41.1370 ms |
| Verifier latency p50 / p95 / p99 | 721.3354 / 7698.4791 / 11026.7970 ms |
| Verifier tokens/s p50 / p95 | 142.1211 / 181.8401 |

## Baseline comparison

The existing transformers baseline report under
`runs/verifier-vllm-bakeoff/transformers/report.json` used `configs/tiny.yaml`:
YOLOE-S plus `embedl/Cosmos-Reason2-2B-W4A16`. It is not an isolated
backend-only comparison against BF16 Cosmos, but it is the practical 16 GB
baseline available on this host.

| Metric | transformers W4A16 tiny | vLLM BF16 Cosmos |
|--------|--------------------------|------------------|
| Config | `configs/tiny.yaml` | `configs/vllm-cosmos.yaml` |
| Detector | `yoloe-11s-seg.pt` | `yoloe-11l-seg.pt` |
| Verifier model | `embedl/Cosmos-Reason2-2B-W4A16` | `nvidia/Cosmos-Reason2-2B` |
| Precision / recall / F1 | 0.0000 / 0.0000 / 0.0000 | 0.0189 / 0.3333 / 0.0357 |
| Malformed JSON rate | 0.0 | 0.0 |
| Verifier p95 | 1017.3270 ms | 7698.4791 ms |

## Notes

- `vllm` is pinned to `0.19.1` after this validation. That release requires a
  newer torch stack and is incompatible with the repository's `cu121` extra.
- vLLM 0.19 uses `SamplingParams(structured_outputs=StructuredOutputsParams(...))`
  rather than the older `GuidedDecodingParams` import path.
- vLLM's xgrammar structured-output path emitted nanobind leak warnings at
  process exit. The smoke and eval commands still exited 0 and reported
  `malformed_json_rate: 0.0`.
- The eval quality is too weak for model adoption. Treat this as backend
  validation, not a recommendation to promote Cosmos/vLLM as the production
  verifier.
- A Qwen-class served or vLLM-supported candidate still needs a comparable eval
  before production model lock-in.
