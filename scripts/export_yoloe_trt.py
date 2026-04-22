"""Export a YOLOE checkpoint to a TensorRT ``.engine`` using Ultralytics.

This is the faster of the two export paths documented in
``vrs/triage/tensorrt_detector.py``. It loads the watch policy so the
text prompt embeddings are baked into the engine exactly the way the
runtime pipeline uses them.

Usage::

    python scripts/export_yoloe_trt.py \\
        --weights yoloe-11l-seg.pt \\
        --policy  configs/policies/safety.yaml \\
        --imgsz   640 \\
        --half    \\
        --out     runs/engines/yoloe-11l-seg.engine

Then point ``configs/default.yaml``'s ``detector.model`` at the ``.engine``
file and set ``detector.backend: tensorrt``.

Notes:

- TRT engines are GPU-architecture-specific. An engine built on Ada
  (RTX 4090) will refuse to load on Blackwell (RTX 5080). Build on the
  same arch as the deployment target.
- ``set_classes`` / ``get_text_pe`` *must* run before the export so the
  text-PE reparameterization is fused into the engine.
- For operator-driven workflows with fine-tuning + QAT, NVIDIA TAO's
  ``tao-deploy`` flow produces an equivalent ``.engine`` that this
  runtime accepts identically. This script covers the common zero-shot
  case.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vrs import setup_logging
from vrs.policy import load_watch_policy

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Export YOLOE to TensorRT via Ultralytics")
    ap.add_argument("--weights", required=True, help="YOLOE .pt checkpoint")
    ap.add_argument("--policy", default="configs/policies/safety.yaml",
                    help="watch policy whose prompts will be baked in")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true",
                    help="FP16 precision in the engine (recommended on Ada/Hopper/Blackwell)")
    ap.add_argument("--device", default="0",
                    help="CUDA device ordinal to build against")
    ap.add_argument("--out", default=None,
                    help="optional destination path; if omitted, Ultralytics writes next to --weights")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    # Heavy imports are lazy so `--help` works on CPU-only hosts.
    from ultralytics import YOLOE

    policy = load_watch_policy(args.policy)
    prompts = policy.yoloe_vocabulary()
    logger.info("loading YOLOE from %s (policy has %d prompts)", weights, len(prompts))

    model = YOLOE(str(weights))
    model.set_classes(prompts, model.get_text_pe(prompts))

    logger.info("exporting TRT engine (imgsz=%d, half=%s, device=%s)…",
                args.imgsz, args.half, args.device)
    engine_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
    )
    engine_path = Path(str(engine_path))

    if args.out:
        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        engine_path.rename(dst)
        engine_path = dst

    print(f"\nWritten: {engine_path}")
    print("\nNext — point your config at it:")
    print("  detector:")
    print("    backend: tensorrt")
    print(f"    model: {engine_path}")


if __name__ == "__main__":
    main()
