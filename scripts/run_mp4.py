"""Run VRS over an mp4 file.

Example:
    python scripts/run_mp4.py \
        --video /path/to/cctv.mp4 \
        --config configs/default.yaml \
        --policy configs/policies/safety.yaml \
        --out runs/demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrs.pipeline import build_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="VRS — run on an mp4 file")
    p.add_argument("--video", required=True, help="path to mp4/avi/mkv")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--policy", default="configs/policies/safety.yaml")
    p.add_argument("--out", default="runs/demo", help="output directory")
    args = p.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"video not found: {args.video}")

    pipeline = build_pipeline(args.config, args.policy, args.out)
    pipeline.run(args.video)


if __name__ == "__main__":
    main()
