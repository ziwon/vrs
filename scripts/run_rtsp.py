"""Run VRS over a live RTSP stream.

Example:
    python scripts/run_rtsp.py \
        --rtsp rtsp://u:p@cam.local:554/stream1 \
        --config configs/default.yaml \
        --policy configs/policies/safety.yaml \
        --out runs/live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrs.pipeline import build_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="VRS — run on a live RTSP stream")
    p.add_argument("--rtsp", required=True, help='RTSP URL, e.g. "rtsp://u:p@host:554/stream"')
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--policy", default="configs/policies/safety.yaml")
    p.add_argument("--out", default="runs/live", help="output directory")
    args = p.parse_args()

    pipeline = build_pipeline(args.config, args.policy, args.out)
    pipeline.run(args.rtsp)


if __name__ == "__main__":
    main()
