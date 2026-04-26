"""Run VRS over N concurrent RTSP / mp4 sources on a single GPU.

Example:
    uv run scripts/run_multistream.py \
        --config  configs/default.yaml \
        --policy  configs/policies/safety.yaml \
        --streams configs/multistream.yaml \
        --out     runs/live
"""

from __future__ import annotations

import argparse

from vrs import setup_logging
from vrs.multistream import build_multistream_pipeline


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="VRS — multi-stream cascade on one GPU")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--policy", default="configs/policies/safety.yaml")
    p.add_argument("--streams", default="configs/multistream.yaml")
    p.add_argument("--out", default="runs/live", help="base output directory (per-stream subdirs)")
    p.add_argument(
        "--max-runtime-s",
        type=float,
        default=None,
        help="stop after N seconds (useful for offline runs)",
    )
    args = p.parse_args()

    pipeline = build_multistream_pipeline(
        config_path=args.config,
        policy_path=args.policy,
        streams_path=args.streams,
        out_dir=args.out,
    )
    pipeline.run(max_runtime_s=args.max_runtime_s)


if __name__ == "__main__":
    main()
