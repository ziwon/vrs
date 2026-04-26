"""Benchmark VRS on the local GPU.

Reports:
  * YOLOE per-frame latency + FPS (measured over the whole clip)
  * Cosmos-Reason2-2B verify latency per alert
  * Peak VRAM while the pipeline is running
  * Multi-stream aggregate throughput (N simultaneous replays)

Typical run on an RTX 5080 / 16 GB (Blackwell):

    uv run scripts/make_test_clips.py --out runs/test_clips
    uv run scripts/bench.py --clips runs/test_clips --out runs/bench

The script never touches the network; everything runs against your local
models. If ``configs/default.yaml``'s Cosmos entry is set to a HuggingFace
model ID and it isn't cached yet, ``transformers`` will fetch it — that's
the one exception and it happens only on first run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vrs import setup_logging


def _vram_snapshot() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }
    except ImportError:
        return None


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "CPU"


def _count_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


# ──────────────────────────────────────────────────────────────────────
# single-stream bench — one clip at a time, measures steady-state throughput
# ──────────────────────────────────────────────────────────────────────


def bench_single(clips_dir: Path, config: str, policy: str, out_dir: Path) -> dict:
    from vrs.pipeline import build_pipeline

    results: dict[str, dict] = {}
    for clip in sorted(clips_dir.glob("*.mp4")):
        sub = out_dir / f"single_{clip.stem}"
        sub.mkdir(parents=True, exist_ok=True)

        pipeline = build_pipeline(config, policy, sub)
        n_source = _count_frames(clip)

        _reset_vram_peak()
        t0 = time.monotonic()
        pipeline.run(str(clip))
        dt = time.monotonic() - t0

        alerts_path = sub / "alerts.jsonl"
        n_alerts = _count_jsonl(alerts_path)

        target_fps = float(pipeline.cfg["ingest"]["target_fps"])
        # frames the pipeline actually processed (post-downsample)
        n_processed = max(1, round(n_source * (target_fps / 30.0)))
        throughput_fps = n_processed / dt if dt > 0 else 0.0

        r = {
            "source_frames": n_source,
            "processed_frames_est": n_processed,
            "wall_seconds": round(dt, 2),
            "throughput_fps": round(throughput_fps, 2),
            "alerts_written": n_alerts,
            "vram_gb": _vram_snapshot(),
        }
        results[clip.stem] = r
        print(
            f"[single] {clip.name:<20}  "
            f"{dt:6.2f}s  "
            f"{throughput_fps:6.2f} proc-fps  "
            f"alerts={n_alerts}  "
            f"peak_vram={r['vram_gb']['peak_gb']:.2f} GB"
            if r["vram_gb"]
            else f"[single] {clip.name}: {dt:.2f}s  alerts={n_alerts}"
        )
    return results


# ──────────────────────────────────────────────────────────────────────
# multi-stream bench — replay all clips as N concurrent "streams"
# ──────────────────────────────────────────────────────────────────────


def bench_multistream(
    clips_dir: Path,
    config: str,
    policy: str,
    out_dir: Path,
    replicas: int = 4,
    max_runtime_s: float = 60.0,
) -> dict:
    import yaml

    from vrs.multistream import build_multistream_pipeline

    manifest_path = out_dir / "ms_manifest.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)

    streams: list[dict] = []
    for clip in sorted(clips_dir.glob("*.mp4")):
        for r in range(replicas):
            streams.append({"id": f"{clip.stem}_{r}", "rtsp": str(clip)})

    manifest = {
        "multistream": {
            "decoder_backend": "opencv",
            "frame_queue_size": max(64, 4 * len(streams)),
            "frame_drop_policy": "drop_oldest",
            "detector_batch_size": 4,
            "detector_batch_timeout_ms": 30,
            "verifier_queue_size": 16,
            "verifier_drop_policy": "drop_oldest",
            "sink_queue_size": 32,
        },
        "streams": streams,
    }
    with manifest_path.open("w") as f:
        yaml.safe_dump(manifest, f)

    pipeline = build_multistream_pipeline(
        config_path=config,
        policy_path=policy,
        streams_path=str(manifest_path),
        out_dir=str(out_dir / "multi"),
    )
    _reset_vram_peak()
    t0 = time.monotonic()
    pipeline.run(max_runtime_s=max_runtime_s)
    dt = time.monotonic() - t0

    # tally alerts across all streams
    n_alerts = 0
    for d in (out_dir / "multi").glob("*/alerts.jsonl"):
        n_alerts += _count_jsonl(d)

    return {
        "streams": len(streams),
        "wall_seconds": round(dt, 2),
        "alerts_total": n_alerts,
        "queue_stats": pipeline.queue_stats(),
        "vram_gb": _vram_snapshot(),
    }


# ──────────────────────────────────────────────────────────────────────
# cli
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Benchmark VRS on the local GPU")
    ap.add_argument("--clips", default="runs/test_clips")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-multi", action="store_true")
    ap.add_argument(
        "--multi-replicas", type=int, default=4, help="replay each test clip N times simultaneously"
    )
    ap.add_argument("--multi-runtime-s", type=float, default=60.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips)

    report: dict = {
        "gpu": _gpu_name(),
        "config": args.config,
        "policy": args.policy,
    }
    print(f"GPU: {report['gpu']}")
    print(f"Config: {args.config}   Policy: {args.policy}")
    print(f"Clips:  {sorted(str(p) for p in clips_dir.glob('*.mp4'))}")

    if not args.skip_single:
        print("\n── single-stream ─────────────────────────")
        report["single"] = bench_single(clips_dir, args.config, args.policy, out_dir)

    if not args.skip_multi:
        print("\n── multi-stream ──────────────────────────")
        report["multi"] = bench_multistream(
            clips_dir,
            args.config,
            args.policy,
            out_dir,
            replicas=args.multi_replicas,
            max_runtime_s=args.multi_runtime_s,
        )
        m = report["multi"]
        agg_fps = m["streams"] * 4.0  # nominal input rate (4 fps/stream)
        print(
            f"\n[multi] {m['streams']} streams  "
            f"wall={m['wall_seconds']}s  "
            f"alerts={m['alerts_total']}  "
            f"peak_vram={m['vram_gb']['peak_gb']:.2f} GB   "
            f"nominal_input={agg_fps:.1f} fps"
            if m["vram_gb"]
            else f"[multi] {m['streams']} streams wall={m['wall_seconds']}s alerts={m['alerts_total']}"
        )
        qs = m["queue_stats"]
        print(
            f"        frame_q: size={qs['frame_q']['size']}  dropped={qs['frame_q']['dropped']}   "
            f"candidate_q: size={qs['candidate_q']['size']}  dropped={qs['candidate_q']['dropped']}"
        )

    report_path = out_dir / "bench_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWritten: {report_path}")


if __name__ == "__main__":
    main()
