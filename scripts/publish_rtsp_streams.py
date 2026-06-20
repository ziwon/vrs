"""Publish a local stream manifest to MediaMTX with FFmpeg."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _load_streams(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    streams = cfg.get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"{path}: expected top-level streams list")
    return [s for s in streams if isinstance(s, dict)]


def _ffmpeg_cmd(stream: dict[str, Any]) -> list[str]:
    source = str(stream["source"])
    rtsp = str(stream["rtsp"])
    fps = int(stream.get("fps") or 30)
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-fflags",
        "+genpts",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={fps},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        rtsp,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish local clips to RTSP streams")
    parser.add_argument("--streams", default="configs/local-rtsp-streams.yaml")
    parser.add_argument("--startup-s", type=float, default=2.0)
    args = parser.parse_args()

    streams_path = Path(args.streams)
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for stream in _load_streams(streams_path):
            stream_id = str(stream.get("id") or "")
            source = Path(str(stream.get("source") or ""))
            rtsp = str(stream.get("rtsp") or "")
            if not stream_id or not rtsp:
                raise ValueError(f"{streams_path}: each stream needs id and rtsp")
            if not source.is_file():
                if stream.get("optional"):
                    print(f"skip optional stream {stream_id}: missing {source}", file=sys.stderr)
                    continue
                raise FileNotFoundError(f"missing source for stream {stream_id}: {source}")

            print(f"publishing {stream_id}: {source} -> {rtsp}", flush=True)
            procs.append((stream_id, subprocess.Popen(_ffmpeg_cmd(stream))))

        if not procs:
            raise RuntimeError("no streams were published")

        time.sleep(max(0.0, args.startup_s))
        failed = [(sid, p.returncode) for sid, p in procs if p.poll() is not None]
        if failed:
            for sid, code in failed:
                print(f"publisher exited early: {sid} status={code}", file=sys.stderr)
            return 1

        print("all publishers running; press Ctrl-C to stop", flush=True)
        while True:
            time.sleep(1)
            for sid, proc in procs:
                if proc.poll() is not None:
                    print(f"publisher exited: {sid} status={proc.returncode}", file=sys.stderr)
                    return proc.returncode or 1
    except KeyboardInterrupt:
        return 0
    finally:
        for _, proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for _, proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
