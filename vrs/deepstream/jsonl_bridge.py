"""Publish DeepStream ``detection.v1`` JSONL records to EventTransport."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TextIO

from ..transport import EventMessage, RedisStreamsConfig, RedisStreamsTransport


def detection_message(record: dict, *, stream: str) -> EventMessage:
    key = str(
        record.get("idempotency_key")
        or record.get("detection_id")
        or record.get("stream_id")
        or "deepstream"
    )
    headers = {
        "schema_version": str(record.get("schema_version", "")),
        "record_type": str(record.get("record_type", "")),
        "source_runtime": str(record.get("source_runtime", "")),
    }
    return EventMessage(stream=stream, key=key, payload=record, headers=headers)


def publish_jsonl_file(path: str | Path, transport, *, stream: str) -> int:
    count = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if publish_line(line, transport, stream=stream):
                count += 1
    return count


def publish_line(line: str, transport, *, stream: str) -> bool:
    text = line.strip()
    if not text:
        return False
    record = json.loads(text)
    if not isinstance(record, dict):
        raise ValueError("DeepStream JSONL record must be an object")
    transport.publish(detection_message(record, stream=stream))
    return True


def follow_jsonl_file(
    path: str | Path,
    transport,
    *,
    stream: str,
    poll_interval_s: float = 0.5,
    start_at_end: bool = False,
    stop_after_idle_s: float | None = None,
) -> int:
    path = Path(path)
    published = 0
    idle_started_at: float | None = None
    with path.open(encoding="utf-8") as fh:
        if start_at_end:
            fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                idle_started_at = None
                if publish_line(line, transport, stream=stream):
                    published += 1
                continue
            if stop_after_idle_s is not None:
                now = time.monotonic()
                idle_started_at = idle_started_at or now
                if now - idle_started_at >= stop_after_idle_s:
                    return published
            time.sleep(poll_interval_s)


def wait_for_file(path: Path, *, poll_interval_s: float) -> None:
    while not path.exists():
        time.sleep(poll_interval_s)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Publish DeepStream detection.v1 JSONL to Redis Streams"
    )
    ap.add_argument("--input", required=True, help="DeepStream detection.v1 JSONL path")
    ap.add_argument("--redis-url", required=True, help="Redis URL, e.g. redis://vrs-redis:6379/0")
    ap.add_argument("--stream", default="detections", help="Logical EventTransport stream")
    ap.add_argument("--stream-prefix", default="vrs", help="Redis stream prefix")
    ap.add_argument("--max-len", type=int, default=100_000, help="Approximate Redis stream maxlen")
    ap.add_argument("--poll-interval-s", type=float, default=0.5)
    ap.add_argument(
        "--start-at-end", action="store_true", help="Only publish records appended after startup"
    )
    ap.add_argument("--once", action="store_true", help="Publish existing file and exit")
    return ap


def main(argv: list[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    args = build_arg_parser().parse_args(argv)
    path = Path(args.input)
    wait_for_file(path, poll_interval_s=args.poll_interval_s)
    transport = RedisStreamsTransport(
        RedisStreamsConfig(
            url=args.redis_url,
            stream_prefix=args.stream_prefix,
            max_len=args.max_len,
        )
    )
    if args.once:
        count = publish_jsonl_file(path, transport, stream=args.stream)
    else:
        count = follow_jsonl_file(
            path,
            transport,
            stream=args.stream,
            poll_interval_s=args.poll_interval_s,
            start_at_end=args.start_at_end,
        )
    print(
        f"published {count} records from {path} to {args.stream_prefix}.{args.stream}", file=stdout
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
