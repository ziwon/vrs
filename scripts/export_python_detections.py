"""Export Python detector output as canonical detection.v1 JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrs import setup_logging
from vrs.eval.detection_export import detections_to_contracts, write_detection_jsonl
from vrs.ingest import StreamReader
from vrs.pipeline import load_config
from vrs.policy import load_watch_policy
from vrs.triage import YOLOEConfig, build_detector


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Export Python detector output as detection.v1 JSONL")
    ap.add_argument("sources", nargs="+", help="Video/RTSP/image sources to process")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--out", default="runs/parity/python_detections.jsonl")
    ap.add_argument("--stream-id", help="Override stream_id for a single source")
    ap.add_argument("--detector-id", default="python-yoloe")
    args = ap.parse_args()

    cfg = load_config(args.config, verifier_enabled=False)
    policy = load_watch_policy(args.policy)
    det_cfg = cfg["detector"]
    detector = build_detector(
        YOLOEConfig(
            model=det_cfg["model"],
            device=det_cfg.get("device", "cuda"),
            imgsz=int(det_cfg.get("imgsz", 640)),
            conf_floor=float(det_cfg.get("conf_floor", 0.20)),
            iou=float(det_cfg.get("iou", 0.50)),
            half=bool(det_cfg.get("half", True)),
        ),
        policy,
        backend=det_cfg.get("backend", "ultralytics"),
    )

    records = []
    target_fps = float(cfg["ingest"]["target_fps"])
    for raw_source in args.sources:
        source = Path(raw_source)
        clip_id = source.stem if source.exists() else raw_source
        stream_id = args.stream_id or clip_id
        reader = StreamReader(source=raw_source, target_fps=target_fps)
        for frame in reader:
            detections = detector(frame)
            records.extend(
                detections_to_contracts(
                    detections,
                    frame=frame,
                    stream_id=stream_id,
                    clip_id=clip_id,
                    detector_id=args.detector_id,
                )
            )

    write_detection_jsonl(args.out, records)
    print(f"written {args.out}: {len(records)} detection.v1 records")


if __name__ == "__main__":
    main()
