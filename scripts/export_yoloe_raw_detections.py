"""Export unfiltered YOLOE prompt detections as detection.v1 JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vrs import setup_logging
from vrs.eval.detection_export import write_detection_jsonl
from vrs.policy import load_watch_policy
from vrs.schemas import Detection, Frame


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(
        description="Export raw YOLOE prompt detections without policy min_score filtering"
    )
    ap.add_argument("source", help="Video or image source")
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream-id", help="stream_id for records")
    ap.add_argument("--detector-id", default="python-yoloe-raw")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target-fps", type=float, default=0.0, help="0 means every frame")
    args = ap.parse_args()

    from ultralytics import YOLOE

    source = Path(args.source)
    stream_id = args.stream_id or source.stem
    policy = load_watch_policy(args.policy)
    prompts = policy.yoloe_vocabulary()

    model = YOLOE(args.weights)
    model.set_classes(prompts, model.get_text_pe(prompts))

    records: list[dict] = []
    for frame in iter_frames(str(source), target_fps=args.target_fps):
        result = model.predict(
            frame.image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        cls_idx = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, conf, ci in zip(xyxy, confs, cls_idx, strict=True):
            if ci < 0 or ci >= len(prompts):
                continue
            raw_label = prompts[ci]
            det = Detection(
                class_name=raw_label,
                score=float(conf),
                xyxy=tuple(float(v) for v in box),
                raw_label=raw_label,
            )
            records.append(
                det.to_contract(
                    stream_id=stream_id,
                    clip_id=source.stem,
                    frame=frame,
                    detector_id=args.detector_id,
                    source_runtime="python",
                )
            )

    write_detection_jsonl(args.out, records)
    print(f"written {args.out}: {len(records)} detection.v1 records")


def iter_frames(source: str, *, target_fps: float) -> list[Frame]:
    if Path(source).suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to open image source: {source}")
        return [Frame(index=0, pts_s=0.0, image=image)]

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video source: {source}")
    native_fps = max(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 1.0)
    step = 1 if target_fps <= 0 else max(1, round(native_fps / max(target_fps, 0.1)))
    frames: list[Frame] = []
    src_idx = 0
    out_idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if src_idx % step == 0:
            frames.append(Frame(index=out_idx, pts_s=src_idx / native_fps, image=image))
            out_idx += 1
        src_idx += 1
    cap.release()
    return frames


if __name__ == "__main__":
    main()
