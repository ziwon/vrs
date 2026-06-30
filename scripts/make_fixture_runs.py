#!/usr/bin/env python3
"""Create small deterministic VRS run fixtures for local UI/API development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

COLORS = {
    "fire": (220, 54, 46),
    "smoke": (116, 126, 140),
    "falldown": (42, 132, 214),
    "weapon": (36, 38, 41),
}


def alert_record(
    class_name: str,
    severity: str,
    true_alert: bool,
    confidence: float,
    peak_pts_s: float,
    track_id: int | None,
    thumbnail_name: str,
    *,
    stream_id: str | None = None,
    rationale: str | None = None,
    bbox_xywh_norm: list[float] | None = None,
) -> dict[str, Any]:
    record = {
        "class_name": class_name,
        "severity": severity,
        "start_pts_s": round(max(0.0, peak_pts_s - 1.25), 2),
        "peak_pts_s": peak_pts_s,
        "peak_frame_index": int(peak_pts_s * 5),
        "track_id": track_id,
        "peak_detections": [
            {
                "score": round(min(0.99, confidence + 0.04), 2),
                "xyxy": [20.0, 16.0, 132.0, 92.0],
                "raw_label": class_name,
                "track_id": track_id,
            }
        ],
        "num_keyframes": 3,
        "true_alert": true_alert,
        "confidence": confidence,
        "false_negative_class": None,
        "rationale": rationale or f"Fixture {class_name} alert for local dashboard development.",
        "bbox_xywh_norm": bbox_xywh_norm or [0.16, 0.18, 0.5, 0.42],
        "trajectory_xy_norm": [[0.32, 0.35], [0.42, 0.39], [0.5, 0.44]],
        "verifier_raw": json.dumps({"fixture": True, "class_name": class_name}),
        "thumbnail_path": f"thumbnails/{thumbnail_name}",
    }
    if stream_id:
        record["stream_id"] = stream_id
    return record


def write_fixture_run(out_root: Path) -> None:
    fixture = out_root / "fixture"
    records = [
        alert_record("fire", "critical", True, 0.93, 3.2, 101, "fire.png"),
        alert_record("smoke", "high", True, 0.88, 7.4, 102, "smoke.png"),
        alert_record("falldown", "high", False, 0.41, 12.6, 103, "falldown.png"),
        alert_record("weapon", "critical", True, 0.91, 18.0, 104, "weapon.png"),
    ]
    write_run_dir(fixture, records)

    multi = out_root / "fixture_multi"
    write_run_dir(
        multi / "cam-01",
        [
            alert_record("fire", "critical", True, 0.9, 4.0, 201, "fire.png"),
            alert_record("smoke", "high", True, 0.84, 9.5, 202, "smoke.png"),
        ],
    )
    write_run_dir(
        multi / "cam-02",
        [
            alert_record("falldown", "high", False, 0.39, 5.0, 301, "falldown.png"),
            alert_record("weapon", "critical", True, 0.89, 13.0, 302, "weapon.png"),
        ],
    )
    write_local_rtsp_demo(out_root)


def write_local_rtsp_demo(out_root: Path) -> None:
    """Write a small run that matches docker-compose's falldown RTSP sample."""
    demo = out_root / "local-rtsp-demo"
    records = [
        alert_record(
            "falldown",
            "high",
            False,
            0.72,
            1.8,
            None,
            "falldown_sample_01.jpg",
            stream_id="falldown",
            rationale="Demo candidate from the local falldown RTSP sample.",
            bbox_xywh_norm=[0.46, 0.0, 0.1, 1.0],
        ),
        alert_record(
            "falldown",
            "high",
            True,
            0.86,
            4.1,
            None,
            "falldown_sample_02.jpg",
            stream_id="falldown",
            rationale="The local sample shows a persistent vertical subject near the floor line.",
            bbox_xywh_norm=[0.46, 0.0, 0.1, 1.0],
        ),
        alert_record(
            "falldown",
            "high",
            True,
            0.83,
            7.2,
            None,
            "falldown_sample_03.jpg",
            stream_id="falldown",
            rationale="Repeated alert fixture aligned with the RTSP preview clip.",
            bbox_xywh_norm=[0.46, 0.0, 0.1, 1.0],
        ),
    ]
    write_run_dir(demo, records, thumbnail_writer=write_rtsp_demo_thumbnail)


def write_run_dir(run_dir: Path, records: list[dict[str, Any]], thumbnail_writer=None) -> None:
    writer = thumbnail_writer or write_thumbnail
    thumbs = run_dir / "thumbnails"
    thumbs.mkdir(parents=True, exist_ok=True)
    for record in records:
        class_name = str(record["class_name"])
        thumb_name = Path(str(record["thumbnail_path"])).name
        writer(thumbs / thumb_name, class_name)
    with (run_dir / "alerts.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_thumbnail(path: Path, class_name: str) -> None:
    color = COLORS.get(class_name, (80, 90, 100))
    image = Image.new("RGB", (192, 108), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 14, 178, 94), outline=color, width=4)
    draw.rectangle((28, 30, 148, 78), fill=tuple(max(0, channel - 20) for channel in color))
    draw.text((18, 86), class_name, fill=(20, 24, 28))
    image.save(path)


def write_rtsp_demo_thumbnail(path: Path, class_name: str) -> None:
    image = Image.new("RGB", (640, 360), (15, 22, 24))
    draw = ImageDraw.Draw(image)
    for x in range(0, 640, 48):
        shade = 17 + (x // 48) % 3
        draw.rectangle((x, 0, x + 47, 360), fill=(shade, shade + 7, shade + 8))
    draw.line((0, 241, 640, 241), fill=(62, 80, 88), width=2)
    draw.rounded_rectangle((300, -12, 333, 258), radius=20, fill=(180, 198, 210))
    draw.rounded_rectangle((286, -30, 347, 18), radius=24, fill=(188, 208, 220))
    color = COLORS.get(class_name, (42, 132, 214))
    draw.rounded_rectangle((296, 0, 337, 258), radius=22, outline=color, width=3)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs", help="runs root to write fixtures under")
    args = parser.parse_args()
    out_root = Path(args.out)
    write_fixture_run(out_root)
    print(f"Wrote fixture runs under {out_root}")


if __name__ == "__main__":
    main()
