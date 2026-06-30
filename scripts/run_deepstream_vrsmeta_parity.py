"""Run Python-vs-DeepStream vrsmeta parity for one video source."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: str
    frames: int | None
    duration_s: float | None


def main() -> None:
    args = build_arg_parser().parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_id = args.stream_id or source.stem
    video = probe_video(source)
    python_jsonl = Path(args.python_detections or out_dir / f"python_{stream_id}.jsonl")
    candidate_jsonl = Path(args.candidate_detections or out_dir / f"deepstream_{stream_id}.jsonl")
    strict_report = out_dir / f"parity_{stream_id}_strict.json"
    loose_report = out_dir / f"parity_{stream_id}_loose.json"

    if not args.skip_python:
        run(
            python_export_command(
                source=source,
                out=python_jsonl,
                stream_id=stream_id,
                config=args.config,
                policy=args.policy,
                detector_id=args.python_detector_id,
            )
        )

    if not args.skip_deepstream:
        run(
            deepstream_docker_command(
                source=source,
                out=candidate_jsonl,
                stream_id=stream_id,
                image=args.image,
                engine_dir=Path(args.engine_dir),
                mux_width=args.mux_width,
                mux_height=args.mux_height,
                pgie_config=args.pgie_config,
                labels=args.labels,
                detector_id=args.deepstream_detector_id,
            )
        )

    run(
        compare_command(
            python_jsonl=python_jsonl,
            candidate_jsonl=candidate_jsonl,
            out=strict_report,
            policy=args.policy,
            video=video,
            mux_width=args.mux_width,
            mux_height=args.mux_height,
            time_tolerance_s=args.strict_time_tolerance_s,
            iou_threshold=args.iou_threshold,
        )
    )
    run(
        compare_command(
            python_jsonl=python_jsonl,
            candidate_jsonl=candidate_jsonl,
            out=loose_report,
            policy=args.policy,
            video=video,
            mux_width=args.mux_width,
            mux_height=args.mux_height,
            time_tolerance_s=args.loose_time_tolerance_s,
            iou_threshold=args.iou_threshold,
        )
    )

    summary = {
        "schema_version": "vrs.eval.deepstream_vrsmeta_parity_run.v1",
        "source": str(source),
        "video": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frames": video.frames,
            "duration_s": video.duration_s,
        },
        "python_detections": str(python_jsonl),
        "candidate_detections": str(candidate_jsonl),
        "strict_report": str(strict_report),
        "loose_report": str(loose_report),
        "strict": load_totals(strict_report),
        "loose": load_totals(loose_report),
    }
    summary_path = out_dir / f"summary_{stream_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"written {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run Python YOLOE, DS8 vrsmeta, and aligned parity comparison"
    )
    ap.add_argument("--source", required=True, help="input video path on the host")
    ap.add_argument("--out-dir", default="runs/parity/deepstream-vrsmeta")
    ap.add_argument("--image", default="vrs-deepstream:ds8")
    ap.add_argument("--engine-dir", default="runs/engines")
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument(
        "--pgie-config",
        default="/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt",
    )
    ap.add_argument(
        "--labels",
        default="/opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt",
    )
    ap.add_argument("--mux-width", type=int, default=640)
    ap.add_argument("--mux-height", type=int, default=640)
    ap.add_argument("--stream-id")
    ap.add_argument("--python-detector-id", default="python-yoloe")
    ap.add_argument("--deepstream-detector-id", default="ds8-yoloe-vrsmeta")
    ap.add_argument("--strict-time-tolerance-s", type=float, default=0.08)
    ap.add_argument("--loose-time-tolerance-s", type=float, default=1.0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--python-detections")
    ap.add_argument("--candidate-detections")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--skip-deepstream", action="store_true")
    return ap


def probe_video(path: Path) -> VideoInfo:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream found: {path}")
    stream = streams[0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""),
        frames=_optional_int(stream.get("nb_frames")),
        duration_s=_optional_float(stream.get("duration")),
    )


def python_export_command(
    *,
    source: Path,
    out: Path,
    stream_id: str,
    config: str,
    policy: str,
    detector_id: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/export_python_detections.py",
        "--config",
        config,
        "--policy",
        policy,
        "--out",
        str(out),
        "--stream-id",
        stream_id,
        "--detector-id",
        detector_id,
        str(source),
    ]


def deepstream_docker_command(
    *,
    source: Path,
    out: Path,
    stream_id: str,
    image: str,
    engine_dir: Path,
    mux_width: int,
    mux_height: int,
    pgie_config: str,
    labels: str,
    detector_id: str,
) -> list[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    source_parent = source.resolve().parent
    source_in_container = f"/clips/{source.name}"
    out_in_container = f"/out/{out.name}"
    pipeline = deepstream_pipeline(
        source=source_in_container,
        output=out_in_container,
        stream_id=stream_id,
        detector_id=detector_id,
        mux_width=mux_width,
        mux_height=mux_height,
        pgie_config=pgie_config,
        labels=labels,
    )
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{engine_dir.resolve()}:/models:ro",
        "-v",
        f"{source_parent}:/clips:ro",
        "-v",
        f"{out.parent.resolve()}:/out",
        image,
        "--pipeline",
        pipeline,
        "--disable-probe",
    ]


def deepstream_pipeline(
    *,
    source: str,
    output: str,
    stream_id: str,
    detector_id: str,
    mux_width: int,
    mux_height: int,
    pgie_config: str,
    labels: str,
) -> str:
    source_chain = source_decode_chain(source)
    return (
        f"{source_chain} ! m.sink_0 nvstreammux name=m batch-size=1 "
        f"width={mux_width} height={mux_height} enable-padding=1 live-source=0 "
        f"! nvinfer config-file-path={pgie_config} "
        "! nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        f"! vrsmeta stream-id={stream_id} detector-id={detector_id} labels={labels} "
        f"output-mode=jsonl output-path={output} "
        "! fakesink name=sink sync=false"
    )


def source_decode_chain(source: str) -> str:
    suffix = Path(source).suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return f"filesrc location={source} ! qtdemux ! h264parse ! nvv4l2decoder"
    if suffix == ".avi":
        return (
            f"filesrc location={source} ! avidemux ! decodebin ! videoconvert "
            "! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12"
        )
    raise ValueError(f"unsupported source suffix for default pipeline: {suffix}")


def compare_command(
    *,
    python_jsonl: Path,
    candidate_jsonl: Path,
    out: Path,
    policy: str,
    video: VideoInfo,
    mux_width: int,
    mux_height: int,
    time_tolerance_s: float,
    iou_threshold: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/compare_aligned_detector_parity.py",
        "--python-detections",
        str(python_jsonl),
        "--candidate-detections",
        str(candidate_jsonl),
        "--policy",
        policy,
        "--time-tolerance-s",
        str(time_tolerance_s),
        "--iou-threshold",
        str(iou_threshold),
        "--auto-candidate-letterbox",
        "--source-width",
        str(video.width),
        "--source-height",
        str(video.height),
        "--candidate-mux-width",
        str(mux_width),
        "--candidate-mux-height",
        str(mux_height),
        "--out",
        str(out),
    ]


def load_totals(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "totals": payload["totals"],
        "bbox": payload["bbox"],
        "time_delta_s": payload["time_delta_s"],
        "confidence_delta": payload["confidence_delta"],
    }


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _optional_int(value: object) -> int | None:
    if value in (None, "N/A"):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value in (None, "N/A"):
        return None
    return float(value)


if __name__ == "__main__":
    main()
