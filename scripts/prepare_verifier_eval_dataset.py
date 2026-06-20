"""Prepare a small labeled-dir video dataset for verifier bake-offs.

The output matches ``LabeledDirDataset``: each selected video is copied or
symlinked into one directory with a sidecar JSON file. Positive clips get a
coarse full-clip event window; negative clips get an empty event list.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = (".avi", ".m4v", ".mkv", ".mov", ".mp4")


def discover_videos(root: Path, *, limit: int | None = None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    videos = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if limit is not None:
        videos = videos[:limit]
    return videos


def video_duration_s(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0.0 or frames <= 0.0:
        return 0.0
    return frames / fps


def prepare_dataset(
    *,
    out_dir: Path,
    positive_roots: list[Path],
    positive_class: str,
    positive_limit: int | None,
    negative_roots: list[Path],
    negative_limit: int | None,
    copy_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    selected: list[tuple[Path, list[dict[str, float | str]]]] = []

    for root in positive_roots:
        for video in discover_videos(root, limit=positive_limit):
            duration = video_duration_s(video)
            selected.append(
                (
                    video,
                    [
                        {
                            "class": positive_class,
                            "start_s": 0.0,
                            "end_s": round(duration, 3),
                        }
                    ],
                )
            )

    for root in negative_roots:
        for video in discover_videos(root, limit=negative_limit):
            selected.append((video, []))

    for index, (source, events) in enumerate(selected, start=1):
        target = out_dir / f"{index:04d}-{_slug(source.stem)}{source.suffix.lower()}"
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy_mode == "copy":
            shutil.copy2(source, target)
        elif copy_mode == "symlink":
            target.symlink_to(source)
        else:
            raise ValueError(f"unknown copy_mode: {copy_mode}")

        sidecar = target.with_suffix(".json")
        sidecar.write_text(json.dumps({"events": events}, indent=2) + "\n", encoding="utf-8")
        entries.append(
            {
                "source": str(source),
                "video": str(target),
                "sidecar": str(sidecar),
                "events": events,
            }
        )

    manifest = {
        "schema_version": "vrs.verifier_eval_dataset.v1",
        "copy_mode": copy_mode,
        "positive_class": positive_class,
        "count": len(entries),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare labeled-dir verifier eval clips")
    parser.add_argument("--out", type=Path, default=Path("/data/vrs/verifier-eval"))
    parser.add_argument("--positive-root", type=Path, action="append", default=[])
    parser.add_argument("--positive-class", default="fire")
    parser.add_argument("--positive-limit", type=int, default=3)
    parser.add_argument("--negative-root", type=Path, action="append", default=[])
    parser.add_argument("--negative-limit", type=int, default=3)
    parser.add_argument("--copy-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.positive_root and not args.negative_root:
        raise SystemExit("provide at least one --positive-root or --negative-root")
    manifest = prepare_dataset(
        out_dir=args.out,
        positive_roots=args.positive_root,
        positive_class=args.positive_class,
        positive_limit=args.positive_limit,
        negative_roots=args.negative_root,
        negative_limit=args.negative_limit,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(f"Prepared {manifest['count']} clips in {args.out}")
    print(f"Manifest: {args.out / 'manifest.json'}")


def _slug(value: str) -> str:
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else "-")
    return "-".join(part for part in "".join(out).split("-") if part)


if __name__ == "__main__":
    main()
