"""Le2i Fall Detection Dataset adapter.

Le2i annotations are text files paired with RGB videos. The ground-truth file
for a fall video starts with the frame number where the fall begins and the
frame number where the fall ends; subsequent rows contain per-frame bbox
metadata. Non-fall videos may have no annotation file or no fall frame range.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..schemas import EvalItem, GroundTruthEvent
from .base import Dataset
from .labeled_dir import VIDEO_SUFFIXES

ANNOTATION_DIR_CANDIDATES = (
    "Annotation_files",
    "annotation_files",
    "annotations",
    "Annotations",
    "groundtruth",
    "GroundTruth",
    ".",
)
VIDEO_DIR_CANDIDATES = (
    "Videos",
    "videos",
    "Video_files",
    "video_files",
    ".",
)


class Le2iDataset(Dataset):
    """Load Le2i fall videos and frame-range annotations."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: float = 25.0,
        class_name: str = "falldown",
        video_dir: str | Path | None = None,
        annotation_dir: str | Path | None = None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("Le2i fps must be > 0")
        self.class_name = str(class_name)
        self.video_root = self.root / video_dir if video_dir is not None else _find_dir_with_videos(self.root)
        self.annotation_root = (
            self.root / annotation_dir
            if annotation_dir is not None
            else _find_annotation_dir(self.root)
        )

    def __iter__(self) -> Iterator[EvalItem]:
        for video in _iter_videos(self.video_root):
            annotation = _find_annotation_file(self.annotation_root, video)
            events = _load_fall_event(
                annotation,
                fps=self.fps,
                class_name=self.class_name,
            ) if annotation is not None else []
            yield EvalItem(video_path=video, events=events)


def _find_dir_with_videos(root: Path) -> Path:
    for name in VIDEO_DIR_CANDIDATES:
        path = root / name
        if path.is_dir() and any(p.suffix.lower() in VIDEO_SUFFIXES for p in path.rglob("*")):
            return path
    raise FileNotFoundError(f"missing Le2i video files under {root}")


def _find_annotation_dir(root: Path) -> Path:
    for name in ANNOTATION_DIR_CANDIDATES:
        path = root / name
        if path.is_dir() and any(p.suffix.lower() == ".txt" for p in path.rglob("*")):
            return path
    return root


def _iter_videos(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            yield path


def _find_annotation_file(root: Path, video: Path) -> Path | None:
    candidates = [
        root / f"{video.stem}.txt",
        root / f"{video.name}.txt",
        root / f"{video.stem.replace('_', ' ')}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    normalized = _normalize_name(video.stem)
    for path in root.rglob("*.txt"):
        if _normalize_name(path.stem) == normalized:
            return path
    return None


def _load_fall_event(path: Path, *, fps: float, class_name: str) -> list[GroundTruthEvent]:
    values: list[int] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        value = _single_int_line(line)
        if value is None:
            break
        values.append(value)
        if len(values) == 2:
            break

    if len(values) < 2:
        return []
    start_frame, end_frame = values
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"{path}: expected 0 <= fall_start_frame <= fall_end_frame")
    return [
        GroundTruthEvent(
            class_name=class_name,
            start_s=start_frame / fps,
            end_s=end_frame / fps,
        )
    ]


def _single_int_line(line: str) -> int | None:
    if not re.fullmatch(r"[+-]?\d+", line):
        return None
    return int(line)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
