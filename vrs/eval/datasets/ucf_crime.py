"""UCF-Crime / UCA temporal-annotation dataset adapter.

The UCA annotation release for UCF-Crime uses either TXT rows:

    VideoName StartTime EndTime ##event description

or JSON entries:

    {"VideoName": {"timestamps": [[start, end]], "sentences": [...]}}

This adapter converts those temporal annotations into event-level
``GroundTruthEvent`` records for the existing VRS eval harness.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..schemas import EvalItem, GroundTruthEvent
from .base import Dataset
from .labeled_dir import VIDEO_SUFFIXES

ANNOTATION_CANDIDATES = (
    "annotations.json",
    "annotation.json",
    "uca.json",
    "UCA.json",
    "temporal_annotations.json",
    "Temporal_Anomaly_Annotation.txt",
    "temporal_annotations.txt",
    "annotations.txt",
    "annotation.txt",
    "UCA.txt",
)
VIDEO_DIR_CANDIDATES = ("videos", "Videos", "UCF_Crimes", "UCF-Crime", ".")

UCF_CATEGORY_MAP = {
    "abuse": "abuse",
    "arrest": "arrest",
    "arson": "arson",
    "assault": "assault",
    "burglary": "burglary",
    "explosion": "explosion",
    "fighting": "fighting",
    "normal": "normal",
    "roadaccidents": "road_accident",
    "robbery": "robbery",
    "shooting": "shooting",
    "shoplifting": "shoplifting",
    "stealing": "stealing",
    "vandalism": "vandalism",
}

DESCRIPTION_KEYWORDS = {
    "fire": ("fire", "flame", "burn", "arson"),
    "explosion": ("explosion", "blast"),
    "weapon": ("gun", "knife", "weapon", "shooting"),
    "fighting": ("fight", "assault", "attack", "violence"),
    "road_accident": ("accident", "crash", "collision"),
    "robbery": ("robbery", "steal", "theft", "shoplift", "burglary"),
}


class UCFCrimeDataset(Dataset):
    """Load UCF-Crime/UCA videos and temporal annotations."""

    def __init__(
        self,
        root: str | Path,
        *,
        annotation_file: str | Path | None = None,
        video_dir: str | Path | None = None,
        class_map: Mapping[str, str] | None = None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")
        self.annotation_path = (
            self.root / annotation_file if annotation_file is not None else _find_annotation(self.root)
        )
        self.video_root = self.root / video_dir if video_dir is not None else _find_video_root(self.root)
        self.class_map = {str(k).lower(): str(v) for k, v in (class_map or {}).items()}
        self._videos = _index_videos(self.video_root)
        self._events_by_stem = self._load_annotations()

    def __iter__(self) -> Iterator[EvalItem]:
        yielded: set[Path] = set()
        for stem in sorted(self._events_by_stem):
            video = self._videos.get(stem)
            if video is None:
                raise FileNotFoundError(
                    f"UCF-Crime annotation references {stem!r}, but no matching video was found "
                    f"under {self.video_root}"
                )
            yielded.add(video)
            yield EvalItem(video_path=video, events=self._events_by_stem[stem])

        for video in sorted(set(self._videos.values()) - yielded):
            yield EvalItem(video_path=video, events=[])

    def _load_annotations(self) -> dict[str, list[GroundTruthEvent]]:
        if self.annotation_path.suffix.lower() == ".json":
            return _load_json_annotations(self.annotation_path, self.class_map)
        return _load_txt_annotations(self.annotation_path, self.class_map)


def _find_annotation(root: Path) -> Path:
    for name in ANNOTATION_CANDIDATES:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing UCF-Crime annotation file under {root}; tried {list(ANNOTATION_CANDIDATES)}"
    )


def _find_video_root(root: Path) -> Path:
    for name in VIDEO_DIR_CANDIDATES:
        path = root / name
        if path.is_dir() and any(p.suffix.lower() in VIDEO_SUFFIXES for p in path.rglob("*")):
            return path
    raise FileNotFoundError(f"missing UCF-Crime video files under {root}")


def _index_videos(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            out.setdefault(path.stem, path)
            out.setdefault(path.name, path)
    return out


def _load_txt_annotations(
    path: Path,
    class_map: Mapping[str, str],
) -> dict[str, list[GroundTruthEvent]]:
    out: dict[str, list[GroundTruthEvent]] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        head, _, description = line.partition("##")
        parts = head.split()
        if len(parts) < 3:
            raise ValueError(f"{path}:{line_no}: expected VideoName StartTime EndTime")
        video_name = parts[0]
        start_s, end_s = _parse_interval(parts[1], parts[2], path=path, line_no=line_no)
        class_name = _class_name(video_name, description, class_map)
        out.setdefault(Path(video_name).stem, []).append(
            GroundTruthEvent(class_name=class_name, start_s=start_s, end_s=end_s)
        )
    return out


def _load_json_annotations(
    path: Path,
    class_map: Mapping[str, str],
) -> dict[str, list[GroundTruthEvent]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by video name")
    out: dict[str, list[GroundTruthEvent]] = {}
    for video_name, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: annotation for {video_name!r} must be an object")
        timestamps = payload.get("timestamps") or []
        sentences = payload.get("sentences") or []
        for idx, pair in enumerate(timestamps):
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                raise ValueError(f"{path}: timestamp {idx} for {video_name!r} must be [start, end]")
            description = str(sentences[idx]) if idx < len(sentences) else ""
            start_s, end_s = _parse_interval(pair[0], pair[1], path=path, line_no=idx + 1)
            class_name = _class_name(video_name, description, class_map)
            out.setdefault(Path(str(video_name)).stem, []).append(
                GroundTruthEvent(class_name=class_name, start_s=start_s, end_s=end_s)
            )
    return out


def _parse_interval(start: Any, end: Any, *, path: Path, line_no: int) -> tuple[float, float]:
    try:
        start_s = float(start)
        end_s = float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid start/end time") from exc
    if start_s < 0 or end_s < start_s:
        raise ValueError(f"{path}:{line_no}: expected 0 <= start <= end")
    return start_s, end_s


def _class_name(video_name: str, description: str, class_map: Mapping[str, str]) -> str:
    category = _category_from_video_name(video_name)
    mapped = class_map.get(category)
    if mapped is not None:
        return mapped
    if category and category != "normal":
        return UCF_CATEGORY_MAP.get(category, category)
    desc = description.lower()
    for class_name, keywords in DESCRIPTION_KEYWORDS.items():
        if any(keyword in desc for keyword in keywords):
            return class_map.get(class_name, class_name)
    return class_map.get(category, UCF_CATEGORY_MAP.get(category, category or "anomaly"))


def _category_from_video_name(video_name: str) -> str:
    stem = Path(video_name).stem
    match = re.match(r"([A-Za-z]+)", stem)
    if not match:
        return ""
    raw = match.group(1)
    return raw.lower()
