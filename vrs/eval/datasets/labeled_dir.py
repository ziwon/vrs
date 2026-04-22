"""Adapter: directory of ``*.mp4`` files with sidecar ``*.json`` labels.

Layout expected::

    root/
      cam01_fire.mp4
      cam01_fire.json
      cam02_smoke.mp4
      cam02_smoke.json

Each sidecar JSON::

    {
      "events": [
        {"class": "fire",     "start_s": 2.5, "end_s": 8.0},
        {"class": "falldown", "start_s": 12.0, "end_s": 14.5}
      ]
    }

Videos without a sidecar are still yielded (with an empty events list) so
they can score a detector's false-positive rate on "quiet" footage.

This is the generic case we can rely on locally in tests and in the lab.
Public-dataset adapters (D-Fire, Le2i) convert their native formats into
the same on-the-fly shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from ..schemas import EvalItem, GroundTruthEvent
from .base import Dataset


class LabeledDirDataset(Dataset):
    def __init__(self, root: str | Path, *, video_glob: str = "*.mp4"):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")
        self.video_glob = video_glob

    def __iter__(self) -> Iterator[EvalItem]:
        for video in sorted(self.root.glob(self.video_glob)):
            yield EvalItem(video_path=video, events=_load_sidecar(video))


def _load_sidecar(video: Path) -> List[GroundTruthEvent]:
    label = video.with_suffix(".json")
    if not label.exists():
        return []
    raw = json.loads(label.read_text(encoding="utf-8"))
    out: List[GroundTruthEvent] = []
    for e in raw.get("events", []):
        out.append(GroundTruthEvent(
            class_name=str(e["class"]),
            start_s=float(e["start_s"]),
            end_s=float(e["end_s"]),
        ))
    return out
