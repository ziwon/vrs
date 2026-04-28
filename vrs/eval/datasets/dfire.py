"""Adapter for the D-Fire image dataset.

D-Fire labels are YOLO text files: one object per line as
``<class_id> <x_center> <y_center> <width> <height>``, with all coordinates
normalized to ``0..1``. The common D-Fire YOLO mapping is ``0=smoke`` and
``1=fire``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from PIL import Image

from ..schemas import EvalItem, GroundTruthEvent
from .base import Dataset

DEFAULT_DFIRE_CLASSES = ("smoke", "fire")
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DFireDataset(Dataset):
    """Iterate a local D-Fire-style ``images/`` + ``labels/`` directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        images_dir: str | Path = "images",
        labels_dir: str | Path = "labels",
        class_names: Sequence[str] = DEFAULT_DFIRE_CLASSES,
        image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")

        self.images_dir = self.root / images_dir
        self.labels_dir = self.root / labels_dir
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"{self.images_dir} is not a directory")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"{self.labels_dir} is not a directory")

        self.class_names = tuple(str(name) for name in class_names)
        if not self.class_names:
            raise ValueError("class_names must not be empty")
        self.image_extensions = {ext.lower() for ext in image_extensions}

    def __iter__(self) -> Iterator[EvalItem]:
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in self.image_extensions:
                continue
            with Image.open(image_path) as img:
                image_size = img.size
            yield EvalItem(
                video_path=image_path,
                events=_load_yolo_label(
                    self.labels_dir / f"{image_path.stem}.txt",
                    class_names=self.class_names,
                ),
                image_size=image_size,
            )


def _load_yolo_label(label_path: Path, *, class_names: Sequence[str]) -> list[GroundTruthEvent]:
    if not label_path.exists():
        return []

    events: list[GroundTruthEvent] = []
    for lineno, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{lineno}: expected 5 YOLO fields, got {len(parts)}")

        class_id = _parse_class_id(parts[0], label_path=label_path, lineno=lineno)
        if class_id >= len(class_names):
            raise ValueError(
                f"{label_path}:{lineno}: class id {class_id} has no configured class name"
            )

        bbox = tuple(float(v) for v in parts[1:5])
        _validate_bbox(bbox, label_path=label_path, lineno=lineno)
        events.append(
            GroundTruthEvent(
                class_name=class_names[class_id],
                start_s=0.0,
                end_s=0.0,
                bbox_xywh_norm=bbox,
            )
        )
    return events


def _parse_class_id(raw: str, *, label_path: Path, lineno: int) -> int:
    try:
        class_id = int(raw)
    except ValueError as e:
        raise ValueError(f"{label_path}:{lineno}: class id must be an integer") from e
    if class_id < 0:
        raise ValueError(f"{label_path}:{lineno}: class id must be non-negative")
    return class_id


def _validate_bbox(
    bbox: tuple[float, float, float, float], *, label_path: Path, lineno: int
) -> None:
    x, y, w, h = bbox
    if not all(0.0 <= v <= 1.0 for v in bbox):
        raise ValueError(f"{label_path}:{lineno}: YOLO bbox values must be between 0 and 1")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"{label_path}:{lineno}: YOLO bbox width/height must be positive")
    if x - w / 2.0 < 0.0 or x + w / 2.0 > 1.0 or y - h / 2.0 < 0.0 or y + h / 2.0 > 1.0:
        raise ValueError(f"{label_path}:{lineno}: YOLO bbox must fit inside the image")
