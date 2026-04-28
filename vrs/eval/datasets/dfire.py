"""D-Fire dataset adapter.

D-Fire is an image dataset with YOLO-format labels. The adapter exposes each
image as one ``EvalItem`` and converts every label row to a zero-duration
``GroundTruthEvent`` with an attached normalized bbox.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from ..schemas import EvalItem, GroundTruthEvent
from .base import Dataset

DEFAULT_CLASS_MAP = {
    0: "smoke",
    1: "fire",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class DFireDataset(Dataset):
    """Load a D-Fire style ``images/`` + ``labels/`` tree.

    Label files use YOLO rows:
    ``<class_id> <center_x> <center_y> <width> <height>``.
    Coordinates must already be normalized to ``0..1``. The adapter converts
    them to VRS ``bbox_xywh_norm`` values: ``x_min, y_min, width, height``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        image_dir: str = "images",
        label_dir: str = "labels",
        class_map: Mapping[int, str] | None = None,
        require_labels: bool = False,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")

        self.image_root = self.root / image_dir
        self.label_root = self.root / label_dir
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"{self.image_root} is not a directory")
        if not self.label_root.is_dir():
            raise FileNotFoundError(f"{self.label_root} is not a directory")

        self.class_map = dict(class_map or DEFAULT_CLASS_MAP)
        self.require_labels = require_labels

    def __iter__(self) -> Iterator[EvalItem]:
        for image in _iter_images(self.image_root):
            yield EvalItem(video_path=image, events=self._load_labels(image))

    def _load_labels(self, image: Path) -> list[GroundTruthEvent]:
        label = self.label_root / f"{image.stem}.txt"
        if not label.exists():
            if self.require_labels:
                raise FileNotFoundError(f"missing D-Fire label file: {label}")
            return []

        events: list[GroundTruthEvent] = []
        for line_no, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            events.append(
                _parse_yolo_label(line, label=label, line_no=line_no, class_map=self.class_map)
            )
        return events


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _parse_yolo_label(
    line: str,
    *,
    label: Path,
    line_no: int,
    class_map: Mapping[int, str],
) -> GroundTruthEvent:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{label}:{line_no}: expected 5 YOLO fields, got {len(parts)}")

    try:
        class_id = int(parts[0])
        bbox = tuple(float(value) for value in parts[1:5])
    except ValueError as exc:
        raise ValueError(f"{label}:{line_no}: invalid YOLO label row: {line!r}") from exc

    if class_id not in class_map:
        raise ValueError(f"{label}:{line_no}: unknown D-Fire class id {class_id}")
    if len(bbox) != 4 or not _valid_yolo_bbox(bbox):
        raise ValueError(f"{label}:{line_no}: bbox values must be normalized YOLO coordinates")

    return GroundTruthEvent(
        class_name=class_map[class_id],
        start_s=0.0,
        end_s=0.0,
        bbox_xywh_norm=_yolo_to_vrs_bbox(bbox),
    )


def _valid_yolo_bbox(bbox: tuple[float, ...]) -> bool:
    cx, cy, width, height = bbox
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0


def _yolo_to_vrs_bbox(bbox: tuple[float, ...]) -> tuple[float, float, float, float]:
    cx, cy, width, height = bbox
    return (cx - width / 2, cy - height / 2, width, height)
