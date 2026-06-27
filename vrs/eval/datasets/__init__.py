"""Dataset adapters for the eval harness.

All adapters yield ``EvalItem`` records. Add new datasets by implementing
the ``Dataset`` protocol in ``base.py``; no other part of the harness needs
to change.
"""

from __future__ import annotations

from pathlib import Path

from .base import Dataset
from .dfire import DFireDataset
from .labeled_dir import LabeledDirDataset
from .ucf_crime import UCFCrimeDataset

DATASET_ADAPTERS = {
    "labeled_dir": LabeledDirDataset,
    "dfire": DFireDataset,
    "ucf_crime": UCFCrimeDataset,
}


def build_dataset(adapter: str, root: str | Path) -> Dataset:
    try:
        dataset_cls = DATASET_ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset adapter: {adapter!r}. Valid: {sorted(DATASET_ADAPTERS)}"
        ) from exc
    return dataset_cls(root)


__all__ = [
    "DATASET_ADAPTERS",
    "DFireDataset",
    "Dataset",
    "LabeledDirDataset",
    "UCFCrimeDataset",
    "build_dataset",
]
