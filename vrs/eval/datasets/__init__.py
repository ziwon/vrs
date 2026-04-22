"""Dataset adapters for the eval harness.

All adapters yield ``EvalItem`` records. Add new datasets by implementing
the ``Dataset`` protocol in ``base.py``; no other part of the harness needs
to change.
"""
from __future__ import annotations

from .base import Dataset
from .labeled_dir import LabeledDirDataset

__all__ = ["Dataset", "LabeledDirDataset"]
