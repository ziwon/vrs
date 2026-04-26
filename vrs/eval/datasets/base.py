"""Dataset protocol — an iterable of ``EvalItem``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..schemas import EvalItem


@runtime_checkable
class Dataset(Protocol):
    """Iterable of labeled videos."""

    def __iter__(self) -> Iterator[EvalItem]: ...
