"""Dataset protocol — an iterable of ``EvalItem``."""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from ..schemas import EvalItem


@runtime_checkable
class Dataset(Protocol):
    """Iterable of labeled videos."""
    def __iter__(self) -> Iterator[EvalItem]: ...
