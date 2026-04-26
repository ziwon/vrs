"""Append-only JSONL writer for VerifiedAlert records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from ..schemas import VerifiedAlert


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: IO[str] | None = None

    def __enter__(self) -> JsonlSink:
        self._fp = self.path.open("a", encoding="utf-8")
        return self

    def write(self, alert: VerifiedAlert) -> None:
        assert self._fp is not None, "JsonlSink used outside of a with-block"
        self._fp.write(json.dumps(alert.to_json(), ensure_ascii=False) + "\n")
        self._fp.flush()

    def __exit__(self, *exc) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
