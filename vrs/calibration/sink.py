"""Append-only JSONL writer for calibration suggestions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Optional

from .schemas import Suggestion

logger = logging.getLogger(__name__)


class CalibrationSink:
    """Writes one line of JSON per suggestion.

    Opened lazily on first write so a calibrator that never suggests doesn't
    leave an empty file behind. Safe to call in a worker thread — the
    underlying ``open`` + ``write`` is only hit from the single consumer
    (verifier worker) that owns the Calibrator.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fp: Optional[IO[str]] = None

    def _open(self) -> None:
        if self._fp is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, s: Suggestion) -> None:
        self._open()
        assert self._fp is not None
        self._fp.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        self._fp.flush()
        logger.info(
            "calibration suggestion: stream=%s class=%s %s %.2f → %.2f  (flip_rate=%.2f, n=%d)",
            s.stream_id, s.class_name, s.direction,
            s.current_min_score, s.suggested_min_score,
            s.flip_rate, s.n_alerts,
        )

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "CalibrationSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
