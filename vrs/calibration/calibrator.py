"""Stateful wrapper around ``suggest()`` — maintains per-key rolling windows
and gates emission so one sustained regime produces one suggestion (not
one-per-alert).

After each emission the window is cleared. Rationale: the operator is
meant to review the suggestion and either apply it (which changes
``current_min_score`` out-of-band) or ignore it. Either way, the next
meaningful signal comes from a fresh ``min_sample`` worth of verdicts
under whatever policy is current — so we start a new sample.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from ..policy import WatchPolicy
from ..schemas import VerifiedAlert
from .schemas import Suggestion, WindowEntry
from .sink import CalibrationSink
from .suggester import suggest

logger = logging.getLogger(__name__)


class Calibrator:
    """Accumulates verifier verdicts per ``(stream_id, class)`` and emits
    log-only suggestions via a ``CalibrationSink``.
    """

    def __init__(
        self,
        policy: WatchPolicy,
        sink: CalibrationSink,
        *,
        window_size: int = 100,
        min_sample: int = 10,
        max_flip_rate: float = 0.30,
        min_flip_rate: float = 0.05,
        score_delta: float = 0.02,
        min_score_cap: float = 0.15,
        max_score_cap: float = 0.80,
        target_alerts_per_hour: Optional[float] = None,
    ):
        self.policy = policy
        self.sink = sink
        self.window_size = int(window_size)
        self.min_sample = int(min_sample)
        self.max_flip_rate = float(max_flip_rate)
        self.min_flip_rate = float(min_flip_rate)
        self.score_delta = float(score_delta)
        self.min_score_cap = float(min_score_cap)
        self.max_score_cap = float(max_score_cap)
        self.target_alerts_per_hour = (
            float(target_alerts_per_hour) if target_alerts_per_hour is not None else None
        )

        self._windows: Dict[Tuple[str, str], Deque[WindowEntry]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    # ---- public api -----------------------------------------------

    def record(self, stream_id: str, verified: VerifiedAlert) -> Optional[Suggestion]:
        """Log one verdict. Returns the emitted suggestion if any."""
        cls = verified.candidate.class_name
        item = self.policy.get(cls)
        if item is None:
            # An FN-flagged alert can arrive for a class the current policy
            # still lists; an alert for a class no longer in the policy is
            # a noop for calibration — nothing to tune.
            return None

        key = (stream_id, cls)
        self._windows[key].append(WindowEntry(
            ts_monotonic=time.monotonic(),
            was_flipped=not verified.true_alert,
            had_fn_flag=verified.false_negative_class is not None,
        ))

        suggestion = suggest(
            stream_id=stream_id,
            class_name=cls,
            current_min_score=float(item.min_score),
            window=self._windows[key],
            max_flip_rate=self.max_flip_rate,
            min_flip_rate=self.min_flip_rate,
            min_sample=self.min_sample,
            score_delta=self.score_delta,
            min_score_cap=self.min_score_cap,
            max_score_cap=self.max_score_cap,
            target_alerts_per_hour=self.target_alerts_per_hour,
        )
        if suggestion is None:
            return None

        self.sink.write(suggestion)
        # fresh sample after each emission — see module docstring
        self._windows[key].clear()
        return suggestion

    def close(self) -> None:
        self.sink.close()


# ──────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────

def build_calibrator(
    cfg: Optional[dict],
    policy: WatchPolicy,
    out_dir: str | Path,
    *,
    filename: str = "calibration_suggestions.jsonl",
) -> Optional[Calibrator]:
    """Construct a Calibrator from a YAML block, or ``None`` when disabled.

    Accepted shape::

        calibration:
          enabled: true
          window_size: 100
          min_sample: 10
          max_flip_rate: 0.30
          min_flip_rate: 0.05
          score_delta: 0.02
          min_score_cap: 0.15
          max_score_cap: 0.80
          target_alerts_per_hour: null   # loosen-arm gate; null = tighten only
    """
    if not cfg or not cfg.get("enabled", False):
        return None

    out = Path(out_dir)
    sink = CalibrationSink(out / filename)
    return Calibrator(
        policy=policy,
        sink=sink,
        window_size=int(cfg.get("window_size", 100)),
        min_sample=int(cfg.get("min_sample", 10)),
        max_flip_rate=float(cfg.get("max_flip_rate", 0.30)),
        min_flip_rate=float(cfg.get("min_flip_rate", 0.05)),
        score_delta=float(cfg.get("score_delta", 0.02)),
        min_score_cap=float(cfg.get("min_score_cap", 0.15)),
        max_score_cap=float(cfg.get("max_score_cap", 0.80)),
        target_alerts_per_hour=cfg.get("target_alerts_per_hour"),
    )
