"""Stateless decision function: rolling window → optional suggestion.

Pure function so it's trivial to unit-test. The stateful ``Calibrator``
wraps it with bookkeeping (window, last-emitted, JSONL sink), but every
policy decision lives here.

Decision rules, directly from the improvements doc:

    if flip_rate > max_flip_rate:
        min_score += score_delta    # tighten — too many false alarms
    elif flip_rate < min_flip_rate and alerts_per_hour < target_alerts_per_hour:
        min_score -= score_delta    # loosen — we may be missing cases

The loosening arm is gated on an explicit ``target_alerts_per_hour``
passed by the operator. Without a target it can't tell "quiet site, right
thresholds" apart from "thresholds too high, missing cases", so it stays
silent — tightening on obvious over-firing is the only fully-safe
default.
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable, Optional

from .schemas import Suggestion, WindowEntry


def _iso_ts() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _alerts_per_hour(window: list[WindowEntry]) -> Optional[float]:
    if len(window) < 2:
        return None
    elapsed = window[-1].ts_monotonic - window[0].ts_monotonic
    if elapsed <= 0:
        return None
    return len(window) / (elapsed / 3600.0)


def suggest(
    stream_id: str,
    class_name: str,
    current_min_score: float,
    window: Iterable[WindowEntry],
    *,
    max_flip_rate: float = 0.30,
    min_flip_rate: float = 0.05,
    min_sample: int = 10,
    score_delta: float = 0.02,
    min_score_cap: float = 0.15,
    max_score_cap: float = 0.80,
    target_alerts_per_hour: Optional[float] = None,
    ts: Optional[str] = None,
) -> Optional[Suggestion]:
    """Return a Suggestion if the window warrants one, else ``None``."""
    if not 0.0 <= min_flip_rate <= max_flip_rate <= 1.0:
        raise ValueError("must satisfy 0 <= min_flip_rate <= max_flip_rate <= 1")
    if score_delta <= 0:
        raise ValueError("score_delta must be > 0")
    if min_score_cap > max_score_cap:
        raise ValueError("min_score_cap must be <= max_score_cap")

    entries = list(window)
    n = len(entries)
    if n < min_sample:
        return None

    n_flipped = sum(1 for e in entries if e.was_flipped)
    n_fn = sum(1 for e in entries if e.had_fn_flag)
    flip_rate = n_flipped / n
    fn_flag_rate = n_fn / n
    rate = _alerts_per_hour(entries)

    if flip_rate > max_flip_rate:
        new_score = min(current_min_score + score_delta, max_score_cap)
        if new_score <= current_min_score:
            return None   # already at the ceiling — nothing to suggest
        reason = (
            f"flip_rate={flip_rate:.2f} > {max_flip_rate:.2f} over {n} alerts "
            f"— verifier is overruling the detector too often; tightening "
            f"min_score reduces false alarms at the cost of recall."
        )
        return Suggestion(
            ts=ts or _iso_ts(),
            stream_id=stream_id,
            class_name=class_name,
            current_min_score=float(current_min_score),
            suggested_min_score=float(new_score),
            direction="tighten",
            reason=reason,
            flip_rate=flip_rate,
            fn_flag_rate=fn_flag_rate,
            n_alerts=n,
            alerts_per_hour=rate,
        )

    if (
        flip_rate < min_flip_rate
        and target_alerts_per_hour is not None
        and rate is not None
        and rate < target_alerts_per_hour
    ):
        new_score = max(current_min_score - score_delta, min_score_cap)
        if new_score >= current_min_score:
            return None   # already at the floor
        reason = (
            f"flip_rate={flip_rate:.2f} < {min_flip_rate:.2f} and "
            f"alerts_per_hour={rate:.2f} < target={target_alerts_per_hour:.2f} "
            f"— verifier is rubber-stamping and we're below the alert-rate "
            f"target; loosening min_score to surface more candidates."
        )
        return Suggestion(
            ts=ts or _iso_ts(),
            stream_id=stream_id,
            class_name=class_name,
            current_min_score=float(current_min_score),
            suggested_min_score=float(new_score),
            direction="loosen",
            reason=reason,
            flip_rate=flip_rate,
            fn_flag_rate=fn_flag_rate,
            n_alerts=n,
            alerts_per_hour=rate,
        )

    return None
