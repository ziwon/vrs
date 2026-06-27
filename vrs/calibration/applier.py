"""Controlled Stage-B application of calibration suggestions."""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import IO, Any

import yaml

from ..policy import WatchPolicy
from .schemas import Suggestion


class CalibrationApplier:
    """Apply suggestions to an exported per-stream/class threshold snapshot.

    The running detector policy remains immutable. Stage B records applied
    thresholds and exports them for operator review or the next process start.
    """

    def __init__(
        self,
        policy: WatchPolicy,
        out_dir: str | Path,
        *,
        cooldown_s: float = 3600.0,
        min_score_cap: float = 0.15,
        max_score_cap: float = 0.80,
        audit_filename: str = "calibration_applied.jsonl",
        export_filename: str = "calibration_overrides.yaml",
    ):
        self.policy = policy
        self.out_dir = Path(out_dir)
        self.cooldown_s = float(cooldown_s)
        if self.cooldown_s < 0:
            raise ValueError("calibration.apply_cooldown_s must be >= 0")
        self.min_score_cap = float(min_score_cap)
        self.max_score_cap = float(max_score_cap)
        if self.min_score_cap > self.max_score_cap:
            raise ValueError("calibration min_score_cap must be <= max_score_cap")
        self.audit_path = self.out_dir / audit_filename
        self.export_path = self.out_dir / export_filename
        self._audit_fp: IO[str] | None = None
        self._scores: dict[tuple[str, str], float] = {}
        self._last_apply_monotonic: dict[tuple[str, str], float] = {}

    def current_min_score(self, stream_id: str, class_name: str, fallback: float) -> float:
        return self._scores.get((stream_id, class_name), float(fallback))

    def apply(self, suggestion: Suggestion, *, now_monotonic: float | None = None) -> dict | None:
        key = (suggestion.stream_id, suggestion.class_name)
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        last = self._last_apply_monotonic.get(key)
        if last is not None and (now - last) < self.cooldown_s:
            return None

        old_score = self.current_min_score(
            suggestion.stream_id,
            suggestion.class_name,
            suggestion.current_min_score,
        )
        new_score = max(self.min_score_cap, min(self.max_score_cap, suggestion.suggested_min_score))
        if new_score == old_score:
            return None

        self._scores[key] = float(new_score)
        self._last_apply_monotonic[key] = now
        record = self._record(suggestion, old_score=old_score, new_score=new_score)
        self._write_audit(record)
        self._write_export(record["ts"])
        return record

    def close(self) -> None:
        if self._audit_fp is not None:
            self._audit_fp.close()
            self._audit_fp = None

    def _record(self, suggestion: Suggestion, *, old_score: float, new_score: float) -> dict:
        return {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "stream_id": suggestion.stream_id,
            "class_name": suggestion.class_name,
            "old_min_score": float(old_score),
            "new_min_score": float(new_score),
            "delta": float(new_score - old_score),
            "direction": suggestion.direction,
            "reason": suggestion.reason,
            "flip_rate": float(suggestion.flip_rate),
            "fn_flag_rate": float(suggestion.fn_flag_rate),
            "n_alerts": int(suggestion.n_alerts),
            "alerts_per_hour": (
                float(suggestion.alerts_per_hour)
                if suggestion.alerts_per_hour is not None
                else None
            ),
            "cooldown_s": float(self.cooldown_s),
            "export_path": str(self.export_path),
        }

    def _write_audit(self, record: dict) -> None:
        if self._audit_fp is None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_fp = self.audit_path.open("a", encoding="utf-8")
        self._audit_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._audit_fp.flush()

    def _write_export(self, generated_at: str) -> None:
        payload: dict[str, Any] = {
            "calibration_overrides": {
                "generated_at": generated_at,
                "note": (
                    "Operator-reviewable Stage-B threshold export. Apply by "
                    "copying selected min_score values into the watch policy, "
                    "or keep this file as a rollback/reference snapshot."
                ),
                "thresholds": {},
            }
        }
        thresholds = payload["calibration_overrides"]["thresholds"]
        for (stream_id, class_name), score in sorted(self._scores.items()):
            thresholds.setdefault(stream_id, {})[class_name] = {
                "min_score": float(score),
                "source": "calibration_stage_b",
            }

        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.export_path.with_suffix(self.export_path.suffix + ".tmp")
        tmp_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.export_path)
