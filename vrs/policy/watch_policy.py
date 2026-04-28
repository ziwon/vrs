"""Watch Policy — the only file an operator maintains.

A watch policy lists named events in plain English. Each item has:

  detector  : noun-phrase prompts that YOLOE will turn into open-vocab classes.
              Several phrases per event widen recall (e.g. "fire", "open flame").
  verifier  : a one-sentence definition handed to the VLM verifier.
  severity  : info | low | medium | high | critical
  min_score : per-class score floor (overrides config.detector.conf_floor up).
  min_persist_frames : temporal persistence required to raise a CandidateAlert.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_SEVERITY = {"info", "low", "medium", "high", "critical"}


@dataclass(frozen=True)
class WatchItem:
    name: str
    detector_prompts: list[str]
    verifier_prompt: str
    severity: str
    min_score: float
    min_persist_frames: int


class WatchPolicy:
    """Indexed view over the operator's watch list."""

    def __init__(self, items: Sequence[WatchItem]):
        if not items:
            raise ValueError("WatchPolicy must contain at least one item")
        self._by_name: dict[str, WatchItem] = {it.name: it for it in items}

        # YOLOE consumes a flat vocabulary; remember which prompt → which event
        self._flat_prompts: list[str] = []
        self._prompt_to_event: list[str] = []
        for it in items:
            for p in it.detector_prompts:
                self._flat_prompts.append(p)
                self._prompt_to_event.append(it.name)

    # ---- lookup -----------------------------------------------------

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> WatchItem:
        return self._by_name[name]

    def get(self, name: str) -> WatchItem | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    # ---- detector glue ---------------------------------------------

    def yoloe_vocabulary(self) -> list[str]:
        """Flat list of noun phrases to push into YOLOE's class names."""
        return list(self._flat_prompts)

    def event_for_prompt_index(self, idx: int) -> str:
        """Map a YOLOE prediction back to the event name."""
        return self._prompt_to_event[idx]

    # ---- verifier glue ---------------------------------------------

    def verifier_definitions(self) -> dict[str, str]:
        return {it.name: it.verifier_prompt for it in self._by_name.values()}


def _validate_item(raw: dict) -> WatchItem:
    name = str(raw["name"]).strip()
    detector = raw.get("detector") or [name]
    if isinstance(detector, str):
        detector = [detector]
    detector = [str(p).strip() for p in detector if str(p).strip()]
    if not detector:
        raise ValueError(f"watch[{name}].detector is empty")

    verifier = str(raw.get("verifier", "")).strip()
    if not verifier:
        raise ValueError(f"watch[{name}].verifier must be a one-sentence definition")

    severity = str(raw.get("severity", "medium")).lower().strip()
    if severity not in _VALID_SEVERITY:
        raise ValueError(f"watch[{name}].severity={severity!r} not in {_VALID_SEVERITY}")

    min_score = float(raw.get("min_score", 0.30))
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"watch[{name}].min_score must be in [0,1]")

    min_persist = int(raw.get("min_persist_frames", 2))
    if min_persist < 1:
        raise ValueError(f"watch[{name}].min_persist_frames must be >= 1")

    return WatchItem(
        name=name,
        detector_prompts=detector,
        verifier_prompt=verifier,
        severity=severity,
        min_score=min_score,
        min_persist_frames=min_persist,
    )


def load_watch_policy(path: str | Path) -> WatchPolicy:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    items_raw = raw.get("watch") or []
    if not items_raw:
        raise ValueError(f"{path}: 'watch' key is empty or missing")
    items = [_validate_item(it) for it in items_raw]

    # uniqueness check on names
    seen = set()
    for it in items:
        if it.name in seen:
            raise ValueError(f"duplicate watch entry: {it.name!r}")
        seen.add(it.name)

    return WatchPolicy(items)
