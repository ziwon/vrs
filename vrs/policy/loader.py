"""YAML loading helpers for scenario verifier policy packs."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PolicyPack


def load_policy_pack(path: str | Path) -> PolicyPack:
    """Load a scenario verifier policy pack from YAML."""

    policy_path = Path(path)
    with policy_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return PolicyPack.from_mapping(raw)
    except ValueError as e:
        raise ValueError(f"{policy_path}: {e}") from e


def load_policy_packs(paths: list[str | Path] | tuple[str | Path, ...]) -> list[PolicyPack]:
    """Load multiple policy packs in caller-provided priority order."""

    return [load_policy_pack(path) for path in paths]
