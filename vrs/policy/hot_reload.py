"""Safe hot reload for watch-policy changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .watch_policy import WatchPolicy, load_watch_policy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyReloadResult:
    reloaded: bool
    reason: str


def is_runtime_safe_policy_update(current: WatchPolicy, new: WatchPolicy) -> tuple[bool, str]:
    """Return whether ``new`` can be applied without rebuilding the detector."""
    if current.names() != new.names():
        return False, "event names changed; detector/verifier reload required"
    if current.yoloe_vocabulary() != new.yoloe_vocabulary():
        return False, "detector prompts changed; detector reload required"
    return True, "runtime-safe policy fields changed"


class PolicyReloader:
    """Poll a watch-policy file and apply runtime-safe changes in place."""

    def __init__(self, path: str | Path, policy: WatchPolicy):
        self.path = Path(path)
        self.policy = policy
        self._mtime_ns = self._stat_mtime_ns()

    def maybe_reload(self, *, force: bool = False) -> PolicyReloadResult:
        try:
            mtime_ns = self._stat_mtime_ns()
        except OSError as exc:
            logger.warning("policy reload skipped; cannot stat %s: %s", self.path, exc)
            return PolicyReloadResult(False, f"stat failed: {exc}")

        if not force and mtime_ns == self._mtime_ns:
            return PolicyReloadResult(False, "unchanged")

        try:
            new_policy = load_watch_policy(self.path)
        except Exception as exc:
            logger.warning("policy reload rejected; invalid policy %s: %s", self.path, exc)
            return PolicyReloadResult(False, f"invalid policy: {exc}")

        ok, reason = is_runtime_safe_policy_update(self.policy, new_policy)
        if not ok:
            logger.warning("policy reload rejected for %s: %s", self.path, reason)
            self._mtime_ns = mtime_ns
            return PolicyReloadResult(False, reason)

        self.policy = new_policy
        self._mtime_ns = mtime_ns
        logger.info("policy reloaded from %s: %s", self.path, reason)
        return PolicyReloadResult(True, reason)

    def _stat_mtime_ns(self) -> int:
        return self.path.stat().st_mtime_ns
