"""VRS — Video Reasoning System.

Two-stage cascade:
  1. YOLOE-L open-vocabulary detector (fast path, ~6 ms / frame).
  2. Cosmos-Reason2-2B physical-reasoning VLM (slow path, runs only on candidates).
"""

from __future__ import annotations

import logging

__version__ = "0.3.0"

# Library-safe default: NullHandler so consumers see no output unless they
# configure logging themselves.  Scripts call setup_logging() to get the
# human-readable default.
logging.getLogger("vrs").addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO) -> None:
    """Activate VRS's default human-readable log format.

    Called automatically by the CLI scripts in ``scripts/``.  Library
    consumers can ignore this and attach their own handlers to the ``"vrs"``
    logger instead.
    """
    root = logging.getLogger("vrs")
    # Don't double-add if the user already configured handlers.
    if any(not isinstance(h, logging.NullHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
