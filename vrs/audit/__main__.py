"""CLI entrypoint for audit-log verification.

Example:
    python -m vrs.audit --log runs/demo/alerts.jsonl \
        --mode hmac_sha256 --key-env VRS_AUDIT_HMAC_KEY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .signing import verify_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify tamper-evident VRS alert JSONL logs")
    p.add_argument("log_path", nargs="?", help="path to alerts.jsonl")
    p.add_argument("--log", dest="log_opt", help="path to alerts.jsonl")
    p.add_argument(
        "--mode",
        choices=("sha256", "hmac_sha256"),
        default=None,
        help="expected audit mode; defaults to each record's audit_mode field",
    )
    p.add_argument("--key", default=None, help="HMAC key literal; prefer --key-env")
    p.add_argument("--key-env", default=None, help="environment variable containing HMAC key")
    p.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="skip unsigned lines instead of treating them as verification failures",
    )
    args = p.parse_args(argv)

    log_path = args.log_opt or args.log_path
    if not log_path:
        p.error("missing log path")

    result = verify_jsonl(
        Path(log_path),
        mode=args.mode,
        key=args.key,
        key_env=args.key_env,
        allow_unsigned=args.allow_unsigned,
    )
    if result.valid:
        print(
            f"OK: verified {result.checked_records} signed record(s)"
            f" ({result.unsigned_records} unsigned skipped)"
        )
        return 0

    for error in result.errors:
        print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
