"""Regression gate: compare a new eval report against a committed baseline.

Usage::

    python -m vrs.eval.ci \\
        --baseline runs/baseline/report.json \\
        --current  runs/eval/report.json \\
        --max-f1-drop 0.02

Exit codes:
    0   every per-class F1 and the overall F1 are within tolerance
    1   at least one F1 dropped by more than --max-f1-drop
    2   a structural error (missing file, malformed report, etc.)

The gate is deliberately narrow: it only checks F1 deltas, since flip-rate
and FN-flag-rate are diagnostic signals whose "right direction" depends on
whether the detector or the verifier changed. Those numbers are still
printed so a reviewer can eyeball them.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ClassDelta:
    class_name: str
    baseline_f1: float
    current_f1: float
    regressed: bool
    note: str = ""   # e.g. "missing in current", "new class"

    @property
    def delta_f1(self) -> float:
        return self.current_f1 - self.baseline_f1


@dataclass
class GateResult:
    passed: bool
    overall: ClassDelta
    per_class: List[ClassDelta] = field(default_factory=list)
    max_f1_drop: float = 0.02
    baseline_flip_rate: float = 0.0
    current_flip_rate: float = 0.0
    baseline_fn_flag_rate: float = 0.0
    current_fn_flag_rate: float = 0.0

    def regressions(self) -> List[ClassDelta]:
        out = [d for d in self.per_class if d.regressed]
        if self.overall.regressed:
            out.append(self.overall)
        return out

    def render(self) -> str:
        lines: List[str] = []
        lines.append(f"Regression gate (max F1 drop: {self.max_f1_drop:+.3f})")
        lines.append("")
        lines.append(f"  {'class':<14} {'baseline':>10} {'current':>10} {'delta':>10}  status")
        lines.append(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}  {'-'*6}")
        for d in sorted(self.per_class, key=lambda x: x.class_name):
            lines.append(self._fmt_row(d))
        lines.append(self._fmt_row(self.overall))
        lines.append("")
        lines.append(
            f"  flip_rate     : baseline={self.baseline_flip_rate:.3f}   "
            f"current={self.current_flip_rate:.3f}   "
            f"delta={self.current_flip_rate - self.baseline_flip_rate:+.3f}"
        )
        lines.append(
            f"  fn_flag_rate  : baseline={self.baseline_fn_flag_rate:.3f}   "
            f"current={self.current_fn_flag_rate:.3f}   "
            f"delta={self.current_fn_flag_rate - self.baseline_fn_flag_rate:+.3f}"
        )
        lines.append("")
        lines.append("PASS" if self.passed else f"FAIL — {len(self.regressions())} regression(s)")
        return "\n".join(lines)

    @staticmethod
    def _fmt_row(d: ClassDelta) -> str:
        status = "FAIL" if d.regressed else "ok"
        note = f"  ({d.note})" if d.note else ""
        return (
            f"  {d.class_name:<14} {d.baseline_f1:>10.3f} "
            f"{d.current_f1:>10.3f} {d.delta_f1:>+10.3f}  {status}{note}"
        )


# ──────────────────────────────────────────────────────────────────────

def _per_class_f1(report: dict) -> Dict[str, float]:
    per = report.get("aggregate", {}).get("per_class", {})
    return {cls: float(m.get("f1", 0.0)) for cls, m in per.items()}


def _overall_f1(report: dict) -> float:
    return float(report.get("aggregate", {}).get("overall", {}).get("f1", 0.0))


def compare_reports(
    baseline: dict,
    current: dict,
    *,
    max_f1_drop: float = 0.02,
    classes: Optional[Iterable[str]] = None,
) -> GateResult:
    """Diff two eval reports. Returns a ``GateResult`` whose ``passed`` flag
    is False if any per-class or overall F1 dropped by more than
    ``max_f1_drop``.

    Classes present only in the current report are reported as informational
    ("new class") and never fail the gate. Classes present only in the
    baseline are treated as regressions — the current run should produce at
    least the same classes; if a class was intentionally removed, update the
    baseline.
    """
    if max_f1_drop < 0:
        raise ValueError("max_f1_drop must be >= 0")
    if "aggregate" not in baseline or "aggregate" not in current:
        raise ValueError("both reports must have an 'aggregate' key")

    b_pc = _per_class_f1(baseline)
    c_pc = _per_class_f1(current)
    all_classes = set(b_pc) | set(c_pc)
    if classes is not None:
        all_classes &= set(classes)

    per_class: List[ClassDelta] = []
    for cls in sorted(all_classes):
        bf1 = b_pc.get(cls, 0.0)
        cf1 = c_pc.get(cls, 0.0)
        note = ""
        if cls not in b_pc:
            note = "new class"
            regressed = False                                  # can't regress what didn't exist
        elif cls not in c_pc:
            note = "missing in current"
            regressed = bf1 > 0.0                              # treat as F1 dropped to 0
        else:
            regressed = (bf1 - cf1) > max_f1_drop
        per_class.append(ClassDelta(
            class_name=cls,
            baseline_f1=bf1,
            current_f1=cf1,
            regressed=regressed,
            note=note,
        ))

    bof1 = _overall_f1(baseline)
    cof1 = _overall_f1(current)
    overall = ClassDelta(
        class_name="OVERALL",
        baseline_f1=bof1,
        current_f1=cof1,
        regressed=(bof1 - cof1) > max_f1_drop,
    )

    b_agg = baseline.get("aggregate", {})
    c_agg = current.get("aggregate", {})
    passed = not overall.regressed and not any(d.regressed for d in per_class)
    return GateResult(
        passed=passed,
        per_class=per_class,
        overall=overall,
        max_f1_drop=max_f1_drop,
        baseline_flip_rate=float(b_agg.get("flip_rate", 0.0)),
        current_flip_rate=float(c_agg.get("flip_rate", 0.0)),
        baseline_fn_flag_rate=float(b_agg.get("fn_flag_rate", 0.0)),
        current_fn_flag_rate=float(c_agg.get("fn_flag_rate", 0.0)),
    )


# ──────────────────────────────────────────────────────────────────────
# cli
# ──────────────────────────────────────────────────────────────────────

def _load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="VRS eval regression gate — fail when per-class or overall F1 drops too far"
    )
    ap.add_argument("--baseline", required=True, help="path to baseline report.json")
    ap.add_argument("--current", required=True, help="path to current report.json")
    ap.add_argument("--max-f1-drop", type=float, default=0.02,
                    help="allowed F1 drop per class before failing (default: 0.02)")
    ap.add_argument("--classes", default=None,
                    help="comma-separated subset of classes to gate on (default: all)")
    args = ap.parse_args(argv)

    try:
        baseline = _load_report(Path(args.baseline))
        current = _load_report(Path(args.current))
    except FileNotFoundError as e:
        print(f"error: report not found: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"error: malformed JSON: {e}", file=sys.stderr)
        return 2

    try:
        classes = None
        if args.classes:
            classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        result = compare_reports(
            baseline, current, max_f1_drop=args.max_f1_drop, classes=classes,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(result.render())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
