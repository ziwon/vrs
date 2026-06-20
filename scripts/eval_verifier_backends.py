"""Run identical full-cascade evals across verifier backend configs.

This is the Task-4 bake-off wrapper: it does not decide a production model on
its own. It runs each candidate config against the same labeled dataset, reads
the stable ``report.json`` files, and emits one comparison JSON containing the
quality, latency, malformed-output, and runtime fields needed for review.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrs.eval import EvalReport


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    config_path: Path


def parse_candidate(raw: str) -> CandidateSpec:
    """Parse ``name=path.yaml`` or infer a stable name from ``path.yaml``."""
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(raw).stem
    name = _slug(name)
    if not name:
        raise ValueError(f"invalid candidate name in {raw!r}")
    return CandidateSpec(name=name, config_path=Path(path))


def summarize_report(candidate: CandidateSpec, report_path: Path) -> dict[str, Any]:
    report = EvalReport.load(report_path)
    verifier = report.models.verifier
    return {
        "name": candidate.name,
        "config_path": str(candidate.config_path),
        "report_path": str(report_path),
        "verifier": verifier.to_dict() if verifier is not None else None,
        "metrics": report.metrics.to_dict(),
        "quality_signals": report.quality_signals.to_dict(),
        "latency": report.latency.to_dict(),
        "runtime": report.runtime.to_dict(),
    }


def build_eval_command(
    *,
    candidate: CandidateSpec,
    dataset: Path,
    dataset_format: str,
    policy: Path,
    out_dir: Path,
    tolerance_s: float,
    bbox_iou_threshold: float | None,
) -> list[str]:
    report_path = out_dir / candidate.name / "report.json"
    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--dataset",
        str(dataset),
        "--dataset-format",
        dataset_format,
        "--config",
        str(candidate.config_path),
        "--policy",
        str(policy),
        "--mode",
        "full_cascade",
        "--out",
        str(out_dir / candidate.name),
        "--report",
        str(report_path),
        "--tolerance-s",
        str(tolerance_s),
    ]
    if bbox_iou_threshold is not None:
        cmd.extend(["--bbox-iou-threshold", str(bbox_iou_threshold)])
    return cmd


def run_bakeoff(
    *,
    candidates: list[CandidateSpec],
    dataset: Path,
    dataset_format: str,
    policy: Path,
    out_dir: Path,
    tolerance_s: float,
    bbox_iou_threshold: float | None = None,
    skip_run: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_out = out_dir / candidate.name
        candidate_out.mkdir(parents=True, exist_ok=True)
        report_path = candidate_out / "report.json"
        if not skip_run:
            cmd = build_eval_command(
                candidate=candidate,
                dataset=dataset,
                dataset_format=dataset_format,
                policy=policy,
                out_dir=out_dir,
                tolerance_s=tolerance_s,
                bbox_iou_threshold=bbox_iou_threshold,
            )
            subprocess.run(cmd, check=True)
        rows.append(summarize_report(candidate, report_path))

    comparison = {
        "dataset": str(dataset),
        "dataset_format": dataset_format,
        "policy": str(policy),
        "tolerance_s": tolerance_s,
        "bbox_iou_threshold": bbox_iou_threshold,
        "candidates": rows,
    }
    (out_dir / "verifier_bakeoff.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verifier backend bake-off evals")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-format", default="labeled_dir")
    parser.add_argument("--policy", default="configs/policies/safety.yaml", type=Path)
    parser.add_argument("--out", default="runs/verifier-bakeoff", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate config as name=path.yaml; repeat for Cosmos/Qwen/etc.",
    )
    parser.add_argument("--tolerance-s", type=float, default=1.0)
    parser.add_argument("--bbox-iou-threshold", type=float, default=None)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="summarize existing <out>/<candidate>/report.json files without rerunning eval",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    candidates = [parse_candidate(raw) for raw in args.candidate]
    comparison = run_bakeoff(
        candidates=candidates,
        dataset=args.dataset,
        dataset_format=args.dataset_format,
        policy=args.policy,
        out_dir=args.out,
        tolerance_s=args.tolerance_s,
        bbox_iou_threshold=args.bbox_iou_threshold,
        skip_run=args.skip_run,
    )
    print(f"Report: {args.out / 'verifier_bakeoff.json'}")
    for row in comparison["candidates"]:
        overall = row["metrics"]["overall"]
        quality = row["quality_signals"]
        latency = row["latency"]
        print(
            f"{row['name']}: "
            f"P={_fmt_float(overall['precision'])} "
            f"R={_fmt_float(overall['recall'])} "
            f"F1={_fmt_float(overall['f1'])} "
            f"flip={_fmt_float(quality['verifier_flip_rate'])} "
            f"bad_json={_fmt_float(quality['malformed_json_rate'])} "
            f"verifier_p95_ms={latency['verifier_p95_ms']}"
        )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _fmt_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
