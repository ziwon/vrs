"""Write the committed mini eval baseline report.

This is intentionally synthetic: it exercises the stable report schema and CI
regression gate without requiring GPU inference, model downloads, or video
artifacts in the repository.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vrs.eval import (
    EvalReport,
    HarnessResult,
    ReportRuntime,
    RunScore,
)
from vrs.eval.schemas import ClassMetrics


def _score(
    *,
    fire: ClassMetrics,
    smoke: ClassMetrics,
    n_alerts_total: int,
    n_alerts_true: int,
    n_fn_flagged: int,
    n_events: int,
) -> RunScore:
    return RunScore(
        per_class={"fire": fire, "smoke": smoke},
        n_alerts_total=n_alerts_total,
        n_alerts_true=n_alerts_true,
        n_fn_flagged=n_fn_flagged,
        n_events=n_events,
    )


def build_baseline_report() -> EvalReport:
    """Build the deterministic report used by ``baselines/eval/report.json``."""
    aggregate = _score(
        fire=ClassMetrics(tp=3, fp=1, fn=1),
        smoke=ClassMetrics(tp=2, fp=1, fn=0),
        n_alerts_total=8,
        n_alerts_true=7,
        n_fn_flagged=1,
        n_events=6,
    )
    per_video = [
        (
            Path("mini-fire-smoke-001.mp4"),
            _score(
                fire=ClassMetrics(tp=2, fp=0, fn=0),
                smoke=ClassMetrics(tp=1, fp=1, fn=0),
                n_alerts_total=4,
                n_alerts_true=3,
                n_fn_flagged=0,
                n_events=3,
            ),
        ),
        (
            Path("mini-fire-smoke-002.mp4"),
            _score(
                fire=ClassMetrics(tp=1, fp=1, fn=1),
                smoke=ClassMetrics(tp=1, fp=0, fn=0),
                n_alerts_total=4,
                n_alerts_true=4,
                n_fn_flagged=1,
                n_events=3,
            ),
        ),
    ]
    report = EvalReport.from_harness_result(
        HarnessResult(aggregate=aggregate, per_video=per_video),
        dataset="mini-eval-fixture",
        config_path="configs/default.yaml",
        policy_path="configs/policies/safety.yaml",
        config={
            "detector": {"backend": "ultralytics", "model": "yoloe-11l-seg.pt"},
            "verifier": {
                "enabled": True,
                "backend": "transformers",
                "model_id": "nvidia/Cosmos-Reason2-2B",
            },
        },
        created_at=datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
        run_id="2026-04-26-mini-eval-fixture-yoloe-11l-seg-cosmos-reason2-2b",
    )
    return replace(report, runtime=ReportRuntime())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the committed mini eval baseline")
    parser.add_argument("--out", default="baselines/eval/report.json")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_baseline_report().write(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
