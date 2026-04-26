"""Run the VRS eval harness against a labeled dataset.

Example:
    python scripts/eval.py \\
        --dataset      /data/eval/our_labeled_clips \\
        --config       configs/default.yaml \\
        --policy       configs/policies/safety.yaml \\
        --out          runs/eval \\
        --tolerance-s  1.0

Layout of the dataset directory — see vrs.eval.datasets.labeled_dir for the
sidecar-JSON schema. Each video gets its own subdir under ``--out`` holding
its ``alerts.jsonl`` and event thumbnails. A single versioned ``report.json``
lands at ``--out`` (or the path given via ``--report``).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from vrs import setup_logging
from vrs.eval import EvalReport, evaluate
from vrs.eval.datasets import LabeledDirDataset

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="VRS — evaluate the cascade on a labeled dataset")
    ap.add_argument("--dataset", required=True, help="labeled dataset root (see LabeledDirDataset)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--tolerance-s", type=float, default=1.0,
                    help="temporal slack around GT event windows when matching alerts")
    ap.add_argument("--report", default=None,
                    help="JSON report path (default: <out>/report.json)")
    args = ap.parse_args()

    # Imported lazily so `python scripts/eval.py --help` works without cv2/torch.
    from vrs.pipeline import build_pipeline

    dataset = LabeledDirDataset(args.dataset)

    def _factory(video_out: Path):
        return build_pipeline(args.config, args.policy, video_out)

    result = evaluate(
        dataset=dataset,
        pipeline_factory=_factory,
        out_dir=args.out,
        tolerance_s=args.tolerance_s,
    )

    report_path = Path(args.report) if args.report else Path(args.out) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    report = EvalReport.from_harness_result(
        result,
        dataset=args.dataset,
        config_path=args.config,
        policy_path=args.policy,
        config=config,
    )
    report.write(report_path)

    agg = result.aggregate
    overall = agg.overall().to_dict()
    print("\n── aggregate ──────────────────────────────")
    print(f"alerts={agg.n_alerts_total} (true={agg.n_alerts_true})   "
          f"events={agg.n_events}   "
          f"flip_rate={agg.flip_rate:.3f}   "
          f"fn_flag_rate={agg.fn_flag_rate:.3f}")
    print(f"P={overall['precision']:.3f}  R={overall['recall']:.3f}  F1={overall['f1']:.3f}")
    for cls, cm in sorted(agg.per_class.items()):
        d = cm.to_dict()
        print(f"  {cls:<10} tp={d['tp']:<3} fp={d['fp']:<3} fn={d['fn']:<3}  "
              f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
