"""Run the VRS eval harness against a labeled dataset.

Example:
    uv run scripts/eval.py \\
        --dataset      /data/eval/our_labeled_clips \\
        --config       configs/default.yaml \\
        --policy       configs/policies/safety.yaml \\
        --out          runs/eval \\
        --tolerance-s  1.0

Layout of the dataset directory — see vrs.eval.datasets.labeled_dir for the
sidecar-JSON schema, or pass ``--dataset-format dfire`` for a D-Fire
``images/`` + ``labels/`` tree. Each media item gets its own subdir under
``--out`` holding its ``alerts.jsonl`` and event thumbnails. A single versioned
``report.json`` lands at ``--out`` (or the path given via ``--report``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vrs import setup_logging
from vrs.eval import EvalReport, config_for_eval_mode, evaluate
from vrs.eval.datasets import DATASET_ADAPTERS, build_dataset

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="VRS — evaluate the cascade on a labeled dataset")
    ap.add_argument("--dataset", required=True, help="dataset root")
    ap.add_argument(
        "--dataset-format",
        choices=tuple(sorted(DATASET_ADAPTERS)),
        default="labeled_dir",
        help="dataset adapter to use",
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument(
        "--mode",
        choices=("full_cascade", "detector_only"),
        default="full_cascade",
        help="evaluation mode: full cascade with verifier, or detector/event-state only",
    )
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument(
        "--tolerance-s",
        type=float,
        default=1.0,
        help="temporal slack around GT event windows when matching alerts",
    )
    ap.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=None,
        help="require bbox IoU at or above this threshold when GT boxes are present",
    )
    ap.add_argument("--report", default=None, help="JSON report path (default: <out>/report.json)")
    return ap


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    # Imported lazily so `uv run scripts/eval.py --help` works without cv2/torch.
    from vrs.pipeline import VRSPipeline, load_config
    from vrs.policy import load_watch_policy

    dataset = build_dataset(args.dataset_format, args.dataset)
    verifier_enabled = False if args.mode == "detector_only" else None
    config = config_for_eval_mode(
        load_config(args.config, verifier_enabled=verifier_enabled),
        args.mode,
    )
    policy = load_watch_policy(args.policy)

    def _factory(video_out: Path):
        return VRSPipeline(config, policy, video_out)

    result = evaluate(
        dataset=dataset,
        pipeline_factory=_factory,
        out_dir=args.out,
        tolerance_s=args.tolerance_s,
        bbox_iou_threshold=args.bbox_iou_threshold,
    )

    report_path = Path(args.report) if args.report else Path(args.out) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
    print(
        f"alerts={agg.n_alerts_total} (true={agg.n_alerts_true})   "
        f"events={agg.n_events}   "
        f"flip_rate={agg.flip_rate:.3f}   "
        f"fn_flag_rate={agg.fn_flag_rate:.3f}"
    )
    print(f"P={overall['precision']:.3f}  R={overall['recall']:.3f}  F1={overall['f1']:.3f}")
    for cls, cm in sorted(agg.per_class.items()):
        d = cm.to_dict()
        print(
            f"  {cls:<10} tp={d['tp']:<3} fp={d['fp']:<3} fn={d['fn']:<3}  "
            f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}"
        )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
