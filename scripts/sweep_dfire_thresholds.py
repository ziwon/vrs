"""Sweep D-Fire detector thresholds from one detector pass.

The detector is run once using the lowest requested threshold, then cached
detections are rescored across the threshold grid. This keeps D-Fire fast-path
calibration practical while preserving the same watch-policy semantics used by
the RTSP runtime.
"""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vrs import setup_logging
from vrs.eval import aggregate_scores, config_for_eval_mode, score_alerts_against_truth
from vrs.eval.datasets import DFireDataset
from vrs.eval.harness import _detections_to_alerts, _load_image
from vrs.eval.schemas import GroundTruthEvent, RunScore
from vrs.policy import WatchItem, WatchPolicy
from vrs.schemas import Frame
from vrs.triage import YOLOEConfig, build_detector

SCHEMA_VERSION = "vrs.eval.dfire_threshold_sweep.v1"


@dataclass(frozen=True)
class CachedItem:
    image_path: Path
    events: list[GroundTruthEvent]
    alerts: list[dict[str, Any]]


def parse_thresholds(raw: str) -> list[float]:
    values = sorted({round(float(part.strip()), 4) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("threshold list is empty")
    bad = [value for value in values if not 0.0 <= value <= 1.0]
    if bad:
        raise ValueError(f"thresholds must be in [0,1], got {bad}")
    return values


def threshold_grid(classes: list[str], values: list[float]) -> list[dict[str, float]]:
    return [
        dict(zip(classes, combo, strict=True))
        for combo in itertools.product(values, repeat=len(classes))
    ]


def score_cached_items(
    cached_items: list[CachedItem],
    thresholds: dict[str, float],
    *,
    bbox_iou_threshold: float | None = None,
) -> RunScore:
    per_item = []
    classes = set(thresholds)
    for item in cached_items:
        alerts = [
            alert
            for alert in item.alerts
            if alert["class_name"] in classes
            and float(alert.get("confidence", 0.0)) >= thresholds[alert["class_name"]]
        ]
        per_item.append(
            score_alerts_against_truth(
                alerts,
                item.events,
                tolerance_s=0.0,
                bbox_iou_threshold=bbox_iou_threshold,
                classes=classes,
            )
        )
    return aggregate_scores(per_item)


def summarize_score(score: RunScore, classes: list[str]) -> dict[str, Any]:
    per_class = {name: score.per_class.get(name) for name in classes}
    macro_f1 = sum((metrics.f1 if metrics is not None else 0.0) for metrics in per_class.values())
    macro_recall = sum(
        (metrics.recall if metrics is not None else 0.0) for metrics in per_class.values()
    )
    if classes:
        macro_f1 /= len(classes)
        macro_recall /= len(classes)
    return {
        "overall": score.overall().to_dict(),
        "per_class": {
            name: (metrics.to_dict() if metrics is not None else _empty_class_metrics())
            for name, metrics in per_class.items()
        },
        "macro_f1": round(macro_f1, 4),
        "macro_recall": round(macro_recall, 4),
        "n_alerts_total": score.n_alerts_total,
        "n_events": score.n_events,
    }


def choose_best(results: list[dict[str, Any]], optimize: str) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        metrics = row["metrics"]
        overall = metrics["overall"]
        if optimize == "overall_f1":
            primary = overall["f1"]
        elif optimize == "overall_recall":
            primary = overall["recall"]
        else:
            primary = metrics["macro_f1"]
        return (float(primary), float(overall["recall"]), float(overall["precision"]))

    return max(results, key=key)


def policy_with_thresholds(policy_path: str | Path, thresholds: dict[str, float]) -> dict[str, Any]:
    raw = _load_policy_yaml(policy_path)
    seen = set()
    for item in raw["watch"]:
        name = str(item.get("name", "")).strip()
        if name in thresholds:
            item["min_score"] = float(thresholds[name])
            seen.add(name)
    missing = sorted(set(thresholds) - seen)
    if missing:
        raise ValueError(f"{policy_path}: missing watch entries for {missing}")
    return raw


def config_with_conf_floor(config: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    cfg = deepcopy(config)
    detector = dict(cfg.get("detector") or {})
    detector["conf_floor"] = min(float(detector.get("conf_floor", 1.0)), min(thresholds.values()))
    cfg["detector"] = detector
    return cfg


def build_policy_from_yaml(raw: dict[str, Any], *, low_threshold: float) -> WatchPolicy:
    items = []
    for item in raw["watch"]:
        min_score = min(float(item.get("min_score", low_threshold)), low_threshold)
        detector = item.get("detector") or [item["name"]]
        if isinstance(detector, str):
            detector = [detector]
        items.append(
            WatchItem(
                name=str(item["name"]),
                detector_prompts=[str(prompt) for prompt in detector],
                verifier_prompt=str(item.get("verifier", item["name"])),
                severity=str(item.get("severity", "medium")),
                min_score=min_score,
                min_persist_frames=int(item.get("min_persist_frames", 1)),
                verifier_window_s=(
                    float(item["verifier_window_s"])
                    if item.get("verifier_window_s") is not None
                    else None
                ),
            )
        )
    return WatchPolicy(items)


def _load_policy_yaml(policy_path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw.get("watch"), list) or not raw["watch"]:
        raise ValueError(f"{policy_path}: 'watch' key is empty or missing")
    return deepcopy(raw)


def _empty_class_metrics() -> dict[str, float]:
    return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep D-Fire detector thresholds")
    parser.add_argument("--dataset", required=True, help="D-Fire images/labels root")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--policy", default="configs/policies/dfire_eval.yaml")
    parser.add_argument("--out", default="runs/eval-dfire-sweep")
    parser.add_argument("--classes", default="fire,smoke")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="comma-separated min_score candidates",
    )
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=None,
        help="require bbox IoU at or above this threshold when GT boxes are present",
    )
    parser.add_argument(
        "--optimize",
        choices=("macro_f1", "overall_f1", "overall_recall"),
        default="macro_f1",
    )
    parser.add_argument(
        "--best-policy",
        default=None,
        help="path for the best threshold policy YAML (default: <out>/best_policy.yaml)",
    )
    parser.add_argument(
        "--best-config",
        default=None,
        help="path for the matching best config YAML (default: <out>/best_config.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    from vrs.pipeline import load_config

    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class")
    thresholds = parse_thresholds(args.thresholds)
    low_threshold = min(thresholds)

    config = config_for_eval_mode(load_config(args.config, verifier_enabled=False), "detector_only")
    det_cfg = dict(config["detector"])
    det_cfg["conf_floor"] = min(float(det_cfg.get("conf_floor", low_threshold)), low_threshold)
    config["detector"] = det_cfg

    raw_policy = _load_policy_yaml(args.policy)
    detector_policy = build_policy_from_yaml(raw_policy, low_threshold=low_threshold)
    detector = build_detector(
        YOLOEConfig(
            model=det_cfg["model"],
            device=det_cfg.get("device", "cuda"),
            imgsz=int(det_cfg.get("imgsz", 640)),
            conf_floor=float(det_cfg.get("conf_floor", 0.20)),
            iou=float(det_cfg.get("iou", 0.50)),
            half=bool(det_cfg.get("half", True)),
        ),
        detector_policy,
        backend=det_cfg.get("backend", "ultralytics"),
    )

    cached_items: list[CachedItem] = []
    for item in DFireDataset(args.dataset):
        image = _load_image(item.video_path)
        detections = detector(Frame(index=0, pts_s=0.0, image=image))
        cached_items.append(
            CachedItem(
                image_path=item.video_path,
                events=item.events,
                alerts=_detections_to_alerts(detections, image),
            )
        )

    results = []
    for candidate in threshold_grid(classes, thresholds):
        score = score_cached_items(
            cached_items,
            candidate,
            bbox_iou_threshold=args.bbox_iou_threshold,
        )
        results.append({"thresholds": candidate, "metrics": summarize_score(score, classes)})

    best = choose_best(results, args.optimize)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_policy_path = Path(args.best_policy) if args.best_policy else out_dir / "best_policy.yaml"
    best_config_path = Path(args.best_config) if args.best_config else out_dir / "best_config.yaml"

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "config_path": str(args.config),
        "policy_path": str(args.policy),
        "best_policy_path": str(best_policy_path),
        "best_config_path": str(best_config_path),
        "classes": classes,
        "threshold_candidates": thresholds,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "optimize": args.optimize,
        "best": best,
        "results": sorted(
            results,
            key=lambda row: (
                row["metrics"]["macro_f1"],
                row["metrics"]["overall"]["recall"],
                row["metrics"]["overall"]["precision"],
            ),
            reverse=True,
        ),
    }
    (out_dir / "sweep.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    best_policy = policy_with_thresholds(args.policy, best["thresholds"])
    best_policy_path.parent.mkdir(parents=True, exist_ok=True)
    best_policy_path.write_text(yaml.safe_dump(best_policy, sort_keys=False), encoding="utf-8")

    best_config = config_with_conf_floor(config, best["thresholds"])
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")

    metrics = best["metrics"]
    print("\n-- best thresholds -------------------------")
    print(" ".join(f"{name}={value:.2f}" for name, value in best["thresholds"].items()))
    print(
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"overall P={metrics['overall']['precision']:.3f} "
        f"R={metrics['overall']['recall']:.3f} "
        f"F1={metrics['overall']['f1']:.3f}"
    )
    for name in classes:
        row = metrics["per_class"][name]
        print(
            f"  {name:<10} tp={row['tp']:<3} fp={row['fp']:<3} fn={row['fn']:<3} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
        )
    print(f"\nReport: {out_dir / 'sweep.json'}")
    print(f"Best policy: {best_policy_path}")
    print(f"Best config: {best_config_path}")


if __name__ == "__main__":
    main()
