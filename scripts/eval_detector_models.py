"""Compare detector model variants on one image-labeled eval set.

Task 3 is deliberately narrower than the prompt/threshold sweeps: keep the
watch policy fixed, swap only the detector model, and record accuracy plus
latency deltas. This makes YOLOE-L versus YOLOE-26 decisions auditable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from vrs import setup_logging
from vrs.eval import (
    aggregate_scores,
    config_for_eval_mode,
    dataset_items_are_images,
    score_alerts_against_truth,
)
from vrs.eval.datasets import DATASET_ADAPTERS, build_dataset
from vrs.eval.harness import _detections_to_alerts, _load_image
from vrs.eval.schemas import RunScore
from vrs.policy import WatchPolicy, load_watch_policy
from vrs.schemas import Frame
from vrs.triage import YOLOEConfig, build_detector

SCHEMA_VERSION = "vrs.eval.detector_model_refresh.v1"


def parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def config_with_model(config: dict[str, Any], *, model: str, policy: WatchPolicy) -> dict[str, Any]:
    cfg = deepcopy(config)
    detector = dict(cfg.get("detector") or {})
    detector["model"] = model
    policy_items = list(policy)
    if policy_items:
        detector["conf_floor"] = min(
            float(detector.get("conf_floor", 1.0)),
            min(float(item.min_score) for item in policy_items),
        )
    cfg["detector"] = detector
    return cfg


def summarize_score(score: RunScore, classes: list[str]) -> dict[str, Any]:
    per_class = {name: score.per_class.get(name) for name in classes}
    macro_f1 = sum((metrics.f1 if metrics is not None else 0.0) for metrics in per_class.values())
    if classes:
        macro_f1 /= len(classes)
    return {
        "overall": score.overall().to_dict(),
        "per_class": {
            name: (metrics.to_dict() if metrics is not None else _empty_class_metrics())
            for name, metrics in per_class.items()
        },
        "macro_f1": round(macro_f1, 4),
        "n_alerts_total": score.n_alerts_total,
        "n_events": score.n_events,
    }


def summarize_latency(latencies_ms: list[float]) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(latencies_ms),
        "mean_ms": round(statistics.fmean(latencies_ms), 4),
        "p50_ms": round(_percentile(latencies_ms, 50), 4),
        "p95_ms": round(_percentile(latencies_ms, 95), 4),
    }


def add_baseline_deltas(rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in rows}
    baseline = by_model.get(baseline_model) or rows[0]
    base_metrics = baseline["metrics"]
    base_latency = baseline["latency"]
    base_p95 = base_latency.get("p95_ms")
    out = []
    for row in rows:
        current_p95 = row["latency"].get("p95_ms")
        row = dict(row)
        row["delta_vs_baseline"] = {
            "macro_f1": round(row["metrics"]["macro_f1"] - base_metrics["macro_f1"], 4),
            "overall_f1": round(
                row["metrics"]["overall"]["f1"] - base_metrics["overall"]["f1"],
                4,
            ),
            "overall_recall": round(
                row["metrics"]["overall"]["recall"] - base_metrics["overall"]["recall"],
                4,
            ),
            "p95_latency_ms": (
                round(float(current_p95) - float(base_p95), 4)
                if current_p95 is not None and base_p95 is not None
                else None
            ),
        }
        out.append(row)
    return out


def choose_best(rows: list[dict[str, Any]], optimize: str) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = row["metrics"]
        overall = metrics["overall"]
        latency = row["latency"].get("p95_ms")
        if optimize == "overall_f1":
            primary = overall["f1"]
        elif optimize == "overall_recall":
            primary = overall["recall"]
        else:
            primary = metrics["macro_f1"]
        return (
            float(primary),
            float(overall["recall"]),
            float(overall["precision"]),
            -float(latency) if latency is not None else 0.0,
        )

    return max(rows, key=key)


def make_decision(
    rows: list[dict[str, Any]],
    *,
    baseline_model: str,
    optimize: str,
    min_metric_gain: float,
    max_p95_latency_ratio: float,
) -> dict[str, Any]:
    rows_by_model = {row["model"]: row for row in rows}
    baseline = rows_by_model.get(baseline_model) or rows[0]
    best = choose_best(rows, optimize)
    base_p95 = baseline["latency"].get("p95_ms")
    best_p95 = best["latency"].get("p95_ms")
    latency_ratio = (
        round(float(best_p95) / float(base_p95), 4)
        if best_p95 is not None and base_p95 not in (None, 0)
        else None
    )
    delta = best["delta_vs_baseline"]
    if optimize == "overall_f1":
        gain = delta["overall_f1"]
    elif optimize == "overall_recall":
        gain = delta["overall_recall"]
    else:
        gain = delta["macro_f1"]
    latency_ok = latency_ratio is None or latency_ratio <= max_p95_latency_ratio
    should_adopt = best["model"] != baseline["model"] and gain >= min_metric_gain and latency_ok
    return {
        "baseline_model": baseline["model"],
        "best_model": best["model"],
        "optimize": optimize,
        "metric_gain": gain,
        "p95_latency_ratio": latency_ratio,
        "action": "adopt_candidate" if should_adopt else "keep_baseline",
        "reason": (
            "candidate clears metric and latency gates"
            if should_adopt
            else "best candidate does not clear metric/latency gates"
        ),
    }


def evaluate_model(
    *,
    dataset,
    config: dict[str, Any],
    policy: WatchPolicy,
    classes: list[str],
    bbox_iou_threshold: float | None,
) -> dict[str, Any]:
    det_cfg = dict(config["detector"])
    try:
        return _evaluate_model_once(
            dataset=dataset,
            det_cfg=det_cfg,
            policy=policy,
            classes=classes,
            bbox_iou_threshold=bbox_iou_threshold,
            half_fallback=False,
        )
    except RuntimeError as exc:
        if not det_cfg.get("half", True) or not _is_half_dtype_mismatch(exc):
            raise
        det_cfg["half"] = False
        row = _evaluate_model_once(
            dataset=dataset,
            det_cfg=det_cfg,
            policy=policy,
            classes=classes,
            bbox_iou_threshold=bbox_iou_threshold,
            half_fallback=True,
        )
        row["warnings"] = [
            "FP16 detector inference failed with a dtype mismatch; retried this model with half=false"
        ]
        return row


def _evaluate_model_once(
    *,
    dataset,
    det_cfg: dict[str, Any],
    policy: WatchPolicy,
    classes: list[str],
    bbox_iou_threshold: float | None,
    half_fallback: bool,
) -> dict[str, Any]:
    detector = build_detector(
        YOLOEConfig(
            model=det_cfg["model"],
            device=det_cfg.get("device", "cuda"),
            imgsz=int(det_cfg.get("imgsz", 640)),
            conf_floor=float(det_cfg.get("conf_floor", 0.20)),
            iou=float(det_cfg.get("iou", 0.50)),
            half=bool(det_cfg.get("half", True)),
        ),
        policy,
        backend=det_cfg.get("backend", "ultralytics"),
    )

    scores = []
    latencies_ms = []
    for item in dataset:
        image = _load_image(item.video_path)
        started = time.perf_counter()
        detections = detector(Frame(index=0, pts_s=0.0, image=image))
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        alerts = _detections_to_alerts(detections, image)
        scores.append(
            score_alerts_against_truth(
                alerts,
                item.events,
                tolerance_s=0.0,
                bbox_iou_threshold=bbox_iou_threshold,
                classes=classes,
            )
        )
    score = aggregate_scores(scores)
    return {
        "model": det_cfg["model"],
        "detector_config": {
            "backend": det_cfg.get("backend", "ultralytics"),
            "imgsz": int(det_cfg.get("imgsz", 640)),
            "conf_floor": float(det_cfg.get("conf_floor", 0.20)),
            "iou": float(det_cfg.get("iou", 0.50)),
            "half": bool(det_cfg.get("half", True)),
            "half_fallback": half_fallback,
        },
        "metrics": summarize_score(score, classes),
        "latency": summarize_latency(latencies_ms),
    }


def _is_half_dtype_mismatch(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "same dtype" in msg and ("half" in msg or "float" in msg)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _empty_class_metrics() -> dict[str, float]:
    return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare detector models on a fixed eval setup")
    parser.add_argument("--dataset", required=True, help="image dataset root")
    parser.add_argument(
        "--dataset-format",
        choices=tuple(sorted(DATASET_ADAPTERS)),
        default="dfire",
    )
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--policy", default="configs/policies/dfire_eval.yaml")
    parser.add_argument("--out", default="runs/eval-dfire-model-refresh")
    parser.add_argument("--models", default="yoloe-11l-seg.pt,yoloe-26l-seg.pt")
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--classes", default="fire,smoke")
    parser.add_argument("--bbox-iou-threshold", type=float, default=None)
    parser.add_argument(
        "--optimize",
        choices=("macro_f1", "overall_f1", "overall_recall"),
        default="macro_f1",
    )
    parser.add_argument("--min-metric-gain", type=float, default=0.01)
    parser.add_argument("--max-p95-latency-ratio", type=float, default=1.10)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    from vrs.pipeline import load_config

    models = parse_models(args.models)
    baseline_model = args.baseline_model or models[0]
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class")

    dataset = build_dataset(args.dataset_format, args.dataset)
    if not dataset_items_are_images(dataset):
        raise ValueError("detector model refresh currently requires an image-backed dataset")

    base_config = config_for_eval_mode(
        load_config(args.config, verifier_enabled=False),
        "detector_only",
    )
    policy = load_watch_policy(args.policy)

    rows = []
    for model in models:
        print(f"running detector model={model}")
        cfg = config_with_model(base_config, model=model, policy=policy)
        rows.append(
            evaluate_model(
                dataset=build_dataset(args.dataset_format, args.dataset),
                config=cfg,
                policy=policy,
                classes=classes,
                bbox_iou_threshold=args.bbox_iou_threshold,
            )
        )

    rows = add_baseline_deltas(rows, baseline_model)
    rows = sorted(
        rows,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["overall"]["recall"],
            row["metrics"]["overall"]["precision"],
            -(row["latency"]["p95_ms"] or 0.0),
        ),
        reverse=True,
    )
    decision = make_decision(
        rows,
        baseline_model=baseline_model,
        optimize=args.optimize,
        min_metric_gain=args.min_metric_gain,
        max_p95_latency_ratio=args.max_p95_latency_ratio,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "dataset_format": args.dataset_format,
        "config_path": str(args.config),
        "policy_path": str(args.policy),
        "classes": classes,
        "models": models,
        "baseline_model": baseline_model,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "optimize": args.optimize,
        "decision": decision,
        "results": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_refresh.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n-- detector model refresh ------------------")
    print(f"decision={decision['action']} best={decision['best_model']}")
    for row in rows:
        metrics = row["metrics"]
        latency = row["latency"]
        delta = row["delta_vs_baseline"]
        print(
            f"{row['model']:<24} macro_f1={metrics['macro_f1']:.3f} "
            f"overall_f1={metrics['overall']['f1']:.3f} "
            f"recall={metrics['overall']['recall']:.3f} "
            f"p95_ms={latency['p95_ms']} "
            f"delta_f1={delta['macro_f1']:+.3f}"
        )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
