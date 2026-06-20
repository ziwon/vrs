"""Sweep D-Fire prompt banks, detector models, and thresholds.

This is the next calibration layer after threshold-only tuning: each prompt
bank is evaluated with each detector model, then cached detections are rescored
across the threshold grid. The output is a policy/config pair that can be used
by the normal D-Fire eval path and reviewed before any RTSP policy promotion.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_dfire_thresholds import (
    CachedItem,
    choose_best,
    config_with_conf_floor,
    parse_thresholds,
    score_cached_items,
    summarize_score,
    threshold_grid,
)
from vrs import setup_logging
from vrs.eval import config_for_eval_mode
from vrs.eval.datasets import DFireDataset
from vrs.eval.harness import _detections_to_alerts, _load_image
from vrs.policy import WatchItem, WatchPolicy
from vrs.schemas import Frame
from vrs.triage import YOLOEConfig, build_detector

SCHEMA_VERSION = "vrs.eval.dfire_prompt_model_sweep.v1"

DEFAULT_PROMPT_SETS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "prompts": {
            "fire": ["fire", "open flame", "burning object"],
            "smoke": ["smoke", "smoke cloud", "billowing smoke"],
        },
    },
    {
        "name": "flame_smoke_visual",
        "prompts": {
            "fire": [
                "flame",
                "flames",
                "visible flames",
                "open flames",
                "active fire",
                "burning material",
            ],
            "smoke": [
                "smoke",
                "smoke plume",
                "gray smoke",
                "black smoke",
                "white smoke",
                "smoke cloud",
            ],
        },
    },
    {
        "name": "incident_context",
        "prompts": {
            "fire": [
                "building fire",
                "vehicle fire",
                "wildfire",
                "fire on the ground",
                "burning flames",
            ],
            "smoke": [
                "smoke from fire",
                "dark smoke",
                "thick smoke",
                "smoke plume",
                "fire smoke",
            ],
        },
    },
    {
        "name": "minimal",
        "prompts": {
            "fire": ["flame"],
            "smoke": ["smoke"],
        },
    },
]


def parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def load_prompt_sets(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return deepcopy(DEFAULT_PROMPT_SETS)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    prompt_sets = raw.get("prompt_sets") if isinstance(raw, dict) else raw
    if not isinstance(prompt_sets, list) or not prompt_sets:
        raise ValueError(f"{path}: expected a non-empty prompt_sets list")
    out = []
    for item in prompt_sets:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: prompt set entries must be mappings")
        name = str(item.get("name", "")).strip()
        prompts = item.get("prompts")
        if not name or not isinstance(prompts, dict):
            raise ValueError(f"{path}: prompt set entries need name and prompts")
        normalized = {}
        for cls, cls_prompts in prompts.items():
            if isinstance(cls_prompts, str):
                cls_prompts = [cls_prompts]
            normalized[str(cls)] = [
                str(prompt).strip() for prompt in cls_prompts if str(prompt).strip()
            ]
            if not normalized[str(cls)]:
                raise ValueError(f"{path}: prompt set {name!r} has empty prompts for {cls!r}")
        out.append({"name": name, "prompts": normalized})
    return out


def load_policy_yaml(policy_path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw.get("watch"), list) or not raw["watch"]:
        raise ValueError(f"{policy_path}: 'watch' key is empty or missing")
    return raw


def policy_yaml_with_prompts_and_thresholds(
    policy_yaml: dict[str, Any],
    prompt_set: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    raw = deepcopy(policy_yaml)
    prompts_by_class = prompt_set["prompts"]
    for item in raw["watch"]:
        name = str(item.get("name", "")).strip()
        if name in prompts_by_class:
            item["detector"] = list(prompts_by_class[name])
        if name in thresholds:
            item["min_score"] = float(thresholds[name])
    return raw


def watch_policy_for_detector(
    policy_yaml: dict[str, Any],
    prompt_set: dict[str, Any],
    *,
    low_threshold: float,
) -> WatchPolicy:
    raw = policy_yaml_with_prompts_and_thresholds(
        policy_yaml,
        prompt_set,
        {name: low_threshold for name in prompt_set["prompts"]},
    )
    items = []
    for item in raw["watch"]:
        detector = item.get("detector") or [item["name"]]
        if isinstance(detector, str):
            detector = [detector]
        items.append(
            WatchItem(
                name=str(item["name"]),
                detector_prompts=[str(prompt) for prompt in detector],
                verifier_prompt=str(item.get("verifier", item["name"])),
                severity=str(item.get("severity", "medium")),
                min_score=min(float(item.get("min_score", low_threshold)), low_threshold),
                min_persist_frames=int(item.get("min_persist_frames", 1)),
                verifier_window_s=(
                    float(item["verifier_window_s"])
                    if item.get("verifier_window_s") is not None
                    else None
                ),
            )
        )
    return WatchPolicy(items)


def config_with_model_and_thresholds(
    config: dict[str, Any],
    *,
    model: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    cfg = config_with_conf_floor(config, thresholds)
    detector = dict(cfg.get("detector") or {})
    detector["model"] = model
    cfg["detector"] = detector
    return cfg


def evaluate_candidate(
    *,
    dataset_root: str | Path,
    base_config: dict[str, Any],
    policy_yaml: dict[str, Any],
    prompt_set: dict[str, Any],
    model: str,
    classes: list[str],
    thresholds: list[float],
    bbox_iou_threshold: float | None,
) -> dict[str, Any]:
    low_threshold = min(thresholds)
    config = config_with_model_and_thresholds(
        base_config,
        model=model,
        thresholds={name: low_threshold for name in classes},
    )
    det_cfg = dict(config["detector"])
    detector_policy = watch_policy_for_detector(
        policy_yaml,
        prompt_set,
        low_threshold=low_threshold,
    )
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

    cached_items = []
    for item in DFireDataset(dataset_root):
        image = _load_image(item.video_path)
        detections = detector(Frame(index=0, pts_s=0.0, image=image))
        cached_items.append(
            CachedItem(
                image_path=item.video_path,
                events=item.events,
                alerts=_detections_to_alerts(detections, image),
            )
        )

    threshold_results = []
    for candidate_thresholds in threshold_grid(classes, thresholds):
        score = score_cached_items(
            cached_items,
            candidate_thresholds,
            bbox_iou_threshold=bbox_iou_threshold,
        )
        threshold_results.append(
            {
                "thresholds": candidate_thresholds,
                "metrics": summarize_score(score, classes),
            }
        )

    return {
        "model": model,
        "prompt_set": prompt_set["name"],
        "prompts": prompt_set["prompts"],
        "best": choose_best(threshold_results, "macro_f1"),
        "threshold_results": sorted(
            threshold_results,
            key=lambda row: (
                row["metrics"]["macro_f1"],
                row["metrics"]["overall"]["recall"],
                row["metrics"]["overall"]["precision"],
            ),
            reverse=True,
        ),
    }


def choose_best_candidate(results: list[dict[str, Any]], optimize: str) -> dict[str, Any]:
    rows = []
    for row in results:
        best = dict(row["best"])
        best["model"] = row["model"]
        best["prompt_set"] = row["prompt_set"]
        best["prompts"] = row["prompts"]
        rows.append(best)
    return choose_best(rows, optimize)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep D-Fire prompt banks and detector models")
    parser.add_argument("--dataset", required=True, help="D-Fire images/labels root")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--policy", default="configs/policies/dfire_eval.yaml")
    parser.add_argument("--out", default="runs/eval-dfire-prompt-sweep")
    parser.add_argument("--classes", default="fire,smoke")
    parser.add_argument("--prompt-sets", default=None, help="optional YAML prompt_sets file")
    parser.add_argument("--models", default="yoloe-11s-seg.pt,yoloe-11l-seg.pt")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="comma-separated min_score candidates",
    )
    parser.add_argument("--bbox-iou-threshold", type=float, default=None)
    parser.add_argument(
        "--optimize",
        choices=("macro_f1", "overall_f1", "overall_recall"),
        default="macro_f1",
    )
    parser.add_argument("--best-policy", default=None)
    parser.add_argument("--best-config", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    from vrs.pipeline import load_config

    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class")
    thresholds = parse_thresholds(args.thresholds)
    models = parse_models(args.models)
    prompt_sets = load_prompt_sets(args.prompt_sets)

    base_config = config_for_eval_mode(
        load_config(args.config, verifier_enabled=False),
        "detector_only",
    )
    policy_yaml = load_policy_yaml(args.policy)

    results = []
    for model in models:
        for prompt_set in prompt_sets:
            print(f"running model={model} prompt_set={prompt_set['name']}")
            results.append(
                evaluate_candidate(
                    dataset_root=args.dataset,
                    base_config=base_config,
                    policy_yaml=policy_yaml,
                    prompt_set=prompt_set,
                    model=model,
                    classes=classes,
                    thresholds=thresholds,
                    bbox_iou_threshold=args.bbox_iou_threshold,
                )
            )

    best = choose_best_candidate(results, args.optimize)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_policy_path = Path(args.best_policy) if args.best_policy else out_dir / "best_policy.yaml"
    best_config_path = Path(args.best_config) if args.best_config else out_dir / "best_config.yaml"

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "config_path": str(args.config),
        "policy_path": str(args.policy),
        "classes": classes,
        "models": models,
        "prompt_sets": prompt_sets,
        "threshold_candidates": thresholds,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "optimize": args.optimize,
        "best_policy_path": str(best_policy_path),
        "best_config_path": str(best_config_path),
        "best": best,
        "results": sorted(
            results,
            key=lambda row: (
                row["best"]["metrics"]["macro_f1"],
                row["best"]["metrics"]["overall"]["recall"],
                row["best"]["metrics"]["overall"]["precision"],
            ),
            reverse=True,
        ),
    }
    (out_dir / "sweep.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    best_policy = policy_yaml_with_prompts_and_thresholds(
        policy_yaml,
        {"name": best["prompt_set"], "prompts": best["prompts"]},
        best["thresholds"],
    )
    best_policy_path.parent.mkdir(parents=True, exist_ok=True)
    best_policy_path.write_text(yaml.safe_dump(best_policy, sort_keys=False), encoding="utf-8")

    best_config = config_with_model_and_thresholds(
        base_config,
        model=best["model"],
        thresholds=best["thresholds"],
    )
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")

    metrics = best["metrics"]
    print("\n-- best prompt/model -----------------------")
    print(f"model={best['model']} prompt_set={best['prompt_set']}")
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
