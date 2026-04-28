"""Evaluation harness — iterate dataset → run cascade → score.

Pipeline-agnostic: the caller supplies a ``pipeline_factory`` that maps a
per-video output directory to something with ``.run(source_path)``. Production
code passes a closure over ``vrs.pipeline.build_pipeline``; tests inject a
stub so scoring can be exercised without GPUs or video decode.

After each run the harness reads ``<video_out>/alerts.jsonl`` and scores it
against the item's ground-truth events. The final ``HarnessResult`` holds
both the per-video breakdown (useful for spotting which clips drag metrics
down) and the aggregate (headline numbers).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .datasets.base import Dataset
from .metrics import aggregate_scores, score_alerts_against_truth
from .schemas import RunScore

logger = logging.getLogger(__name__)


PipelineFactory = Callable[[Path], Any]  # (out_dir) -> something with .run(source)
EvalMode = Literal["full_cascade", "detector_only"]


@dataclass
class HarnessResult:
    aggregate: RunScore
    per_video: list[tuple[Path, RunScore]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "aggregate": self.aggregate.to_dict(),
            "per_video": [{"video": str(p), "score": s.to_dict()} for p, s in self.per_video],
        }


def config_for_eval_mode(config: dict[str, Any], mode: EvalMode) -> dict[str, Any]:
    """Return a copy of ``config`` adjusted for the requested eval mode."""
    if mode not in ("full_cascade", "detector_only"):
        raise ValueError(f"unknown eval mode: {mode!r}")

    cfg = deepcopy(config)
    if mode == "detector_only":
        verifier = cfg.setdefault("verifier", {})
        verifier["enabled"] = False
    return cfg


def evaluate(
    dataset: Dataset,
    pipeline_factory: PipelineFactory,
    out_dir: str | Path,
    *,
    tolerance_s: float = 1.0,
    alerts_filename: str = "alerts.jsonl",
    classes: Iterable[str] | None = None,
) -> HarnessResult:
    """Run the cascade over every item in ``dataset`` and return scored results.

    Args:
        dataset: anything that iterates ``EvalItem`` (e.g. ``LabeledDirDataset``).
        pipeline_factory: ``out_dir -> pipeline``. The harness calls
            ``pipeline.run(str(item.video_path))`` for each item.
        out_dir: parent dir; each item gets its own subdir keyed by
            ``video_path.stem``.
        tolerance_s: temporal slack around ground-truth event windows.
        alerts_filename: name of the JSONL the pipeline writes in its out dir.
        classes: restrict scoring to these class names (``None`` = union of
            classes seen in alerts / events).
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    classes_set = set(classes) if classes is not None else None
    per_video: list[tuple[Path, RunScore]] = []

    for item in dataset:
        video_out = root / item.video_path.stem
        video_out.mkdir(parents=True, exist_ok=True)

        pipeline = pipeline_factory(video_out)
        try:
            pipeline.run(str(item.video_path))
        except Exception as e:
            logger.error("pipeline run failed on %s: %s", item.video_path, e)

        alerts = _load_alerts(video_out / alerts_filename)
        score = score_alerts_against_truth(
            alerts,
            item.events,
            tolerance_s=tolerance_s,
            classes=classes_set,
        )
        per_video.append((item.video_path, score))
        logger.info(
            "scored %s — alerts=%d events=%d per_class=%s flip_rate=%.3f",
            item.video_path.name,
            score.n_alerts_total,
            score.n_events,
            {k: v.to_dict() for k, v in score.per_class.items()},
            score.flip_rate,
        )

    return HarnessResult(
        aggregate=aggregate_scores(s for _, s in per_video),
        per_video=per_video,
    )


def _load_alerts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed alert line in %s: %s", path, e)
    return out
