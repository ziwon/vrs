"""Stable, versioned evaluation report schema.

The report object is intentionally decoupled from the scoring internals so the
JSON emitted by ``scripts/eval.py`` stays stable even as the harness gains new
dataset adapters or scoring modes.
"""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .harness import HarnessResult
from .schemas import ClassMetrics, RunScore

SCHEMA_VERSION = "vrs.eval.report.v1"


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _slug(value: str | None, *, fallback: str) -> str:
    raw = (value or "").strip().lower()
    raw = raw.split("/")[-1]
    raw = raw.removesuffix(".pt").removesuffix(".engine")
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or fallback


def _utc_timestamp(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ReportClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @classmethod
    def from_class_metrics(cls, metrics: ClassMetrics) -> ReportClassMetrics:
        return cls(
            tp=int(metrics.tp),
            fp=int(metrics.fp),
            fn=int(metrics.fn),
            precision=_round_metric(metrics.precision),
            recall=_round_metric(metrics.recall),
            f1=_round_metric(metrics.f1),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportClassMetrics:
        return cls(
            tp=int(data.get("tp", 0)),
            fp=int(data.get("fp", 0)),
            fn=int(data.get("fn", 0)),
            precision=float(data.get("precision", 0.0)),
            recall=float(data.get("recall", 0.0)),
            f1=float(data.get("f1", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "precision": _round_metric(self.precision),
            "recall": _round_metric(self.recall),
            "f1": _round_metric(self.f1),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class ReportMetrics:
    overall: ReportClassMetrics = field(default_factory=ReportClassMetrics)
    per_class: dict[str, ReportClassMetrics] = field(default_factory=dict)

    @classmethod
    def from_run_score(cls, score: RunScore) -> ReportMetrics:
        return cls(
            overall=ReportClassMetrics.from_class_metrics(score.overall()),
            per_class={
                name: ReportClassMetrics.from_class_metrics(score.per_class[name])
                for name in sorted(score.per_class)
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportMetrics:
        return cls(
            overall=ReportClassMetrics.from_dict(data.get("overall", {})),
            per_class={
                name: ReportClassMetrics.from_dict(metrics)
                for name, metrics in sorted((data.get("per_class") or {}).items())
            },
        )

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "per_class": {name: self.per_class[name].to_dict() for name in sorted(self.per_class)},
        }


@dataclass(frozen=True)
class ReportLatency:
    detector_p50_ms: float | None = None
    detector_p95_ms: float | None = None
    verifier_p50_ms: float | None = None
    verifier_p95_ms: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportLatency:
        return cls(
            detector_p50_ms=_round_metric(data.get("detector_p50_ms")),
            detector_p95_ms=_round_metric(data.get("detector_p95_ms")),
            verifier_p50_ms=_round_metric(data.get("verifier_p50_ms")),
            verifier_p95_ms=_round_metric(data.get("verifier_p95_ms")),
        )

    def to_dict(self) -> dict:
        return {
            "detector_p50_ms": _round_metric(self.detector_p50_ms),
            "detector_p95_ms": _round_metric(self.detector_p95_ms),
            "verifier_p50_ms": _round_metric(self.verifier_p50_ms),
            "verifier_p95_ms": _round_metric(self.verifier_p95_ms),
        }


@dataclass(frozen=True)
class ReportRuntime:
    python: str | None = None
    torch: str | None = None
    cuda: str | None = None
    gpu_name: str | None = None
    peak_vram_mb: float | None = None

    @classmethod
    def current(cls) -> ReportRuntime:
        return cls(python=platform.python_version())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportRuntime:
        return cls(
            python=data.get("python"),
            torch=data.get("torch"),
            cuda=data.get("cuda"),
            gpu_name=data.get("gpu_name"),
            peak_vram_mb=_round_metric(data.get("peak_vram_mb")),
        )

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "torch": self.torch,
            "cuda": self.cuda,
            "gpu_name": self.gpu_name,
            "peak_vram_mb": _round_metric(self.peak_vram_mb),
        }


@dataclass(frozen=True)
class ReportQualitySignals:
    verifier_flip_rate: float | None = None
    false_negative_flag_rate: float | None = None
    malformed_json_rate: float | None = None
    queue_drops: int | None = None

    @classmethod
    def from_run_score(cls, score: RunScore) -> ReportQualitySignals:
        return cls(
            verifier_flip_rate=_round_metric(score.flip_rate),
            false_negative_flag_rate=_round_metric(score.fn_flag_rate),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportQualitySignals:
        queue_drops = data.get("queue_drops")
        return cls(
            verifier_flip_rate=_round_metric(data.get("verifier_flip_rate")),
            false_negative_flag_rate=_round_metric(data.get("false_negative_flag_rate")),
            malformed_json_rate=_round_metric(data.get("malformed_json_rate")),
            queue_drops=int(queue_drops) if queue_drops is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "verifier_flip_rate": _round_metric(self.verifier_flip_rate),
            "false_negative_flag_rate": _round_metric(self.false_negative_flag_rate),
            "malformed_json_rate": _round_metric(self.malformed_json_rate),
            "queue_drops": self.queue_drops,
        }


@dataclass(frozen=True)
class ReportModel:
    backend: str
    model: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportModel:
        return cls(
            backend=str(data.get("backend", "")),
            model=str(data.get("model", "")),
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
        }


@dataclass(frozen=True)
class ReportModels:
    detector: ReportModel | None = None
    verifier: ReportModel | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ReportModels:
        det_cfg = config.get("detector") or {}
        ver_cfg = config.get("verifier") or {}

        detector = None
        if det_cfg:
            detector = ReportModel(
                backend=str(det_cfg.get("backend", "unknown")),
                model=str(det_cfg.get("model", "unknown")),
            )

        verifier = None
        if ver_cfg and ver_cfg.get("enabled", True):
            verifier = ReportModel(
                backend=str(ver_cfg.get("backend", "unknown")),
                model=str(ver_cfg.get("model_id", "unknown")),
            )

        return cls(detector=detector, verifier=verifier)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportModels:
        detector = data.get("detector")
        verifier = data.get("verifier")
        return cls(
            detector=ReportModel.from_dict(detector) if isinstance(detector, Mapping) else None,
            verifier=ReportModel.from_dict(verifier) if isinstance(verifier, Mapping) else None,
        )

    def to_dict(self) -> dict:
        return {
            "detector": self.detector.to_dict() if self.detector is not None else None,
            "verifier": self.verifier.to_dict() if self.verifier is not None else None,
        }


@dataclass(frozen=True)
class ReportRun:
    run_id: str
    created_at: str
    dataset: str
    mode: str
    policy_path: str
    config_path: str

    @classmethod
    def build(
        cls,
        *,
        dataset: str | Path,
        policy_path: str | Path,
        config_path: str | Path,
        models: ReportModels,
        verifier_enabled: bool,
        created_at: datetime | None = None,
        run_id: str | None = None,
    ) -> ReportRun:
        created = created_at or datetime.now(UTC)
        detector_slug = _slug(
            models.detector.model if models.detector else None, fallback="detector"
        )
        verifier_slug = _slug(
            models.verifier.model if models.verifier else None, fallback="detector-only"
        )
        dataset_name = Path(dataset).name
        return cls(
            run_id=run_id
            or f"{created.date().isoformat()}-{_slug(dataset_name, fallback='dataset')}-{detector_slug}-{verifier_slug}",
            created_at=_utc_timestamp(created),
            dataset=dataset_name,
            mode="full_cascade" if verifier_enabled else "detector_only",
            policy_path=str(policy_path),
            config_path=str(config_path),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportRun:
        return cls(
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at", "")),
            dataset=str(data.get("dataset", "")),
            mode=str(data.get("mode", "")),
            policy_path=str(data.get("policy_path", "")),
            config_path=str(data.get("config_path", "")),
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "mode": self.mode,
            "policy_path": self.policy_path,
            "config_path": self.config_path,
        }


@dataclass(frozen=True)
class PerVideoReport:
    video: str
    metrics: ReportMetrics
    quality_signals: ReportQualitySignals

    @classmethod
    def from_run_score(cls, video: str | Path, score: RunScore) -> PerVideoReport:
        return cls(
            video=str(video),
            metrics=ReportMetrics.from_run_score(score),
            quality_signals=ReportQualitySignals.from_run_score(score),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerVideoReport:
        return cls(
            video=str(data.get("video", "")),
            metrics=ReportMetrics.from_dict(data.get("metrics", {})),
            quality_signals=ReportQualitySignals.from_dict(data.get("quality_signals", {})),
        )

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "metrics": self.metrics.to_dict(),
            "quality_signals": self.quality_signals.to_dict(),
        }


@dataclass(frozen=True)
class EvalReport:
    schema_version: str
    run: ReportRun
    models: ReportModels
    metrics: ReportMetrics
    latency: ReportLatency
    runtime: ReportRuntime
    quality_signals: ReportQualitySignals
    per_video: Sequence[PerVideoReport] = field(default_factory=tuple)

    @classmethod
    def from_harness_result(
        cls,
        result: HarnessResult,
        *,
        dataset: str | Path,
        config_path: str | Path,
        policy_path: str | Path,
        config: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
        run_id: str | None = None,
    ) -> EvalReport:
        cfg = dict(config or {})
        models = ReportModels.from_config(cfg)
        verifier_enabled = bool((cfg.get("verifier") or {}).get("enabled", True))
        per_video = tuple(
            PerVideoReport.from_run_score(path, score)
            for path, score in sorted(result.per_video, key=lambda item: str(item[0]))
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            run=ReportRun.build(
                dataset=dataset,
                policy_path=policy_path,
                config_path=config_path,
                models=models,
                verifier_enabled=verifier_enabled,
                created_at=created_at,
                run_id=run_id,
            ),
            models=models,
            metrics=ReportMetrics.from_run_score(result.aggregate),
            latency=ReportLatency(),
            runtime=ReportRuntime.current(),
            quality_signals=ReportQualitySignals.from_run_score(result.aggregate),
            per_video=per_video,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalReport:
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            run=ReportRun.from_dict(data.get("run", {})),
            models=ReportModels.from_dict(data.get("models", {})),
            metrics=ReportMetrics.from_dict(data.get("metrics", {})),
            latency=ReportLatency.from_dict(data.get("latency", {})),
            runtime=ReportRuntime.from_dict(data.get("runtime", {})),
            quality_signals=ReportQualitySignals.from_dict(data.get("quality_signals", {})),
            per_video=tuple(PerVideoReport.from_dict(item) for item in data.get("per_video", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "models": self.models.to_dict(),
            "metrics": self.metrics.to_dict(),
            "latency": self.latency.to_dict(),
            "runtime": self.runtime.to_dict(),
            "quality_signals": self.quality_signals.to_dict(),
            "per_video": [entry.to_dict() for entry in self.per_video],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")
