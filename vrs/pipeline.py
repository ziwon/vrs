"""End-to-end cascade: ingest → YOLOE → event-state → Cosmos verifier → sinks.

Glue code only — every component is configured via YAML and lives in its own
module so this file stays short.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .ingest import StreamReader
from .policy import WatchPolicy, load_watch_policy
from .runtime import CosmosConfig, CosmosReason2
from .schemas import VerifiedAlert
from .sinks import JsonlSink, VideoAnnotator
from .triage import EventStateQueue, YOLOEConfig, YOLOEDetector
from .verifier import AlertVerifier

logger = logging.getLogger(__name__)


def _require_key(cfg: dict, section: str, key: str, path: str) -> None:
    sec = cfg.get(section)
    if not isinstance(sec, dict) or key not in sec:
        raise ValueError(f"{path}: missing required key '{section}.{key}'")


def _validate_config(cfg: dict, path: str = "<config>") -> None:
    """Check that required sections and keys are present."""
    for section in ("ingest", "detector", "event_state", "verifier", "sink"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"{path}: missing required section '{section}'")
    _require_key(cfg, "ingest", "target_fps", path)
    _require_key(cfg, "detector", "model", path)
    _require_key(cfg, "event_state", "window", path)
    if cfg.get("verifier", {}).get("enabled", True):
        _require_key(cfg, "verifier", "model_id", path)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg or {}, str(path))
    return cfg


class VRSPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        policy: WatchPolicy,
        out_dir: str | Path,
    ):
        self.cfg = config
        self.policy = policy
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        det_cfg = config["detector"]
        ing_cfg = config["ingest"]
        es_cfg = config["event_state"]
        ver_cfg = config["verifier"]

        # --- fast path ---
        self.detector = YOLOEDetector(
            YOLOEConfig(
                model=det_cfg["model"],
                device=det_cfg.get("device", "cuda"),
                imgsz=int(det_cfg.get("imgsz", 640)),
                conf_floor=float(det_cfg.get("conf_floor", 0.20)),
                iou=float(det_cfg.get("iou", 0.50)),
                half=bool(det_cfg.get("half", True)),
            ),
            policy,
        )
        self.event_state = EventStateQueue(
            policy=policy,
            window=int(es_cfg["window"]),
            cooldown_s=float(es_cfg.get("cooldown_s", 10.0)),
            keyframes=int(ver_cfg.get("keyframes", 6)) if ver_cfg.get("enabled", True) else 6,
            context_window_s=float(ver_cfg.get("context_window_s", 3.0)),
            target_fps=float(ing_cfg["target_fps"]),
        )

        # --- slow path (lazy: only spin up the VLM if enabled) ---
        self.verifier: Optional[AlertVerifier] = None
        if ver_cfg.get("enabled", True):
            cosmos = CosmosReason2(
                CosmosConfig(
                    model_id=ver_cfg["model_id"],
                    dtype=ver_cfg.get("dtype", "bf16"),
                    device=ver_cfg.get("device", "cuda"),
                    max_new_tokens=int(ver_cfg.get("max_new_tokens", 1024)),
                    temperature=float(ver_cfg.get("temperature", 0.2)),
                    clip_fps=int(ver_cfg.get("clip_fps", 4)),
                )
            )
            self.verifier = AlertVerifier(
                cosmos=cosmos,
                policy=policy,
                request_bbox=bool(ver_cfg.get("request_bbox", True)),
                request_trajectory=bool(ver_cfg.get("request_trajectory", True)),
                clip_fps=int(ver_cfg.get("clip_fps", 4)),
            )

    # ---- main loop --------------------------------------------------

    def run(self, source: str) -> None:
        ing_cfg = self.cfg["ingest"]
        sink_cfg = self.cfg["sink"]

        reader = StreamReader(
            source=source,
            target_fps=float(ing_cfg["target_fps"]),
            roi_polygon=ing_cfg.get("roi_polygon"),
        )

        jsonl_path = self.out_dir / sink_cfg.get("jsonl", "alerts.jsonl")
        annotator: Optional[VideoAnnotator] = None
        if sink_cfg.get("write_annotated", True):
            annotator = VideoAnnotator(
                self.out_dir / sink_cfg.get("annotated_mp4", "annotated.mp4"),
                fps=float(ing_cfg["target_fps"]),
            )

        try:
            with JsonlSink(jsonl_path) as jsonl:
                for frame in reader:
                    detections = self.detector(frame)
                    candidates = self.event_state.step(frame, detections)

                    for cand in candidates:
                        if self.verifier is not None:
                            verified = self.verifier.verify(cand)
                        else:
                            verified = VerifiedAlert(
                                candidate=cand,
                                true_alert=True,
                                confidence=1.0,
                                false_negative_class=None,
                                rationale="verifier disabled",
                            )
                        jsonl.write(verified)
                        if annotator is not None:
                            annotator.note_alert(verified)
                        self._log(verified)

                    if annotator is not None:
                        annotator.write(frame, detections)
        finally:
            if annotator is not None:
                annotator.close()

    @staticmethod
    def _log(v: VerifiedAlert) -> None:
        tag = "TRUE " if v.true_alert else "FALSE"
        extra = f"  (fn={v.false_negative_class})" if v.false_negative_class else ""
        logger.info(
            "[%s] t=%7.2fs  class=%-10s  sev=%-8s  conf=%.2f%s   -- %s",
            tag, v.candidate.peak_pts_s, v.candidate.class_name,
            v.candidate.severity, v.confidence, extra, v.rationale,
        )


def build_pipeline(
    config_path: str | Path,
    policy_path: str | Path,
    out_dir: str | Path,
) -> VRSPipeline:
    cfg = load_config(config_path)
    policy = load_watch_policy(policy_path)
    return VRSPipeline(cfg, policy, out_dir)
