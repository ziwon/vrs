"""End-to-end cascade: ingest → YOLOE → event-state → VLM verifier → sinks.

Glue code only — every component is configured via YAML and lives in its own
module so this file stays short.
"""

from __future__ import annotations

import logging
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from .calibration import build_calibrator
from .ingest import StreamReader
from .observability import build_metrics
from .policy import WatchPolicy, load_watch_policy
from .policy.hot_reload import PolicyReloader
from .privacy import build_face_detector
from .runtime import VLMConfig, build_vlm_backend
from .schemas import VerifiedAlert
from .sinks import EventThumbnailSink, JsonlSink, VideoAnnotator
from .triage import EventStateQueue, YOLOEConfig, build_detector, build_tracker
from .verifier import AlertVerifier

logger = logging.getLogger(__name__)

# Compatibility for external tests/tools that monkeypatch the old pipeline
# factory symbol. New construction goes through build_vlm_backend.
build_cosmos_backend = build_vlm_backend


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


def load_config(path: str | Path, *, verifier_enabled: bool | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if verifier_enabled is not None:
        cfg = dict(cfg or {})
        verifier = dict(cfg.get("verifier") or {})
        verifier["enabled"] = verifier_enabled
        cfg["verifier"] = verifier
    _validate_config(cfg or {}, str(path))
    return cfg


class VRSPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        policy: WatchPolicy,
        out_dir: str | Path,
        policy_path: str | Path | None = None,
    ):
        self.cfg = config
        self.policy = policy
        self.policy_path = Path(policy_path) if policy_path is not None else None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = build_metrics(config)
        self.detector_latencies_ms: list[float] = []
        self.verifier_latencies_ms: list[float] = []
        self.verifier_tokens_per_second: list[float] = []

        det_cfg = config["detector"]
        ing_cfg = config["ingest"]
        es_cfg = config["event_state"]
        ver_cfg = config["verifier"]
        self._verifier_backend = str(ver_cfg.get("backend", "transformers"))
        verifier_first = self._verifier_backend == "vllm"
        self.verifier: AlertVerifier | None = None

        # vLLM manages its own CUDA worker processes. Initializing it before
        # Ultralytics avoids worker startup stalls after the detector creates a
        # CUDA context in the parent process.
        if verifier_first:
            self.verifier = self._build_verifier(ver_cfg, policy)

        # --- fast path ---
        self.detector = build_detector(
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
        self.tracker = build_tracker(config.get("tracker"))
        self.event_state = EventStateQueue(
            policy=policy,
            window=int(es_cfg["window"]),
            cooldown_s=float(es_cfg.get("cooldown_s", 10.0)),
            keyframes=int(ver_cfg.get("keyframes", 6)) if ver_cfg.get("enabled", True) else 6,
            context_window_s=float(ver_cfg.get("context_window_s", 3.0)),
            target_fps=float(ing_cfg["target_fps"]),
        )

        # --- slow path (lazy: only spin up the VLM if enabled) ---
        if not verifier_first:
            self.verifier = self._build_verifier(ver_cfg, policy)

        # --- calibration (stage A, log-only) ---
        self.calibrator = build_calibrator(config.get("calibration"), policy, self.out_dir)
        reload_cfg = config.get("policy_reload") or {}
        self._policy_reloader = (
            PolicyReloader(self.policy_path, policy)
            if self.policy_path is not None and bool(reload_cfg.get("enabled", False))
            else None
        )
        self._policy_reload_interval_s = float(reload_cfg.get("poll_interval_s", 1.0))
        self._policy_reload_requested = False
        self._last_policy_reload_check = 0.0

    @staticmethod
    def _build_verifier(ver_cfg: dict[str, Any], policy: WatchPolicy) -> AlertVerifier | None:
        if not ver_cfg.get("enabled", True):
            return None

        vlm = build_vlm_backend(
            VLMConfig(
                model_id=ver_cfg["model_id"],
                dtype=ver_cfg.get("dtype", "bf16"),
                device=ver_cfg.get("device", "cuda"),
                max_new_tokens=int(ver_cfg.get("max_new_tokens", 1024)),
                temperature=float(ver_cfg.get("temperature", 0.2)),
                clip_fps=int(ver_cfg.get("clip_fps", 4)),
                base_url=ver_cfg.get("base_url"),
                api_key_env=ver_cfg.get("api_key_env"),
                timeout_s=float(ver_cfg.get("timeout_s", 60.0)),
                max_frame_width=(
                    int(ver_cfg["max_frame_width"])
                    if ver_cfg.get("max_frame_width") is not None
                    else None
                ),
                max_model_len=(
                    int(ver_cfg["max_model_len"])
                    if ver_cfg.get("max_model_len") is not None
                    else None
                ),
                gpu_memory_utilization=(
                    float(ver_cfg["gpu_memory_utilization"])
                    if ver_cfg.get("gpu_memory_utilization") is not None
                    else None
                ),
                tokenizer_id=ver_cfg.get("tokenizer_id"),
                guided_decoding_backend=ver_cfg.get("guided_decoding_backend"),
                draft_model_id=ver_cfg.get("draft_model_id"),
                draft_engine_dir=ver_cfg.get("draft_engine_dir"),
                speculative_tokens=(
                    int(ver_cfg["speculative_tokens"])
                    if ver_cfg.get("speculative_tokens") is not None
                    else None
                ),
                trtllm_extra_llm_kwargs=ver_cfg.get("trtllm_extra_llm_kwargs"),
            ),
            backend=ver_cfg.get("backend", "transformers"),
        )
        return AlertVerifier(
            vlm=vlm,
            policy=policy,
            request_bbox=bool(ver_cfg.get("request_bbox", True)),
            request_trajectory=bool(ver_cfg.get("request_trajectory", True)),
            clip_fps=int(ver_cfg.get("clip_fps", 4)),
            failure_policy=ver_cfg.get("failure_policy"),
        )

    # ---- main loop --------------------------------------------------

    def run(self, source: str) -> None:
        ing_cfg = self.cfg["ingest"]
        sink_cfg = self.cfg["sink"]
        audit_cfg = self.cfg.get("audit")

        reader = StreamReader(
            source=source,
            target_fps=float(ing_cfg["target_fps"]),
            roi_polygon=ing_cfg.get("roi_polygon"),
        )

        previous_sighup = self._install_sighup_handler()

        jsonl_path = self.out_dir / sink_cfg.get("jsonl", "alerts.jsonl")
        privacy_cfg = self.cfg.get("privacy") or {}

        def privacy_failure_cb(backend: str, exc: Exception) -> None:
            self.metrics.inc_privacy_setup_failures(backend)

        thumbnail_sink: EventThumbnailSink | None = None
        if sink_cfg.get("write_thumbnails", True):
            thumbnail_sink = EventThumbnailSink(
                self.out_dir,
                dir_name=sink_cfg.get("thumbnails_dir", "thumbnails"),
                ext=sink_cfg.get("thumbnail_ext", "jpg"),
                quality=int(sink_cfg.get("thumbnail_quality", 90)),
                face_detector=build_face_detector(
                    privacy_cfg,
                    setup_failure_callback=privacy_failure_cb,
                ),
                blur_kernel=int(privacy_cfg.get("blur_kernel", 31)),
                blur_margin_pct=float(privacy_cfg.get("margin_pct", 0.15)),
            )
        annotator: VideoAnnotator | None = None
        if sink_cfg.get("write_annotated", False):
            annotator = VideoAnnotator(
                self.out_dir / sink_cfg.get("annotated_mp4", "annotated.mp4"),
                fps=float(ing_cfg["target_fps"]),
                face_detector=build_face_detector(
                    privacy_cfg,
                    setup_failure_callback=privacy_failure_cb,
                ),
                blur_kernel=int(privacy_cfg.get("blur_kernel", 31)),
                blur_margin_pct=float(privacy_cfg.get("margin_pct", 0.15)),
            )

        try:
            with JsonlSink(jsonl_path, audit=audit_cfg) as jsonl:
                for frame in reader:
                    self._maybe_reload_policy()
                    detector_t0 = time.perf_counter()
                    try:
                        detections = self.detector(frame)
                    finally:
                        detector_elapsed = time.perf_counter() - detector_t0
                        self.metrics.observe_detector_latency(detector_elapsed)
                        self.detector_latencies_ms.append(detector_elapsed * 1000.0)
                    detections = self.tracker.update(detections, frame.index)
                    candidates = self.event_state.step(frame, detections)

                    for cand in candidates:
                        self.metrics.inc_candidates("default", cand.class_name)
                        if self.verifier is not None:
                            verifier_t0 = time.perf_counter()
                            try:
                                verified = self.verifier.verify(cand)
                            except Exception:
                                self.metrics.inc_verifier_errors(self._verifier_backend)
                                raise
                            finally:
                                verifier_elapsed = time.perf_counter() - verifier_t0
                                self.metrics.observe_verifier_latency(verifier_elapsed)
                                self.verifier_latencies_ms.append(verifier_elapsed * 1000.0)
                                stats = getattr(
                                    getattr(self.verifier, "vlm", None),
                                    "last_generation_stats",
                                    None,
                                )
                                if isinstance(stats, dict):
                                    tokens_per_second = stats.get("tokens_per_second")
                                    if tokens_per_second is not None:
                                        value = float(tokens_per_second)
                                        self.metrics.observe_verifier_tokens_per_second(
                                            self._verifier_backend,
                                            value,
                                        )
                                        self.verifier_tokens_per_second.append(value)
                        else:
                            verified = VerifiedAlert(
                                candidate=cand,
                                true_alert=True,
                                confidence=1.0,
                                false_negative_class=None,
                                rationale="verifier disabled",
                            )
                        verdict = "true_alert" if verified.true_alert else "false_alert"
                        self.metrics.inc_verified_alerts(
                            "default",
                            verified.candidate.class_name,
                            verdict,
                            verified.candidate.severity,
                        )
                        if thumbnail_sink is not None:
                            try:
                                thumbnail_sink.write(verified)
                            except Exception as e:
                                self.metrics.inc_sink_write_errors("default")
                                logger.warning("thumbnail write failed: %s", e)
                        jsonl.write(verified)
                        if annotator is not None:
                            annotator.note_alert(verified)
                        if self.calibrator is not None:
                            self.calibrator.record("default", verified)
                        self._log(verified)

                    if annotator is not None:
                        annotator.write(frame, detections)
        finally:
            self._restore_sighup_handler(previous_sighup)
            if annotator is not None:
                annotator.close()
            if self.calibrator is not None:
                self.calibrator.close()
            self.metrics.close()

    def close(self) -> None:
        if self.verifier is None:
            return
        vlm = getattr(self.verifier, "vlm", None)
        close = getattr(vlm, "close", None)
        if callable(close):
            close()

    def _install_sighup_handler(self):
        if self._policy_reloader is None or not hasattr(signal, "SIGHUP"):
            return None
        try:
            previous = signal.getsignal(signal.SIGHUP)

            def _handler(signum, frame):
                logger.info("policy reload requested (signal %s)", signum)
                self._policy_reload_requested = True

            signal.signal(signal.SIGHUP, _handler)
            return previous
        except ValueError:
            return None

    def _restore_sighup_handler(self, previous) -> None:
        if previous is None or not hasattr(signal, "SIGHUP"):
            return
        with suppress(ValueError):
            signal.signal(signal.SIGHUP, previous)

    def _maybe_reload_policy(self) -> None:
        if self._policy_reloader is None:
            return
        now = time.monotonic()
        force = self._policy_reload_requested
        if not force and (now - self._last_policy_reload_check) < self._policy_reload_interval_s:
            return
        self._policy_reload_requested = False
        self._last_policy_reload_check = now
        result = self._policy_reloader.maybe_reload(force=force)
        if not result.reloaded:
            return
        new_policy = self._policy_reloader.policy
        self.policy = new_policy
        self.detector.update_policy(new_policy)
        self.event_state.update_policy(new_policy)
        if self.verifier is not None:
            self.verifier.update_policy(new_policy)
        if self.calibrator is not None:
            self.calibrator.policy = new_policy

    @staticmethod
    def _log(v: VerifiedAlert) -> None:
        tag = "TRUE " if v.true_alert else "FALSE"
        extra = f"  (fn={v.false_negative_class})" if v.false_negative_class else ""
        logger.info(
            "[%s] t=%7.2fs  class=%-10s  sev=%-8s  conf=%.2f%s   -- %s",
            tag,
            v.candidate.peak_pts_s,
            v.candidate.class_name,
            v.candidate.severity,
            v.confidence,
            extra,
            v.rationale,
        )


def build_pipeline(
    config_path: str | Path,
    policy_path: str | Path,
    out_dir: str | Path,
) -> VRSPipeline:
    cfg = load_config(config_path)
    policy = load_watch_policy(policy_path)
    return VRSPipeline(cfg, policy, out_dir, policy_path=policy_path)
