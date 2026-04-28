"""MultiStreamPipeline — orchestrates N concurrent RTSP / mp4 sources.

One shared YOLOE detector (fast path) and one shared Cosmos-Reason2-2B
verifier (slow path) serve every stream. Bounded queues provide backpressure
so any one noisy camera can't starve the others.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..calibration import build_calibrator
from ..pipeline import load_config
from ..policy import WatchPolicy, load_watch_policy
from ..runtime import CosmosConfig, build_cosmos_backend
from ..triage import YOLOEConfig, build_detector
from ..verifier import AlertVerifier
from .queues import BoundedQueue, DropPolicy
from .readers import build_reader
from .workers import DecoderThread, DetectorWorker, SinkWorker, VerifierWorker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class StreamSpec:
    id: str
    source: str  # rtsp://... or mp4 path
    roi_polygon: list | None = None

    @staticmethod
    def from_dict(d: dict) -> StreamSpec:
        return StreamSpec(
            id=str(d["id"]),
            source=str(d.get("rtsp") or d.get("source") or d.get("video")),
            roi_polygon=d.get("roi_polygon"),
        )


def _validate_multistream_spec(cfg: dict, path: str = "<streams>") -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    streams = cfg.get("streams")
    if streams is None:
        raise ValueError(f"{path}: missing required top-level 'streams' key")
    if not isinstance(streams, list):
        raise ValueError(f"{path}: 'streams' must be a list")
    for i, item in enumerate(streams):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: streams[{i}] must be a mapping")
        if "id" not in item:
            raise ValueError(f"{path}: streams[{i}] missing required key 'id'")
        if not any(k in item for k in ("rtsp", "source", "video")):
            raise ValueError(
                f"{path}: streams[{i}] must define one of 'rtsp', 'source', or 'video'"
            )


def load_multistream_spec(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _validate_multistream_spec(cfg, str(path))
    return cfg


# ──────────────────────────────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────────────────────────────


class MultiStreamPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        policy: WatchPolicy,
        streams: list[StreamSpec],
        out_dir: str | Path,
    ):
        if not streams:
            raise ValueError("no streams configured")
        seen: set[str] = set()
        for s in streams:
            if s.id in seen:
                raise ValueError(f"duplicate stream id: {s.id!r}")
            seen.add(s.id)

        self.cfg = config
        self.policy = policy
        self.streams = streams
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # --- shared GPU models (created once, owned by their worker threads) ---
        det_cfg = config["detector"]
        self._detector = build_detector(
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

        ver_cfg = config["verifier"]
        self._verifier: AlertVerifier | None = None
        if ver_cfg.get("enabled", True):
            cosmos = build_cosmos_backend(
                CosmosConfig(
                    model_id=ver_cfg["model_id"],
                    dtype=ver_cfg.get("dtype", "bf16"),
                    device=ver_cfg.get("device", "cuda"),
                    max_new_tokens=int(ver_cfg.get("max_new_tokens", 1024)),
                    temperature=float(ver_cfg.get("temperature", 0.2)),
                    clip_fps=int(ver_cfg.get("clip_fps", 4)),
                ),
                backend=ver_cfg.get("backend", "transformers"),
            )
            self._verifier = AlertVerifier(
                cosmos=cosmos,
                policy=policy,
                request_bbox=bool(ver_cfg.get("request_bbox", True)),
                request_trajectory=bool(ver_cfg.get("request_trajectory", True)),
                clip_fps=int(ver_cfg.get("clip_fps", 4)),
                failure_policy=ver_cfg.get("failure_policy"),
            )

        # --- calibration (stage A, log-only) ---
        # Shared across streams — suggestion lines carry stream_id so one
        # JSONL at the run root stays comprehensible.
        self._calibrator = build_calibrator(config.get("calibration"), policy, self.out_dir)

        # --- queues ---
        ms_cfg = config.get("multistream") or {}
        self._stop = threading.Event()
        self._decoder_stop = threading.Event()
        self._shutdown_mode = str(ms_cfg.get("shutdown_mode", "fast")).lower()
        if self._shutdown_mode not in {"fast", "drain"}:
            raise ValueError("multistream.shutdown_mode must be 'fast' or 'drain'")
        self._shutdown_drain_timeout_s = float(ms_cfg.get("shutdown_drain_timeout_s", 10.0))
        if self._shutdown_drain_timeout_s < 0:
            raise ValueError("multistream.shutdown_drain_timeout_s must be >= 0")
        self._frame_q = BoundedQueue(
            maxsize=int(ms_cfg.get("frame_queue_size", max(32, 4 * len(streams)))),
            policy=DropPolicy(ms_cfg.get("frame_drop_policy", "drop_oldest")),
        )
        self._candidate_q = BoundedQueue(
            maxsize=int(ms_cfg.get("verifier_queue_size", 16)),
            policy=DropPolicy(ms_cfg.get("verifier_drop_policy", "drop_oldest")),
        )
        self._sink_qs: dict[str, BoundedQueue] = {
            s.id: BoundedQueue(
                maxsize=int(ms_cfg.get("sink_queue_size", 32)), policy=DropPolicy.DROP_OLDEST
            )
            for s in streams
        }

        # --- worker construction (threads start in .run()) ---
        self._ms_cfg = ms_cfg
        self._ing_cfg = config["ingest"]
        self._es_cfg = config["event_state"]
        self._ver_cfg = config["verifier"]
        self._tracker_cfg = config.get("tracker")
        self._privacy_cfg = config.get("privacy") or {}
        self._sink_cfg = config["sink"]

        self._decoders: list[DecoderThread] = []
        self._sinks: list[SinkWorker] = []
        self._detector_worker: DetectorWorker | None = None
        self._verifier_worker: VerifierWorker | None = None

    # ---- lifecycle --------------------------------------------------

    def _build_threads(self) -> None:
        ms = self._ms_cfg
        ing = self._ing_cfg

        # sink workers (per stream)
        for s in self.streams:
            stream_out = self.out_dir / s.id
            self._sinks.append(
                SinkWorker(
                    stream_id=s.id,
                    out_dir=stream_out,
                    fps=float(ing["target_fps"]),
                    write_annotated=bool(self._sink_cfg.get("write_annotated", False)),
                    write_thumbnails=bool(self._sink_cfg.get("write_thumbnails", True)),
                    jsonl_name=self._sink_cfg.get("jsonl", "alerts.jsonl"),
                    mp4_name=self._sink_cfg.get("annotated_mp4", "annotated.mp4"),
                    thumbnails_dir=self._sink_cfg.get("thumbnails_dir", "thumbnails"),
                    thumbnail_ext=self._sink_cfg.get("thumbnail_ext", "jpg"),
                    thumbnail_quality=int(self._sink_cfg.get("thumbnail_quality", 90)),
                    sink_q=self._sink_qs[s.id],
                    stop_event=self._stop,
                    privacy_cfg=self._privacy_cfg,
                )
            )

        # verifier worker
        self._verifier_worker = VerifierWorker(
            verifier=self._verifier,
            candidate_q=self._candidate_q,
            sink_queues=self._sink_qs,
            stop_event=self._stop,
            calibrator=self._calibrator,
        )

        # detector worker
        self._detector_worker = DetectorWorker(
            detector=self._detector,
            policy=self.policy,
            frame_q=self._frame_q,
            candidate_q=self._candidate_q,
            sink_queues=self._sink_qs,
            stream_ids=[s.id for s in self.streams],
            stop_event=self._stop,
            batch_size=int(ms.get("detector_batch_size", 4)),
            batch_timeout_ms=int(ms.get("detector_batch_timeout_ms", 30)),
            event_state_cfg=self._es_cfg,
            verifier_cfg=self._ver_cfg,
            tracker_cfg=self._tracker_cfg,
            target_fps=float(ing["target_fps"]),
        )

        # decoder threads (per stream)
        backend = ms.get("decoder_backend", "opencv")
        for s in self.streams:
            reader = build_reader(
                backend=backend,
                source=s.source,
                target_fps=float(ing["target_fps"]),
                roi_polygon=s.roi_polygon,
            )
            self._decoders.append(
                DecoderThread(
                    stream_id=s.id,
                    reader=reader,
                    frame_q=self._frame_q,
                    stop_event=self._decoder_stop,
                )
            )

    def run(self, max_runtime_s: float | None = None) -> None:
        """Start all workers and block until stop() or max_runtime_s."""
        self._build_threads()

        # start sinks and GPU workers first so they're ready when frames arrive
        for w in self._sinks:
            w.start()
        assert self._verifier_worker is not None and self._detector_worker is not None
        self._verifier_worker.start()
        self._detector_worker.start()
        for d in self._decoders:
            d.start()

        # graceful SIGINT
        def _handler(signum, frame):
            logger.info("shutting down (signal %s)…", signum)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            # not in main thread — tests call run() from worker threads
            pass

        # Backpressure visibility: sample drop counters on an interval and
        # log a single WARNING naming every queue whose counter grew. 0
        # disables. Default 5 s — chatty enough to catch degradation, quiet
        # enough to stay out of the way on a healthy pipeline.
        drop_log_interval = float(self._ms_cfg.get("drop_log_interval_s", 5.0))
        last_drop_log = time.monotonic()
        last_drop_snapshot = self._drop_counters()

        t0 = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if max_runtime_s is not None and (now - t0) >= max_runtime_s:
                    break

                if drop_log_interval > 0 and (now - last_drop_log) >= drop_log_interval:
                    cur = self._drop_counters()
                    self._log_drop_deltas(last_drop_snapshot, cur)
                    last_drop_snapshot = cur
                    last_drop_log = now

                # join briefly on decoder threads so we exit naturally when files end
                alive = any(d.is_alive() for d in self._decoders)
                if not alive and self._frame_q.qsize() == 0 and self._candidate_q.qsize() == 0:
                    # everything drained
                    break
                time.sleep(0.2)
        finally:
            self.stop()

    def stop(self, mode: str | None = None) -> None:
        shutdown_mode = str(mode or getattr(self, "_shutdown_mode", "fast")).lower()
        if shutdown_mode == "drain":
            self._stop_drain()
        elif shutdown_mode == "fast":
            self._stop_fast()
        else:
            raise ValueError("shutdown mode must be 'fast' or 'drain'")

    def _stop_fast(self) -> None:
        self._decoder_stop_event().set()
        self._stop.set()
        self._frame_q.close()
        self._candidate_q.close()

        # best-effort join — name anything that exceeds its timeout so a hung
        # shutdown is debuggable from the log alone.
        hung: list[str] = []
        for d in self._decoders:
            self._join_or_track_hung(d, timeout=2.0, hung=hung)
        if self._detector_worker:
            self._join_or_track_hung(self._detector_worker, timeout=5.0, hung=hung)
        if self._verifier_worker:
            self._join_or_track_hung(self._verifier_worker, timeout=5.0, hung=hung)

        # Keep sink queues open until detector/verifier workers have stopped:
        # either worker may still be finishing an in-flight frame/candidate and
        # enqueueing its final message after stop() begins. Closing sink queues
        # earlier can turn a graceful shutdown into alert loss.
        for q in self._sink_qs.values():
            q.close()
        for w in self._sinks:
            self._join_or_track_hung(w, timeout=2.0, hung=hung)

        if hung:
            logger.warning(
                "shutdown left %d thread(s) still alive past their join timeout: %s",
                len(hung),
                ", ".join(hung),
            )

        if self._calibrator is not None:
            self._calibrator.close()

    def _stop_drain(self) -> None:
        timeout_s = float(getattr(self, "_shutdown_drain_timeout_s", 10.0))
        deadline = time.monotonic() + timeout_s
        deadline_exceeded = False

        self._decoder_stop_event().set()

        for d in self._decoders:
            if not self._join_until_deadline(d, deadline):
                deadline_exceeded = True

        self._frame_q.close()
        if self._detector_worker and not self._join_until_deadline(self._detector_worker, deadline):
            deadline_exceeded = True

        self._candidate_q.close()
        if self._verifier_worker and not self._join_until_deadline(self._verifier_worker, deadline):
            deadline_exceeded = True

        for q in self._sink_qs.values():
            q.close()
        for w in self._sinks:
            if not self._join_until_deadline(w, deadline):
                deadline_exceeded = True

        if deadline_exceeded:
            self._log_drain_deadline_exceeded(timeout_s)
            self._stop.set()
            self._frame_q.close()
            self._candidate_q.close()
            for q in self._sink_qs.values():
                q.close()
        else:
            self._stop.set()

        if self._calibrator is not None:
            self._calibrator.close()

    @staticmethod
    def _join_or_track_hung(thread: threading.Thread, timeout: float, hung: list[str]) -> None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            hung.append(f"{thread.name}(>{timeout:g}s)")

    @staticmethod
    def _join_until_deadline(thread: threading.Thread, deadline: float) -> bool:
        timeout = max(0.0, deadline - time.monotonic())
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _decoder_stop_event(self) -> threading.Event:
        event = getattr(self, "_decoder_stop", None)
        if event is None:
            event = self._stop
            self._decoder_stop = event
        return event

    def _log_drain_deadline_exceeded(self, timeout_s: float) -> None:
        queue_sizes = self._queue_sizes()
        queue_parts = ", ".join(f"{name}={size}" for name, size in queue_sizes.items())
        alive = self._alive_worker_names()
        alive_part = ", ".join(alive) if alive else "none"
        logger.warning(
            "drain shutdown deadline exceeded after %.2fs; remaining queues: %s; alive workers: %s",
            timeout_s,
            queue_parts,
            alive_part,
        )

    def _queue_sizes(self) -> dict[str, int]:
        out = {
            "frame_q": self._frame_q.qsize(),
            "candidate_q": self._candidate_q.qsize(),
        }
        for sid, q in self._sink_qs.items():
            out[f"sink[{sid}]"] = q.qsize()
        return out

    def _alive_worker_names(self) -> list[str]:
        threads: list[threading.Thread] = []
        threads.extend(getattr(self, "_decoders", []))
        for attr in ("_detector_worker", "_verifier_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                threads.append(worker)
        threads.extend(getattr(self, "_sinks", []))
        return [t.name for t in threads if t.is_alive()]

    # ---- backpressure diagnostics -----------------------------------

    def _drop_counters(self) -> dict[str, int]:
        out: dict[str, int] = {
            "frame_q": self._frame_q.puts_dropped,
            "candidate_q": self._candidate_q.puts_dropped,
        }
        for sid, q in self._sink_qs.items():
            out[f"sink[{sid}]"] = q.puts_dropped
        return out

    @staticmethod
    def _log_drop_deltas(prev: dict[str, int], cur: dict[str, int]) -> None:
        deltas = [
            (name, cur[name] - prev.get(name, 0)) for name in cur if cur[name] > prev.get(name, 0)
        ]
        if not deltas:
            return
        parts = ", ".join(f"{name}+{delta}" for name, delta in deltas)
        logger.warning("queue backpressure: dropped in last window — %s", parts)

    # ---- introspection for metrics / tests --------------------------

    def queue_stats(self) -> dict:
        return {
            "frame_q": {"size": self._frame_q.qsize(), "dropped": self._frame_q.puts_dropped},
            "candidate_q": {
                "size": self._candidate_q.qsize(),
                "dropped": self._candidate_q.puts_dropped,
            },
            "sink_q": {
                sid: {"size": q.qsize(), "dropped": q.puts_dropped}
                for sid, q in self._sink_qs.items()
            },
        }


# ──────────────────────────────────────────────────────────────────────
# builder
# ──────────────────────────────────────────────────────────────────────


def build_multistream_pipeline(
    config_path: str | Path,
    policy_path: str | Path,
    streams_path: str | Path,
    out_dir: str | Path,
) -> MultiStreamPipeline:
    """Load config + policy + a streams manifest, return a ready-to-run pipeline.

    Streams manifest may live in its own YAML, or may be embedded in the main
    config under ``streams:``. The manifest format:

        streams:
          - id: cam01
            rtsp: rtsp://...
            roi_polygon: null
    """
    cfg = load_config(config_path)
    policy = load_watch_policy(policy_path)
    streams_cfg = load_multistream_spec(streams_path) if streams_path else cfg
    raw_streams = streams_cfg.get("streams") or cfg.get("streams") or []
    if not raw_streams:
        raise ValueError("no streams listed — expected a top-level 'streams:' key")
    streams = [StreamSpec.from_dict(d) for d in raw_streams]

    # merge multistream overrides from the streams file into cfg
    if "multistream" in streams_cfg:
        cfg["multistream"] = {**cfg.get("multistream", {}), **streams_cfg["multistream"]}

    return MultiStreamPipeline(cfg, policy, streams, out_dir)
