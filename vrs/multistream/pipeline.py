"""MultiStreamPipeline — orchestrates N concurrent RTSP / mp4 sources.

One shared YOLOE detector (fast path) and one shared Cosmos-Reason2-2B
verifier (slow path) serve every stream. Bounded queues provide backpressure
so any one noisy camera can't starve the others.
"""
from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..policy import WatchPolicy, load_watch_policy


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config. Duplicated from vrs.pipeline to avoid pulling cv2
    into the import chain of tests that don't exercise the decoder."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
from ..runtime import CosmosConfig, CosmosReason2
from ..triage import YOLOEConfig, YOLOEDetector
from ..verifier import AlertVerifier
from .queues import BoundedQueue, DropPolicy
from .readers import Reader, build_reader
from .workers import DecoderThread, DetectorWorker, SinkWorker, VerifierWorker


# ──────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StreamSpec:
    id: str
    source: str                  # rtsp://... or mp4 path
    roi_polygon: Optional[List] = None

    @staticmethod
    def from_dict(d: dict) -> "StreamSpec":
        return StreamSpec(
            id=str(d["id"]),
            source=str(d.get("rtsp") or d.get("source") or d.get("video")),
            roi_polygon=d.get("roi_polygon"),
        )


def load_multistream_spec(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────────────────────────────

class MultiStreamPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        policy: WatchPolicy,
        streams: List[StreamSpec],
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
        self._detector = YOLOEDetector(
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

        ver_cfg = config["verifier"]
        self._verifier: Optional[AlertVerifier] = None
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
            self._verifier = AlertVerifier(
                cosmos=cosmos,
                policy=policy,
                request_bbox=bool(ver_cfg.get("request_bbox", True)),
                request_trajectory=bool(ver_cfg.get("request_trajectory", True)),
                clip_fps=int(ver_cfg.get("clip_fps", 4)),
            )

        # --- queues ---
        ms_cfg = config.get("multistream") or {}
        self._stop = threading.Event()
        self._frame_q = BoundedQueue(
            maxsize=int(ms_cfg.get("frame_queue_size", max(32, 4 * len(streams)))),
            policy=DropPolicy(ms_cfg.get("frame_drop_policy", "drop_oldest")),
        )
        self._candidate_q = BoundedQueue(
            maxsize=int(ms_cfg.get("verifier_queue_size", 16)),
            policy=DropPolicy(ms_cfg.get("verifier_drop_policy", "drop_oldest")),
        )
        self._sink_qs: Dict[str, BoundedQueue] = {
            s.id: BoundedQueue(maxsize=int(ms_cfg.get("sink_queue_size", 32)),
                               policy=DropPolicy.DROP_OLDEST)
            for s in streams
        }

        # --- worker construction (threads start in .run()) ---
        self._ms_cfg = ms_cfg
        self._ing_cfg = config["ingest"]
        self._es_cfg = config["event_state"]
        self._sink_cfg = config["sink"]

        self._decoders: List[DecoderThread] = []
        self._sinks: List[SinkWorker] = []
        self._detector_worker: Optional[DetectorWorker] = None
        self._verifier_worker: Optional[VerifierWorker] = None

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
                    write_annotated=bool(self._sink_cfg.get("write_annotated", True)),
                    jsonl_name=self._sink_cfg.get("jsonl", "alerts.jsonl"),
                    mp4_name=self._sink_cfg.get("annotated_mp4", "annotated.mp4"),
                    sink_q=self._sink_qs[s.id],
                    stop_event=self._stop,
                )
            )

        # verifier worker
        self._verifier_worker = VerifierWorker(
            verifier=self._verifier,
            candidate_q=self._candidate_q,
            sink_queues=self._sink_qs,
            stop_event=self._stop,
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
                    stop_event=self._stop,
                )
            )

    def run(self, max_runtime_s: Optional[float] = None) -> None:
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
        def _handler(signum, frame):  # noqa: ARG001
            print("\n[multistream] shutting down…")
            self.stop()
        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            # not in main thread — tests call run() from worker threads
            pass

        t0 = time.monotonic()
        try:
            while not self._stop.is_set():
                if max_runtime_s is not None and (time.monotonic() - t0) >= max_runtime_s:
                    break
                # join briefly on decoder threads so we exit naturally when files end
                alive = any(d.is_alive() for d in self._decoders)
                if not alive and self._frame_q.qsize() == 0 and self._candidate_q.qsize() == 0:
                    # everything drained
                    break
                time.sleep(0.2)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._frame_q.close()
        self._candidate_q.close()
        for q in self._sink_qs.values():
            q.close()

        # best-effort join
        for d in self._decoders:
            d.join(timeout=2.0)
        if self._detector_worker:
            self._detector_worker.join(timeout=5.0)
        if self._verifier_worker:
            self._verifier_worker.join(timeout=5.0)
        for w in self._sinks:
            w.join(timeout=2.0)

    # ---- introspection for metrics / tests --------------------------

    def queue_stats(self) -> dict:
        return {
            "frame_q": {"size": self._frame_q.qsize(), "dropped": self._frame_q.puts_dropped},
            "candidate_q": {"size": self._candidate_q.qsize(), "dropped": self._candidate_q.puts_dropped},
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
