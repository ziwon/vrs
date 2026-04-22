"""Worker threads: decoder (per-stream) + detector + verifier + sink (per-stream).

Ownership rules:
  * one ``Detector`` instance (ultralytics or tensorrt) lives inside the
    DetectorWorker thread;
    no other thread touches the CUDA model.
  * one ``AlertVerifier`` / verifier backend lives inside the VerifierWorker.
  * per-stream ``EventStateQueue``, ``JsonlSink``, ``EventThumbnailSink``,
    ``VideoAnnotator`` live inside the DetectorWorker (event_state) and the
    per-stream SinkWorker.

All communication is through ``BoundedQueue``s — producers never block the
RTSP read thread when a downstream worker is slow.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..policy import WatchPolicy
from ..schemas import CandidateAlert, Detection, Frame, VerifiedAlert
from .queues import BoundedQueue, DropPolicy

logger = logging.getLogger(__name__)

# Heavy deps (cv2 / ultralytics / transformers) are only imported when
# threads actually run. This keeps the unit tests import-clean on CPU-only
# boxes that have neither OpenCV nor Torch/Ultralytics installed.
if TYPE_CHECKING:
    from ..triage import Detector, EventStateQueue
    from ..verifier import AlertVerifier
    from .readers import Reader


# ──────────────────────────────────────────────────────────────────────
# items that flow between workers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _FrameMsg:
    stream_id: str
    frame: Frame


@dataclass
class _CandidateMsg:
    stream_id: str
    alert: CandidateAlert


@dataclass
class _SinkMsg:
    kind: str                          # "frame" | "alert"
    frame: Optional[Frame] = None
    detections: Optional[List[Detection]] = None
    verified: Optional[VerifiedAlert] = None


# ──────────────────────────────────────────────────────────────────────
# decoder — one thread per RTSP/mp4 source
# ──────────────────────────────────────────────────────────────────────

class DecoderThread(threading.Thread):
    def __init__(
        self,
        stream_id: str,
        reader: Reader,
        frame_q: BoundedQueue,
        stop_event: threading.Event,
    ):
        super().__init__(name=f"decoder[{stream_id}]", daemon=True)
        self.stream_id = stream_id
        self.reader = reader
        self.frame_q = frame_q
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            for frame in self.reader:
                if self.stop_event.is_set():
                    break
                self.frame_q.put(_FrameMsg(self.stream_id, frame))
        except Exception as e:  # noqa: BLE001 — log and exit
            logger.error("decoder[%s] terminated: %s", self.stream_id, e)


# ──────────────────────────────────────────────────────────────────────
# detector — one shared thread, batched YOLOE across streams
# ──────────────────────────────────────────────────────────────────────

class DetectorWorker(threading.Thread):
    def __init__(
        self,
        detector: Detector,
        policy: WatchPolicy,
        frame_q: BoundedQueue,
        candidate_q: BoundedQueue,
        sink_queues: Dict[str, BoundedQueue],
        stream_ids: List[str],
        stop_event: threading.Event,
        batch_size: int = 4,
        batch_timeout_ms: int = 30,
        event_state_cfg: Optional[dict] = None,
        verifier_cfg: Optional[dict] = None,
        tracker_cfg: Optional[dict] = None,
        target_fps: float = 4.0,
    ):
        super().__init__(name="detector", daemon=True)
        self.detector = detector
        self.policy = policy
        self.frame_q = frame_q
        self.candidate_q = candidate_q
        self.sink_queues = sink_queues
        self.stop_event = stop_event
        self.batch_size = int(batch_size)
        self.batch_timeout = max(batch_timeout_ms, 1) / 1000.0

        from ..triage import EventStateQueue, build_tracker  # lazy — keep workers import-light

        # keyframes / context_window describe how much video context the
        # verifier receives, so they belong to the verifier section — same as
        # single-stream. window / cooldown stay on event_state where they
        # describe detector-side persistence.
        es_cfg = event_state_cfg or {}
        ver_cfg = verifier_cfg or {}
        ver_enabled = bool(ver_cfg.get("enabled", True))
        self._event_states: Dict[str, Any] = {
            sid: EventStateQueue(
                policy=policy,
                window=int(es_cfg.get("window", 8)),
                cooldown_s=float(es_cfg.get("cooldown_s", 10.0)),
                keyframes=int(ver_cfg.get("keyframes", 6)) if ver_enabled else 6,
                context_window_s=float(ver_cfg.get("context_window_s", 3.0)),
                target_fps=float(target_fps),
            )
            for sid in stream_ids
        }
        # Trackers are stateful; one instance per stream so track ids can't
        # collide across cameras.
        self._trackers: Dict[str, Any] = {
            sid: build_tracker(tracker_cfg) for sid in stream_ids
        }

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                msgs: List[_FrameMsg] = self.frame_q.get_batch(
                    max_items=self.batch_size,
                    timeout=self.batch_timeout,
                )
            except StopIteration:
                break
            if not msgs:
                continue

            # --- batched YOLOE inference ---
            frames = [m.frame for m in msgs]
            try:
                all_dets = self.detector.batch(frames)
            except Exception as e:  # noqa: BLE001 — don't kill the worker on one bad batch
                logger.warning("detector batch failed: %s", e)
                continue

            # --- per-stream tracking + event-state + sink fanout ---
            for msg, dets in zip(msgs, all_dets):
                sid = msg.stream_id
                tracker = self._trackers.get(sid)
                if tracker is not None:
                    dets = tracker.update(dets, msg.frame.index)

                sink_q = self.sink_queues.get(sid)
                if sink_q is not None:
                    sink_q.put(_SinkMsg(kind="frame", frame=msg.frame, detections=dets))

                es = self._event_states.get(sid)
                if es is None:
                    continue
                candidates = es.step(msg.frame, dets)
                for cand in candidates:
                    self.candidate_q.put(_CandidateMsg(stream_id=sid, alert=cand))


# ──────────────────────────────────────────────────────────────────────
# verifier — one shared thread, Cosmos-Reason2-2B
# ──────────────────────────────────────────────────────────────────────

class VerifierWorker(threading.Thread):
    def __init__(
        self,
        verifier: Optional[AlertVerifier],
        candidate_q: BoundedQueue,
        sink_queues: Dict[str, BoundedQueue],
        stop_event: threading.Event,
        calibrator: Optional[Any] = None,
    ):
        super().__init__(name="verifier", daemon=True)
        self.verifier = verifier
        self.candidate_q = candidate_q
        self.sink_queues = sink_queues
        self.stop_event = stop_event
        self.calibrator = calibrator

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                msg: _CandidateMsg = self.candidate_q.get(timeout=0.2)
            except TimeoutError:
                continue
            except StopIteration:
                break

            if self.verifier is not None:
                verified = self.verifier.verify(msg.alert)
            else:
                verified = VerifiedAlert(
                    candidate=msg.alert,
                    true_alert=True,
                    confidence=1.0,
                    false_negative_class=None,
                    rationale="verifier disabled",
                )

            sink_q = self.sink_queues.get(msg.stream_id)
            if sink_q is not None:
                sink_q.put(_SinkMsg(kind="alert", verified=verified))

            if self.calibrator is not None:
                try:
                    self.calibrator.record(msg.stream_id, verified)
                except Exception as e:  # noqa: BLE001 — calibration must never take the worker down
                    logger.warning("calibrator.record failed on [%s]: %s", msg.stream_id, e)

            # mirror the single-stream log line
            tag = "TRUE " if verified.true_alert else "FALSE"
            extra = f"  (fn={verified.false_negative_class})" if verified.false_negative_class else ""
            logger.info(
                "[%s] [%s] t=%7.2fs  class=%-10s  sev=%-8s  conf=%.2f%s   -- %s",
                msg.stream_id, tag, verified.candidate.peak_pts_s,
                verified.candidate.class_name, verified.candidate.severity,
                verified.confidence, extra, verified.rationale,
            )


# ──────────────────────────────────────────────────────────────────────
# sink — one thread per stream (owns the mp4 writer + jsonl)
# ──────────────────────────────────────────────────────────────────────

class SinkWorker(threading.Thread):
    def __init__(
        self,
        stream_id: str,
        out_dir: Path,
        fps: float,
        write_annotated: bool,
        jsonl_name: str,
        mp4_name: str,
        sink_q: BoundedQueue,
        stop_event: threading.Event,
        write_thumbnails: bool = True,
        thumbnails_dir: str = "thumbnails",
        thumbnail_ext: str = "jpg",
        thumbnail_quality: int = 90,
        privacy_cfg: Optional[dict] = None,
    ):
        super().__init__(name=f"sink[{stream_id}]", daemon=True)
        self.stream_id = stream_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.write_annotated = bool(write_annotated)
        self.write_thumbnails = bool(write_thumbnails)
        self.jsonl_name = jsonl_name
        self.mp4_name = mp4_name
        self.thumbnails_dir = thumbnails_dir
        self.thumbnail_ext = thumbnail_ext
        self.thumbnail_quality = int(thumbnail_quality)
        self.q = sink_q
        self.stop_event = stop_event
        self.privacy_cfg = privacy_cfg or {}

    def run(self) -> None:
        # Import each sink directly (not via the package __init__) so an
        # annotator-less run doesn't require cv2.
        from ..sinks.jsonl_sink import JsonlSink
        jsonl = JsonlSink(self.out_dir / self.jsonl_name)

        annotator = None
        thumbnail_sink = None
        if self.write_thumbnails:
            from ..privacy import build_face_detector  # lazy — cv2 dep
            from ..sinks.thumbnail_sink import EventThumbnailSink  # cv2 dep
            thumbnail_sink = EventThumbnailSink(
                self.out_dir,
                dir_name=self.thumbnails_dir,
                ext=self.thumbnail_ext,
                quality=self.thumbnail_quality,
                face_detector=build_face_detector(self.privacy_cfg),
                blur_kernel=int(self.privacy_cfg.get("blur_kernel", 31)),
                blur_margin_pct=float(self.privacy_cfg.get("margin_pct", 0.15)),
            )
        if self.write_annotated:
            from ..privacy import build_face_detector  # lazy — cv2 dep
            from ..sinks.video_annotator import VideoAnnotator  # cv2 dep
            annotator = VideoAnnotator(
                self.out_dir / self.mp4_name, fps=self.fps,
                face_detector=build_face_detector(self.privacy_cfg),
                blur_kernel=int(self.privacy_cfg.get("blur_kernel", 31)),
                blur_margin_pct=float(self.privacy_cfg.get("margin_pct", 0.15)),
            )
        try:
            with jsonl:
                while True:
                    try:
                        msg: _SinkMsg = self.q.get(timeout=0.2)
                    except TimeoutError:
                        if self.stop_event.is_set():
                            break
                        continue
                    except StopIteration:
                        break

                    # Per-message writes are the failure risk (disk full,
                    # permissions flip, annotator mid-flight). Catch + log
                    # rather than letting one bad write tear down the thread
                    # — that would drop every subsequent alert for this
                    # stream and the shutdown diagnostic is the only signal.
                    try:
                        if msg.kind == "frame":
                            if annotator is not None and msg.frame is not None:
                                annotator.write(msg.frame, msg.detections or [])
                        elif msg.kind == "alert":
                            if msg.verified is not None:
                                if thumbnail_sink is not None:
                                    thumbnail_sink.write(msg.verified)
                                jsonl.write(msg.verified)
                                if annotator is not None:
                                    annotator.note_alert(msg.verified)
                    except Exception as e:  # noqa: BLE001 — survive per-message failures
                        logger.warning(
                            "sink[%s] write failed for %s: %s",
                            self.stream_id, msg.kind, e,
                        )
        finally:
            if annotator is not None:
                try:
                    annotator.close()
                except Exception as e:  # noqa: BLE001
                    logger.warning("sink[%s] annotator close failed: %s",
                                   self.stream_id, e)
