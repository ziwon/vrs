"""Multi-stream smoke tests.

Exercise the thread-safe queue + drop policies, and the worker lifecycle with
stubbed YOLOE / Cosmos so nothing GPU-bound is required.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from vrs.multistream.pipeline import load_multistream_spec
from vrs.multistream.queues import BoundedQueue, DropPolicy
from vrs.multistream.workers import (
    DetectorWorker,
    SinkWorker,
    VerifierWorker,
    _FrameMsg,
)
from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.schemas import CandidateAlert, Detection, Frame, VerifiedAlert

# ─── queue policies ───────────────────────────────────────────────────


def test_bounded_queue_drop_oldest_evicts_head_when_full():
    q = BoundedQueue(maxsize=3, policy=DropPolicy.DROP_OLDEST)
    for i in range(5):
        q.put(i)
    assert q.qsize() == 3
    # drop_oldest keeps the most recent 3
    assert q.get() == 2
    assert q.get() == 3
    assert q.get() == 4
    assert q.puts_dropped == 2


def test_bounded_queue_drop_newest_refuses_new_when_full():
    q = BoundedQueue(maxsize=2, policy=DropPolicy.DROP_NEWEST)
    assert q.put("a") is True
    assert q.put("b") is True
    assert q.put("c") is False  # dropped silently
    assert q.qsize() == 2
    assert q.get() == "a"
    assert q.get() == "b"
    assert q.puts_dropped == 1


def test_bounded_queue_block_unblocks_on_consumer_progress():
    q = BoundedQueue(maxsize=1, policy=DropPolicy.BLOCK)
    q.put("first")
    producer_done = threading.Event()

    def _producer():
        q.put("second", timeout=2.0)  # must wait for consumer
        producer_done.set()

    threading.Thread(target=_producer, daemon=True).start()
    time.sleep(0.05)
    assert not producer_done.is_set()  # blocked because full
    assert q.get() == "first"
    assert producer_done.wait(timeout=1.0)
    assert q.get() == "second"


def test_bounded_queue_get_batch_drains_up_to_max():
    q = BoundedQueue(maxsize=16)
    for i in range(5):
        q.put(i)
    batch = q.get_batch(max_items=4, timeout=0.1)
    assert batch == [0, 1, 2, 3]
    assert q.qsize() == 1


def test_bounded_queue_get_batch_returns_empty_on_timeout():
    q = BoundedQueue(maxsize=4)
    assert q.get_batch(max_items=4, timeout=0.05) == []


def test_load_multistream_spec_accepts_partial_manifest(tmp_path):
    manifest = tmp_path / "streams.yaml"
    manifest.write_text(
        "multistream:\n"
        "  detector_batch_size: 4\n"
        "streams:\n"
        "  - id: cam01\n"
        "    rtsp: rtsp://demo.example/stream1\n",
        encoding="utf-8",
    )

    cfg = load_multistream_spec(manifest)

    assert cfg["multistream"]["detector_batch_size"] == 4
    assert cfg["streams"][0]["id"] == "cam01"


# ─── worker fanout with stubs ─────────────────────────────────────────


def _policy(min_persist: int = 1) -> WatchPolicy:
    return WatchPolicy(
        [
            WatchItem(
                name="fire",
                detector_prompts=["fire"],
                verifier_prompt="open flames",
                severity="critical",
                min_score=0.30,
                min_persist_frames=min_persist,
            )
        ]
    )


class _StubDetector:
    """Deterministic stand-in for YOLOEDetector: returns one detection per frame."""

    def batch(self, frames: list[Frame]) -> list[list[Detection]]:
        return [
            [Detection(class_name="fire", score=0.9, xyxy=(1.0, 1.0, 5.0, 5.0))] for _ in frames
        ]


class _StubVerifier:
    """Stand-in for AlertVerifier: returns a true_alert for every candidate."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls: list[str] = []

    def verify(self, alert: CandidateAlert) -> VerifiedAlert:
        self.calls.append(alert.class_name)
        if self.delay_s:
            time.sleep(self.delay_s)
        return VerifiedAlert(
            candidate=alert,
            true_alert=True,
            confidence=0.99,
            false_negative_class=None,
            rationale="stub",
        )


def test_detector_worker_reads_keyframes_and_window_from_verifier_cfg():
    """Regression: multi-stream must read keyframes / context_window_s from the
    'verifier' section, matching single-stream. Previously it read them from
    'event_state', which silently ignored configs/default.yaml."""
    policy = _policy()
    worker = DetectorWorker(
        detector=_StubDetector(),
        policy=policy,
        frame_q=BoundedQueue(maxsize=4),
        candidate_q=BoundedQueue(maxsize=4),
        sink_queues={"s1": BoundedQueue(maxsize=4)},
        stream_ids=["s1"],
        stop_event=threading.Event(),
        event_state_cfg={"window": 8, "cooldown_s": 10.0},
        verifier_cfg={"enabled": True, "keyframes": 3, "context_window_s": 5.5},
        target_fps=4.0,
    )
    es = worker._event_states["s1"]
    assert es.keyframes == 3
    assert es.context_window_s == 5.5


def test_detector_worker_batches_frames_and_fires_candidates(tmp_path, monkeypatch):
    policy = _policy(min_persist=1)
    stop = threading.Event()
    frame_q = BoundedQueue(maxsize=16)
    cand_q = BoundedQueue(maxsize=16)
    sink_qs = {"s1": BoundedQueue(maxsize=16), "s2": BoundedQueue(maxsize=16)}

    worker = DetectorWorker(
        detector=_StubDetector(),
        policy=policy,
        frame_q=frame_q,
        candidate_q=cand_q,
        sink_queues=sink_qs,
        stream_ids=["s1", "s2"],
        stop_event=stop,
        batch_size=4,
        batch_timeout_ms=20,
        event_state_cfg={"window": 2, "cooldown_s": 0.0},
        target_fps=4.0,
    )
    worker.start()

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(3):
        frame_q.put(_FrameMsg("s1", Frame(index=i, pts_s=i * 0.25, image=img)))
        frame_q.put(_FrameMsg("s2", Frame(index=i, pts_s=i * 0.25, image=img)))

    # wait for the batcher to drain (it runs every 20 ms)
    time.sleep(0.6)
    stop.set()
    frame_q.close()
    worker.join(timeout=1.0)

    # each stream hit once the persistence threshold → at least one candidate each
    stream_ids_seen = set()
    while cand_q.qsize():
        msg = cand_q.get(timeout=0.1)
        stream_ids_seen.add(msg.stream_id)
    assert stream_ids_seen == {"s1", "s2"}

    # every frame must have been fanned out to its per-stream sink queue
    assert sink_qs["s1"].qsize() >= 1
    assert sink_qs["s2"].qsize() >= 1


class _RecordingMetrics:
    def __init__(self):
        self.queue_waits: list[tuple[str, str, float]] = []
        self.token_rates: list[tuple[str, float]] = []

    def observe_queue_wait(self, queue: str, stream_id: str, seconds: float) -> None:
        self.queue_waits.append((queue, stream_id, seconds))

    def observe_detector_latency(self, seconds: float) -> None:
        pass

    def observe_verifier_latency(self, seconds: float) -> None:
        pass

    def observe_verifier_tokens_per_second(self, backend: str, tokens_per_second: float) -> None:
        self.token_rates.append((backend, tokens_per_second))

    def inc_candidates(self, stream_id: str, class_name: str, amount: int = 1) -> None:
        pass

    def inc_verified_alerts(
        self,
        stream_id: str,
        class_name: str,
        verdict: str,
        severity: str = "unknown",
        amount: int = 1,
    ) -> None:
        pass

    def inc_verifier_errors(self, backend: str, amount: int = 1) -> None:
        pass

    def inc_privacy_setup_failures(self, backend: str, amount: int = 1) -> None:
        pass


def test_verifier_worker_dispatches_to_correct_stream_sink():
    stop = threading.Event()
    cand_q = BoundedQueue(maxsize=8)
    sink_qs = {"s1": BoundedQueue(maxsize=8), "s2": BoundedQueue(maxsize=8)}

    stub = _StubVerifier()
    from vrs.multistream.workers import _CandidateMsg  # internal

    worker = VerifierWorker(
        verifier=stub,
        candidate_q=cand_q,
        sink_queues=sink_qs,
        stop_event=stop,
    )
    worker.start()

    cand_s1 = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=0.5,
        peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0, 0, 1, 1))],
    )
    cand_s2 = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=0.5,
        peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0, 0, 1, 1))],
    )
    cand_q.put(_CandidateMsg("s1", cand_s1))
    cand_q.put(_CandidateMsg("s2", cand_s2))

    time.sleep(0.3)
    stop.set()
    cand_q.close()
    worker.join(timeout=1.0)

    assert sink_qs["s1"].qsize() == 1
    assert sink_qs["s2"].qsize() == 1
    assert stub.calls == ["fire", "fire"]


def _candidate(
    *,
    class_name: str = "fire",
    severity: str = "critical",
    pts_s: float = 1.0,
) -> CandidateAlert:
    return CandidateAlert(
        class_name=class_name,
        severity=severity,
        start_pts_s=max(0.0, pts_s - 0.5),
        peak_pts_s=pts_s,
        peak_frame_index=int(pts_s * 4),
        peak_detections=[Detection(class_name=class_name, score=0.9, xyxy=(0, 0, 1, 1))],
    )


def _verified(
    *,
    class_name: str = "fire",
    severity: str = "critical",
    pts_s: float = 1.0,
    true_alert: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
) -> VerifiedAlert:
    return VerifiedAlert(
        candidate=_candidate(class_name=class_name, severity=severity, pts_s=pts_s),
        true_alert=true_alert,
        confidence=0.9,
        false_negative_class=None,
        rationale="stub",
        bbox_xywh_norm=bbox,
    )


def test_incident_correlator_groups_adjacent_streams_inside_window():
    from vrs.multistream.incidents import IncidentCorrelator

    corr = IncidentCorrelator(
        {
            "enabled": True,
            "window_s": 3.0,
            "adjacency": {"cam-a": ["cam-b"]},
        }
    )

    first = corr.assign("cam-a", _verified(pts_s=10.0))
    second = corr.assign("cam-b", _verified(pts_s=11.0))

    assert first.incident_id == "inc-000001"
    assert second.incident_id == first.incident_id
    assert second.incident_stream_ids == ["cam-a", "cam-b"]
    assert second.incident_primary_stream_id == "cam-a"


def test_incident_correlator_requires_adjacency_and_time_window():
    from vrs.multistream.incidents import IncidentCorrelator

    corr = IncidentCorrelator(
        {
            "enabled": True,
            "window_s": 1.0,
            "adjacency": {"cam-a": ["cam-b"]},
        }
    )

    first = corr.assign("cam-a", _verified(pts_s=10.0))
    non_adjacent = corr.assign("cam-c", _verified(pts_s=10.2))
    late = corr.assign("cam-b", _verified(pts_s=12.0))

    assert first.incident_id == "inc-000001"
    assert non_adjacent.incident_id == "inc-000002"
    assert late.incident_id == "inc-000003"


def test_incident_correlator_can_require_bbox_overlap():
    from vrs.multistream.incidents import IncidentCorrelator

    corr = IncidentCorrelator(
        {
            "enabled": True,
            "window_s": 3.0,
            "adjacency": {"cam-a": ["cam-b"]},
            "min_bbox_iou": 0.2,
        }
    )

    first = corr.assign("cam-a", _verified(pts_s=10.0, bbox=(0.1, 0.1, 0.3, 0.3)))
    far = corr.assign("cam-b", _verified(pts_s=10.5, bbox=(0.7, 0.7, 0.2, 0.2)))
    near = corr.assign("cam-b", _verified(pts_s=10.8, bbox=(0.12, 0.12, 0.3, 0.3)))

    assert first.incident_id == "inc-000001"
    assert far.incident_id == "inc-000002"
    assert near.incident_id == first.incident_id


def test_verifier_worker_assigns_incident_ids_before_sink():
    from vrs.multistream.incidents import IncidentCorrelator
    from vrs.multistream.workers import _CandidateMsg

    stop = threading.Event()
    cand_q = BoundedQueue(maxsize=8)
    sink_qs = {"cam-a": BoundedQueue(maxsize=8), "cam-b": BoundedQueue(maxsize=8)}
    worker = VerifierWorker(
        verifier=_StubVerifier(),
        candidate_q=cand_q,
        sink_queues=sink_qs,
        stop_event=stop,
        incident_correlator=IncidentCorrelator(
            {
                "enabled": True,
                "window_s": 3.0,
                "adjacency": {"cam-a": ["cam-b"]},
            }
        ),
    )
    worker.start()

    cand_q.put(_CandidateMsg("cam-a", _candidate(pts_s=10.0)))
    cand_q.put(_CandidateMsg("cam-b", _candidate(pts_s=11.0)))

    time.sleep(0.3)
    stop.set()
    cand_q.close()
    worker.join(timeout=1.0)

    first = sink_qs["cam-a"].get(timeout=0.1).verified
    second = sink_qs["cam-b"].get(timeout=0.1).verified
    assert first is not None and second is not None
    assert first.incident_id == "inc-000001"
    assert second.incident_id == first.incident_id
    assert second.to_json()["incident_stream_ids"] == ["cam-a", "cam-b"]


def test_verifier_worker_records_queue_wait_and_token_rate():
    from vrs.multistream.workers import _CandidateMsg

    stop = threading.Event()
    cand_q = BoundedQueue(maxsize=8)
    sink_qs = {"s1": BoundedQueue(maxsize=8)}
    metrics = _RecordingMetrics()

    stub = _StubVerifier()
    stub.vlm = type("_VLM", (), {"last_generation_stats": {"tokens_per_second": 12.5}})()
    worker = VerifierWorker(
        verifier=stub,
        candidate_q=cand_q,
        sink_queues=sink_qs,
        stop_event=stop,
        metrics=metrics,
        verifier_backend="stub",
    )
    worker.start()

    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=0.5,
        peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0, 0, 1, 1))],
    )
    cand_q.put(_CandidateMsg("s1", cand, enqueued_at=time.perf_counter() - 0.01))
    time.sleep(0.2)
    stop.set()
    cand_q.close()
    worker.join(timeout=1.0)

    assert any(
        queue == "candidate" and stream_id == "s1" for queue, stream_id, _ in metrics.queue_waits
    )
    assert metrics.token_rates == [("stub", 12.5)]


def test_sink_worker_writes_jsonl_and_handles_missing_frames(tmp_path):
    stop = threading.Event()
    q = BoundedQueue(maxsize=8)
    w = SinkWorker(
        stream_id="s1",
        out_dir=tmp_path,
        fps=4.0,
        write_annotated=False,  # avoid cv2 VideoWriter in unit test
        jsonl_name="alerts.jsonl",
        mp4_name="annotated.mp4",
        sink_q=q,
        stop_event=stop,
    )
    w.start()

    from vrs.multistream.workers import _SinkMsg

    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=0.5,
        peak_frame_index=2,
        peak_detections=[],
    )
    v = VerifiedAlert(
        candidate=cand,
        true_alert=True,
        confidence=0.8,
        false_negative_class=None,
        rationale="stub",
    )
    q.put(_SinkMsg(kind="alert", verified=v))

    time.sleep(0.3)
    stop.set()
    q.close()
    w.join(timeout=1.0)

    jsonl = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl) == 1
    assert '"fire"' in jsonl[0]
    assert '"true_alert": true' in jsonl[0]
    index_row = json.loads((tmp_path / "object_manifest.index.jsonl").read_text(encoding="utf-8"))
    manifest_path = tmp_path / index_row["manifest_ref"]["metadata"]["relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "object_manifest.v1"
    assert manifest["stream_id"] == "s1"
    assert manifest["records"][0]["schema_version"] == "verified_alert.v1"
    assert manifest["records"][0]["stream_id"] == "s1"


def test_queue_stats_report_backpressure():
    q = BoundedQueue(maxsize=2, policy=DropPolicy.DROP_OLDEST)
    for i in range(10):
        q.put(i)
    assert q.puts_dropped == 8
    assert q.qsize() == 2


def test_fast_shutdown_sets_worker_stop_immediately():
    from vrs.multistream.pipeline import MultiStreamPipeline

    p = MultiStreamPipeline.__new__(MultiStreamPipeline)
    p._stop = threading.Event()
    p._decoder_stop = threading.Event()
    p._frame_q = BoundedQueue(maxsize=4)
    p._candidate_q = BoundedQueue(maxsize=4)
    p._sink_qs = {"s1": BoundedQueue(maxsize=4)}
    p._decoders = []
    p._sinks = []
    p._verifier_worker = None
    p._calibrator = None
    saw_stop = threading.Event()

    def _worker():
        p._stop.wait(timeout=1.0)
        if p._stop.is_set():
            saw_stop.set()

    p._detector_worker = threading.Thread(target=_worker, name="detector", daemon=True)
    p._detector_worker.start()

    p.stop("fast")

    assert p._decoder_stop.is_set()
    assert p._stop.is_set()
    assert saw_stop.is_set()


def test_drain_shutdown_flushes_frame_candidate_and_sink_queues_without_loss():
    from vrs.multistream.pipeline import MultiStreamPipeline

    p = MultiStreamPipeline.__new__(MultiStreamPipeline)
    p._stop = threading.Event()
    p._decoder_stop = threading.Event()
    p._shutdown_drain_timeout_s = 2.0
    p._frame_q = BoundedQueue(maxsize=8)
    p._candidate_q = BoundedQueue(maxsize=8)
    sink_q = BoundedQueue(maxsize=8)
    p._sink_qs = {"s1": sink_q}
    p._decoders = []
    p._calibrator = None
    written: list[str] = []

    for item in ("f1", "f2"):
        p._frame_q.put(item)
    p._candidate_q.put("candidate-before-stop")
    sink_q.put("sink-before-stop")

    def _detector():
        while True:
            try:
                item = p._frame_q.get(timeout=0.05)
            except TimeoutError:
                continue
            except StopIteration:
                break
            p._candidate_q.put(f"candidate:{item}")

    def _verifier():
        while True:
            try:
                item = p._candidate_q.get(timeout=0.05)
            except TimeoutError:
                continue
            except StopIteration:
                break
            sink_q.put(f"verified:{item}")

    def _sink():
        while True:
            try:
                item = sink_q.get(timeout=0.05)
            except TimeoutError:
                continue
            except StopIteration:
                break
            written.append(item)

    p._detector_worker = threading.Thread(target=_detector, name="detector", daemon=True)
    p._verifier_worker = threading.Thread(target=_verifier, name="verifier", daemon=True)
    p._sinks = [threading.Thread(target=_sink, name="sink[s1]", daemon=True)]
    p._detector_worker.start()
    p._verifier_worker.start()
    p._sinks[0].start()

    p.stop("drain")

    assert p._stop.is_set()
    assert set(written) == {
        "sink-before-stop",
        "verified:candidate-before-stop",
        "verified:candidate:f1",
        "verified:candidate:f2",
    }


def test_drain_shutdown_logs_queue_sizes_and_alive_workers_on_deadline(caplog):
    import logging

    from vrs.multistream.pipeline import MultiStreamPipeline

    p = MultiStreamPipeline.__new__(MultiStreamPipeline)
    p._stop = threading.Event()
    p._decoder_stop = threading.Event()
    p._shutdown_drain_timeout_s = 0.01
    p._frame_q = BoundedQueue(maxsize=4)
    p._candidate_q = BoundedQueue(maxsize=4)
    p._sink_qs = {"s1": BoundedQueue(maxsize=4)}
    p._decoders = []
    p._verifier_worker = None
    p._sinks = []
    p._calibrator = None
    release = threading.Event()

    p._frame_q.put("still-buffered")

    def _blocked_detector():
        release.wait(timeout=1.0)

    p._detector_worker = threading.Thread(
        target=_blocked_detector,
        name="detector",
        daemon=True,
    )
    p._detector_worker.start()

    try:
        with caplog.at_level(logging.WARNING, logger="vrs.multistream.pipeline"):
            p.stop("drain")
    finally:
        release.set()
        p._detector_worker.join(timeout=1.0)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("drain shutdown deadline exceeded" in m for m in messages)
    assert any("frame_q=1" in m for m in messages)
    assert any("alive workers: detector" in m for m in messages)


def test_drop_delta_logger_warns_on_increase(caplog):
    """Regression: when queue drop counters grow between samples, the
    pipeline must surface the jump as a WARNING naming every affected
    queue — that's the only operator-visible signal of live-video
    backpressure."""
    import logging

    from vrs.multistream.pipeline import MultiStreamPipeline

    prev = {"frame_q": 0, "candidate_q": 0, "sink[s1]": 0}
    cur = {"frame_q": 5, "candidate_q": 0, "sink[s1]": 2}

    with caplog.at_level(logging.WARNING, logger="vrs.multistream.pipeline"):
        MultiStreamPipeline._log_drop_deltas(prev, cur)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "frame_q+5" in msg
    assert "sink[s1]+2" in msg
    assert "candidate_q" not in msg  # unchanged → not named


def test_drop_delta_logger_silent_when_unchanged(caplog):
    import logging

    from vrs.multistream.pipeline import MultiStreamPipeline

    prev = {"frame_q": 10, "sink[s1]": 3}
    cur = {"frame_q": 10, "sink[s1]": 3}
    with caplog.at_level(logging.WARNING, logger="vrs.multistream.pipeline"):
        MultiStreamPipeline._log_drop_deltas(prev, cur)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


def test_shutdown_helper_names_hung_threads():
    """Regression: shutdown must surface a log-friendly record of any thread
    that outlasts its join timeout, so hung shutdowns are debuggable."""
    from vrs.multistream.pipeline import MultiStreamPipeline

    release = threading.Event()

    def _blocked():
        release.wait()  # only exits when the test releases it

    stuck = threading.Thread(target=_blocked, name="stuck-worker", daemon=True)
    stuck.start()

    hung: list[str] = []
    try:
        MultiStreamPipeline._join_or_track_hung(stuck, timeout=0.05, hung=hung)
        assert len(hung) == 1
        assert "stuck-worker" in hung[0]
        assert "0.05" in hung[0]
    finally:
        release.set()
        stuck.join(timeout=1.0)

    # A thread that exits cleanly within the timeout must NOT be recorded.
    fast = threading.Thread(target=lambda: None, name="fast-worker", daemon=True)
    fast.start()
    hung2: list[str] = []
    MultiStreamPipeline._join_or_track_hung(fast, timeout=1.0, hung=hung2)
    assert hung2 == []
