from __future__ import annotations

import threading
import time
from urllib.request import urlopen

import numpy as np

from vrs.multistream.pipeline import MultiStreamPipeline
from vrs.multistream.queues import BoundedQueue, DropPolicy
from vrs.multistream.workers import DetectorWorker, VerifierWorker, _CandidateMsg, _FrameMsg
from vrs.observability import MetricsRegistry, VRSMetrics, build_metrics
from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.schemas import CandidateAlert, Detection, Frame, VerifiedAlert


def _policy() -> WatchPolicy:
    return WatchPolicy(
        [
            WatchItem(
                name="fire",
                detector_prompts=["fire"],
                verifier_prompt="open flames",
                severity="critical",
                min_score=0.30,
                min_persist_frames=1,
            )
        ]
    )


class _StubDetector:
    def batch(self, frames: list[Frame]) -> list[list[Detection]]:
        return [
            [Detection(class_name="fire", score=0.9, xyxy=(1.0, 1.0, 5.0, 5.0))] for _ in frames
        ]


class _StubVerifier:
    def verify(self, alert: CandidateAlert) -> VerifiedAlert:
        return VerifiedAlert(
            candidate=alert,
            true_alert=True,
            confidence=0.95,
            false_negative_class=None,
            rationale="stub",
        )


def _candidate() -> CandidateAlert:
    return CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=0.5,
        peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0, 0, 1, 1))],
    )


def test_vrs_metrics_render_prometheus_text():
    registry = MetricsRegistry()
    metrics = VRSMetrics(registry)

    metrics.set_queue_depth("frame", "all", 3)
    metrics.set_queue_dropped_total("frame", "all", 2)
    metrics.inc_candidates("cam01", "fire")
    metrics.inc_verified_alerts("cam01", "fire", "true_alert")
    metrics.observe_detector_latency(0.02)
    metrics.observe_verifier_latency(1.5)
    metrics.inc_verifier_errors("transformers")
    metrics.inc_sink_write_errors("cam01")

    text = registry.render()

    assert "# TYPE vrs_queue_depth gauge" in text
    assert 'vrs_queue_depth{queue="frame",stream_id="all"} 3' in text
    assert 'vrs_queue_dropped_total{queue="frame",stream_id="all"} 2' in text
    assert 'vrs_candidates_total{stream_id="cam01",class="fire"} 1' in text
    assert (
        'vrs_verified_alerts_total{stream_id="cam01",class="fire",verdict="true_alert"} 1' in text
    )
    assert "vrs_detector_latency_seconds_count 1" in text
    assert "vrs_verifier_latency_seconds_count 1" in text
    assert 'vrs_verifier_errors_total{backend="transformers"} 1' in text
    assert 'vrs_sink_write_errors_total{stream_id="cam01"} 1' in text


def test_metrics_endpoint_is_disabled_by_default():
    metrics = build_metrics({})
    assert metrics.enabled is False
    assert metrics.url is None


def test_metrics_endpoint_serves_registry():
    metrics = build_metrics(
        {"observability": {"metrics": {"enabled": True, "host": "127.0.0.1", "port": 0}}}
    )
    try:
        metrics.inc_candidates("cam01", "fire")
        assert metrics.url is not None
        with urlopen(metrics.url, timeout=2.0) as response:
            body = response.read().decode("utf-8")
    finally:
        metrics.close()

    assert response.status == 200
    assert 'vrs_candidates_total{stream_id="cam01",class="fire"} 1' in body


def test_multistream_queue_metrics_report_depth_and_drops():
    registry = MetricsRegistry()
    metrics = VRSMetrics(registry)
    pipeline = MultiStreamPipeline.__new__(MultiStreamPipeline)
    pipeline.metrics = metrics
    pipeline._frame_q = BoundedQueue(maxsize=1, policy=DropPolicy.DROP_OLDEST)
    pipeline._candidate_q = BoundedQueue(maxsize=2, policy=DropPolicy.DROP_NEWEST)
    pipeline._sink_qs = {"cam01": BoundedQueue(maxsize=1, policy=DropPolicy.DROP_OLDEST)}

    pipeline._frame_q.put("a")
    pipeline._frame_q.put("b")
    pipeline._candidate_q.put("c")
    pipeline._sink_qs["cam01"].put("d")
    pipeline._sink_qs["cam01"].put("e")

    pipeline._record_queue_metrics()
    text = registry.render()

    assert 'vrs_queue_depth{queue="frame",stream_id="all"} 1' in text
    assert 'vrs_queue_dropped_total{queue="frame",stream_id="all"} 1' in text
    assert 'vrs_queue_depth{queue="candidate",stream_id="all"} 1' in text
    assert 'vrs_queue_depth{queue="sink",stream_id="cam01"} 1' in text
    assert 'vrs_queue_dropped_total{queue="sink",stream_id="cam01"} 1' in text


def test_detector_worker_records_candidate_and_latency_metrics():
    registry = MetricsRegistry()
    metrics = VRSMetrics(registry)
    stop = threading.Event()
    frame_q = BoundedQueue(maxsize=8)
    candidate_q = BoundedQueue(maxsize=8)

    worker = DetectorWorker(
        detector=_StubDetector(),
        policy=_policy(),
        frame_q=frame_q,
        candidate_q=candidate_q,
        sink_queues={"cam01": BoundedQueue(maxsize=8)},
        stream_ids=["cam01"],
        stop_event=stop,
        batch_size=2,
        batch_timeout_ms=10,
        event_state_cfg={"window": 2, "cooldown_s": 0.0},
        metrics=metrics,
    )
    worker.start()
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    frame_q.put(_FrameMsg("cam01", Frame(index=0, pts_s=0.0, image=img)))
    frame_q.put(_FrameMsg("cam01", Frame(index=1, pts_s=0.25, image=img)))

    deadline = time.monotonic() + 1.0
    while candidate_q.qsize() == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    frame_q.close()
    worker.join(timeout=1.0)

    text = registry.render()
    assert 'vrs_candidates_total{stream_id="cam01",class="fire"}' in text
    assert "vrs_detector_latency_seconds_count" in text


def test_verifier_worker_records_verified_alert_and_latency_metrics():
    registry = MetricsRegistry()
    metrics = VRSMetrics(registry)
    stop = threading.Event()
    candidate_q = BoundedQueue(maxsize=8)
    sink_qs = {"cam01": BoundedQueue(maxsize=8)}

    worker = VerifierWorker(
        verifier=_StubVerifier(),
        candidate_q=candidate_q,
        sink_queues=sink_qs,
        stop_event=stop,
        metrics=metrics,
        verifier_backend="stub",
    )
    worker.start()
    candidate_q.put(_CandidateMsg("cam01", _candidate()))

    time.sleep(0.2)
    stop.set()
    candidate_q.close()
    worker.join(timeout=1.0)

    text = registry.render()
    assert (
        'vrs_verified_alerts_total{stream_id="cam01",class="fire",verdict="true_alert"} 1' in text
    )
    assert "vrs_verifier_latency_seconds_count" in text
