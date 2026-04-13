"""Multi-stream smoke tests.

Exercise the thread-safe queue + drop policies, and the worker lifecycle with
stubbed YOLOE / Cosmos so nothing GPU-bound is required.
"""
from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from vrs.multistream.queues import BoundedQueue, DropPolicy
from vrs.multistream.workers import (
    DetectorWorker, SinkWorker, VerifierWorker, _FrameMsg,
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
    assert q.put("c") is False    # dropped silently
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
    assert not producer_done.is_set()     # blocked because full
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


# ─── worker fanout with stubs ─────────────────────────────────────────

def _policy(min_persist: int = 1) -> WatchPolicy:
    return WatchPolicy([
        WatchItem(
            name="fire",
            detector_prompts=["fire"],
            verifier_prompt="open flames",
            severity="critical",
            min_score=0.30,
            min_persist_frames=min_persist,
        )
    ])


class _StubDetector:
    """Deterministic stand-in for YOLOEDetector: returns one detection per frame."""
    def batch(self, frames: List[Frame]) -> List[List[Detection]]:
        return [
            [Detection(class_name="fire", score=0.9, xyxy=(1.0, 1.0, 5.0, 5.0))]
            for _ in frames
        ]


class _StubVerifier:
    """Stand-in for AlertVerifier: returns a true_alert for every candidate."""
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls: List[str] = []

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


def test_verifier_worker_dispatches_to_correct_stream_sink():
    stop = threading.Event()
    cand_q = BoundedQueue(maxsize=8)
    sink_qs = {"s1": BoundedQueue(maxsize=8), "s2": BoundedQueue(maxsize=8)}

    stub = _StubVerifier()
    from vrs.multistream.workers import _CandidateMsg  # internal

    worker = VerifierWorker(
        verifier=stub, candidate_q=cand_q, sink_queues=sink_qs, stop_event=stop,
    )
    worker.start()

    cand_s1 = CandidateAlert(
        class_name="fire", severity="critical",
        start_pts_s=0.0, peak_pts_s=0.5, peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0,0,1,1))],
    )
    cand_s2 = CandidateAlert(
        class_name="fire", severity="critical",
        start_pts_s=0.0, peak_pts_s=0.5, peak_frame_index=2,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0,0,1,1))],
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


def test_sink_worker_writes_jsonl_and_handles_missing_frames(tmp_path):
    stop = threading.Event()
    q = BoundedQueue(maxsize=8)
    w = SinkWorker(
        stream_id="s1", out_dir=tmp_path, fps=4.0,
        write_annotated=False,   # avoid cv2 VideoWriter in unit test
        jsonl_name="alerts.jsonl", mp4_name="annotated.mp4",
        sink_q=q, stop_event=stop,
    )
    w.start()

    from vrs.multistream.workers import _SinkMsg
    cand = CandidateAlert(
        class_name="fire", severity="critical",
        start_pts_s=0.0, peak_pts_s=0.5, peak_frame_index=2,
        peak_detections=[],
    )
    v = VerifiedAlert(
        candidate=cand, true_alert=True, confidence=0.8,
        false_negative_class=None, rationale="stub",
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


def test_queue_stats_report_backpressure():
    q = BoundedQueue(maxsize=2, policy=DropPolicy.DROP_OLDEST)
    for i in range(10):
        q.put(i)
    assert q.puts_dropped == 8
    assert q.qsize() == 2
