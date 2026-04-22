"""Failure-path tests — items the system review / roadmap docs
explicitly called out as weak coverage:

  1. RTSP reconnect behavior
  2. malformed / partial model outputs beyond parser helper cases
  3. sink failures (disk full, annotator write error, etc.)
  4. shutdown under in-flight work
  5. multi-stream config / manifest validation

All tests are CPU-only; GPU paths are stubbed.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from vrs.multistream.pipeline import _validate_multistream_spec, load_multistream_spec
from vrs.multistream.queues import BoundedQueue, DropPolicy
from vrs.multistream.workers import (
    SinkWorker,
    VerifierWorker,
    _CandidateMsg,
    _SinkMsg,
)
from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.schemas import CandidateAlert, Detection, Frame, VerifiedAlert
from vrs.verifier.alert_verifier import AlertVerifier


def _policy() -> WatchPolicy:
    return WatchPolicy([
        WatchItem(name="fire", detector_prompts=["fire"], verifier_prompt="flames",
                  severity="critical", min_score=0.30, min_persist_frames=1),
        WatchItem(name="smoke", detector_prompts=["smoke"], verifier_prompt="smoke",
                  severity="high", min_score=0.30, min_persist_frames=1),
    ])


def _cand(class_name: str = "fire") -> CandidateAlert:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    return CandidateAlert(
        class_name=class_name, severity="critical",
        start_pts_s=0.0, peak_pts_s=1.0, peak_frame_index=4,
        peak_detections=[Detection(class_name=class_name, score=0.9, xyxy=(0, 0, 10, 10))],
        keyframes=[img, img], keyframe_pts=[0.5, 1.0],
    )


# ──────────────────────────────────────────────────────────────────────
# 1. RTSP reconnect behavior
# ──────────────────────────────────────────────────────────────────────

class _FakeCapture:
    """VideoCapture stub whose read() fails after N successful frames."""
    def __init__(self, frames_before_fail: int = 2, fps: float = 25.0,
                 opens_to_succeed: int = 1):
        self._frames = frames_before_fail
        self._fps = fps
        self._calls = 0
        self._opens_left = opens_to_succeed
    def isOpened(self) -> bool:
        if self._opens_left <= 0:
            return False
        self._opens_left -= 1
        return True
    def get(self, prop) -> float:  # noqa: ARG002
        return self._fps
    def read(self):
        self._calls += 1
        if self._calls <= self._frames:
            return True, np.zeros((8, 8, 3), dtype=np.uint8)
        return False, None
    def release(self) -> None:
        pass


def _patch_capture(monkeypatch, cap_factory, module):
    monkeypatch.setattr(module.cv2, "VideoCapture", cap_factory)


def test_opencv_reader_reconnects_on_live_source(monkeypatch):
    """A live RTSP source that fails mid-stream should trigger a reconnect
    and keep yielding frames."""
    from vrs.multistream import readers as R

    opens: List[int] = []
    def _factory(*args, **kwargs):
        cap = _FakeCapture(frames_before_fail=1, opens_to_succeed=1)
        opens.append(id(cap))
        return cap
    monkeypatch.setattr(R.cv2, "VideoCapture", _factory)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)  # don't actually wait

    reader = R.OpenCVReader("rtsp://fake/stream", target_fps=1.0, reconnect_s=0.0)

    it = iter(reader)
    frames = []
    for _ in range(3):
        # second .read() will return ok=False → reconnect → new capture →
        # another successful read, then fail again. After 3 iterations we
        # stop (the third reconnect won't be asked for yet).
        try:
            frames.append(next(it))
        except StopIteration:
            break
    # At least one reconnect must have happened
    assert len(opens) >= 2


def test_opencv_reader_does_not_reconnect_on_file_source(monkeypatch):
    """A file that ends should stop iteration — not reconnect."""
    from vrs.multistream import readers as R

    opens = {"count": 0}
    def _factory(*args, **kwargs):
        opens["count"] += 1
        return _FakeCapture(frames_before_fail=2, opens_to_succeed=1)
    monkeypatch.setattr(R.cv2, "VideoCapture", _factory)
    monkeypatch.setattr(R.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError(
        "file source should never sleep for a reconnect"
    )))

    reader = R.OpenCVReader("/tmp/nonexistent.mp4", target_fps=1.0)
    frames = list(reader)
    assert opens["count"] == 1
    assert len(frames) >= 1


def test_opencv_reader_stops_when_reconnect_fails(monkeypatch):
    """If reconnection can't re-open the source, iteration ends cleanly
    instead of spinning forever."""
    from vrs.multistream import readers as R

    def _factory(*args, **kwargs):
        # Opens once, then every subsequent open reports closed
        return _FakeCapture(frames_before_fail=1, opens_to_succeed=1)
    # Override _open so reopen fails after the first
    call_count = {"n": 0}
    original_factory = _factory
    def _factory2(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCapture(frames_before_fail=1, opens_to_succeed=1)
        return _FakeCapture(frames_before_fail=0, opens_to_succeed=0)
    monkeypatch.setattr(R.cv2, "VideoCapture", _factory2)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    reader = R.OpenCVReader("rtsp://fake/stream", target_fps=1.0, reconnect_s=0.0)
    # Must terminate; no infinite loop
    frames = list(reader)
    # At most the first cap's frames
    assert len(frames) <= 2


def test_stream_reader_reconnects_on_live_source(monkeypatch):
    """Single-stream ingest path must reconnect on live-source read failures
    identically to the multi-stream OpenCVReader."""
    from vrs.ingest import stream_reader as S

    opens: List[int] = []
    def _factory(*args, **kwargs):
        cap = _FakeCapture(frames_before_fail=1, opens_to_succeed=1)
        opens.append(id(cap))
        return cap
    monkeypatch.setattr(S.cv2, "VideoCapture", _factory)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)

    reader = S.StreamReader("rtsp://fake/stream", target_fps=1.0, reconnect_s=0.0)
    it = iter(reader)
    for _ in range(2):
        try:
            next(it)
        except StopIteration:
            break
    assert len(opens) >= 2


# ──────────────────────────────────────────────────────────────────────
# 2. Malformed / partial model outputs
# ──────────────────────────────────────────────────────────────────────

class _CannedBackend:
    """CosmosBackend stub that returns a caller-supplied string (or raises)."""
    def __init__(self, response: str = "", error: Optional[Exception] = None):
        self.response = response
        self.error = error
    def chat_video(self, system_prompt, user_prompt, frames_bgr, *,
                   clip_fps=None, response_schema=None):
        if self.error is not None:
            raise self.error
        return self.response


def test_verifier_truncated_json_passes_through_as_true_alert(caplog):
    backend = _CannedBackend(response='{"true_alert": true, "confidence": 0.')
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    with caplog.at_level(logging.WARNING):
        result = verifier.verify(_cand())
    assert result.true_alert is True            # pass-through default
    assert result.confidence == 0.0             # clamped
    assert any("unparseable" in r.getMessage() for r in caplog.records)


def test_verifier_missing_true_alert_defaults_to_true(caplog):
    """Spec invariant: pipeline never silently drops a detector hit."""
    backend = _CannedBackend(response='{"confidence": 0.5, "rationale": "hmm"}')
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    with caplog.at_level(logging.WARNING):
        result = verifier.verify(_cand())
    assert result.true_alert is True
    assert any("missing 'true_alert'" in r.getMessage() for r in caplog.records)


def test_verifier_clamps_out_of_range_confidence():
    backend = _CannedBackend(response=(
        '{"true_alert": true, "confidence": 1.8, '
        '"false_negative_class": null, "rationale": "ok"}'
    ))
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    hi = verifier.verify(_cand())
    assert hi.confidence == 1.0

    backend.response = ('{"true_alert": true, "confidence": -0.5, '
                        '"false_negative_class": null, "rationale": "ok"}')
    lo = verifier.verify(_cand())
    assert lo.confidence == 0.0


def test_verifier_nulls_illegal_false_negative_class():
    """An FN-class the verifier hallucinates must be dropped — otherwise
    a downstream router would dispatch on a class the pipeline doesn't
    know how to handle."""
    backend = _CannedBackend(response=(
        '{"true_alert": false, "confidence": 0.3, '
        '"false_negative_class": "unicorn", "rationale": "nope"}'
    ))
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    result = verifier.verify(_cand())
    assert result.false_negative_class is None


def test_verifier_preserves_legal_false_negative_class():
    backend = _CannedBackend(response=(
        '{"true_alert": false, "confidence": 0.1, '
        '"false_negative_class": "smoke", "rationale": "it was smoke"}'
    ))
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    result = verifier.verify(_cand())
    assert result.false_negative_class == "smoke"


def test_verifier_backend_exception_produces_diagnostic_passthrough():
    backend = _CannedBackend(error=RuntimeError("CUDA OOM"))
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    result = verifier.verify(_cand())
    assert result.true_alert is True             # don't drop the detector hit
    assert result.confidence == 0.0
    assert "CUDA OOM" in result.rationale


def test_verifier_reject_policy_suppresses_unparseable_response():
    backend = _CannedBackend(response='{"true_alert": true, "confidence": 0.')
    verifier = AlertVerifier(
        cosmos=backend, policy=_policy(), failure_policy="reject",
    )
    result = verifier.verify(_cand())
    assert result.true_alert is False
    assert result.confidence == 0.0


def test_verifier_reject_policy_suppresses_backend_exception():
    backend = _CannedBackend(error=RuntimeError("CUDA OOM"))
    verifier = AlertVerifier(
        cosmos=backend, policy=_policy(), failure_policy="reject",
    )
    result = verifier.verify(_cand())
    assert result.true_alert is False
    assert result.confidence == 0.0
    assert "CUDA OOM" in result.rationale


def test_verifier_empty_keyframes_skips_backend():
    """If no keyframes survived the ring buffer, we short-circuit without
    calling the backend — otherwise we'd pass an empty frames list."""
    backend = _CannedBackend(error=RuntimeError("should not be called"))
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    cand = _cand()
    cand.keyframes = []
    cand.keyframe_pts = []
    result = verifier.verify(cand)
    assert result.true_alert is True
    assert "no keyframes" in result.rationale


def test_verifier_empty_string_response_falls_back_safely():
    """Empty generation (e.g. immediate EOS) must still land a VerifiedAlert."""
    backend = _CannedBackend(response="")
    verifier = AlertVerifier(cosmos=backend, policy=_policy())
    result = verifier.verify(_cand())
    assert result.true_alert is True
    assert result.verifier_raw == ""


# ──────────────────────────────────────────────────────────────────────
# 3. Sink failures
# ──────────────────────────────────────────────────────────────────────

class _ExplodingJsonl:
    """JsonlSink stand-in that raises on the Nth write."""
    def __init__(self, fail_after: int = 0):
        self.writes = 0
        self.fail_after = fail_after
        self.path = Path("/tmp/exploding.jsonl")  # never actually touched
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
    def write(self, verified):
        self.writes += 1
        if self.writes > self.fail_after:
            raise OSError("No space left on device")


def test_sink_worker_survives_write_failure_and_keeps_draining(tmp_path, monkeypatch, caplog):
    """SinkWorker used to let one OSError tear down the thread — which
    then silently lost every subsequent alert for that stream. Regression
    test: a single write failure is logged, the worker stays alive, and
    later alerts still get through once the underlying fault recovers."""
    # Wire up an exploding JsonlSink by monkey-patching the import path
    # that workers.py uses.
    import vrs.sinks.jsonl_sink as J
    instances = {"fake": None}
    class _FakeSink:
        def __init__(self, path):
            self._real_path = tmp_path / "alerts.jsonl"
            self._real_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = None
            self._n = 0
            instances["fake"] = self
        def __enter__(self):
            self._fp = open(self._real_path, "a", encoding="utf-8")
            return self
        def __exit__(self, *exc):
            if self._fp is not None:
                self._fp.close()
        def write(self, verified):
            self._n += 1
            if self._n == 1:
                raise OSError("simulated disk failure")
            # second call writes real JSON so we can assert recovery
            import json as _json
            self._fp.write(_json.dumps(verified.to_json()) + "\n")
            self._fp.flush()
    monkeypatch.setattr(J, "JsonlSink", _FakeSink)

    stop = threading.Event()
    q = BoundedQueue(maxsize=16)
    worker = SinkWorker(
        stream_id="s1", out_dir=tmp_path, fps=4.0,
        write_annotated=False, jsonl_name="alerts.jsonl", mp4_name="x.mp4",
        sink_q=q, stop_event=stop,
    )
    worker.start()

    def _v(label: str) -> VerifiedAlert:
        c = _cand()
        return VerifiedAlert(candidate=c, true_alert=True, confidence=0.8,
                             false_negative_class=None, rationale=label)
    with caplog.at_level(logging.WARNING, logger="vrs.multistream.workers"):
        q.put(_SinkMsg(kind="alert", verified=_v("first-explodes")))
        q.put(_SinkMsg(kind="alert", verified=_v("second-ok")))
        time.sleep(0.4)
        stop.set()
        q.close()
        worker.join(timeout=2.0)

    # The worker is NOT hung and NOT dead-on-arrival:
    assert not worker.is_alive(), "worker should have exited cleanly after stop"
    # The second alert landed despite the first failure:
    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "second-ok" in lines[0]
    # The failure was logged (not swallowed):
    assert any("simulated disk failure" in r.getMessage() for r in caplog.records)


def test_sink_worker_survives_annotator_write_failure(tmp_path, monkeypatch, caplog):
    """Annotator.write raising mid-stream should not take the thread down."""
    import vrs.sinks.jsonl_sink as J
    import vrs.sinks.video_annotator as V

    # Minimal real jsonl so the "alert" branch path still works; focus of
    # the test is the frame-annotator error branch.
    monkeypatch.setattr(J, "JsonlSink", lambda path: _NoopSink(path))
    class _ExplodingAnnotator:
        def __init__(self, path, fps, **kwargs):
            self.written = 0
        def write(self, frame, dets):
            self.written += 1
            if self.written == 1:
                raise RuntimeError("annotator: video writer not open")
        def note_alert(self, v): pass
        def close(self): pass
    monkeypatch.setattr(V, "VideoAnnotator", _ExplodingAnnotator)

    stop = threading.Event()
    q = BoundedQueue(maxsize=16)
    worker = SinkWorker(
        stream_id="s1", out_dir=tmp_path, fps=4.0,
        write_annotated=True, jsonl_name="alerts.jsonl", mp4_name="x.mp4",
        sink_q=q, stop_event=stop,
    )
    worker.start()

    frame = Frame(index=0, pts_s=0.0, image=np.zeros((8, 8, 3), dtype=np.uint8))
    with caplog.at_level(logging.WARNING, logger="vrs.multistream.workers"):
        q.put(_SinkMsg(kind="frame", frame=frame, detections=[]))
        # Worker survived → second frame goes through ok
        q.put(_SinkMsg(kind="frame", frame=frame, detections=[]))
        time.sleep(0.4)
        stop.set()
        q.close()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert any("annotator: video writer not open" in r.getMessage()
               for r in caplog.records)


class _NoopSink:
    """JsonlSink stand-in that swallows everything — for tests focused on
    a different path."""
    def __init__(self, path):
        self.path = path
    def __enter__(self): return self
    def __exit__(self, *exc): pass
    def write(self, verified): pass


# ──────────────────────────────────────────────────────────────────────
# 4. Shutdown under in-flight work
# ──────────────────────────────────────────────────────────────────────

class _SlowStubVerifier:
    """AlertVerifier stand-in whose verify() takes a configurable time."""
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.verdicts = 0
    def verify(self, alert):
        time.sleep(self.delay_s)
        self.verdicts += 1
        return VerifiedAlert(
            candidate=alert, true_alert=True, confidence=0.9,
            false_negative_class=None, rationale="slow but done",
        )


def test_verifier_worker_finishes_current_candidate_before_stopping():
    """A verify already in-flight when stop() fires must complete and emit
    its VerifiedAlert to the sink — otherwise shutdowns lose work the
    operator thought was already committed."""
    stop = threading.Event()
    cand_q = BoundedQueue(maxsize=4)
    sink_q = BoundedQueue(maxsize=4)

    v = _SlowStubVerifier(delay_s=0.2)
    worker = VerifierWorker(
        verifier=v, candidate_q=cand_q,
        sink_queues={"s1": sink_q}, stop_event=stop,
    )
    worker.start()

    cand_q.put(_CandidateMsg(stream_id="s1", alert=_cand()))
    time.sleep(0.05)    # give the worker time to dequeue + start the verify
    stop.set()
    cand_q.close()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert v.verdicts == 1
    # The verdict reached the per-stream sink queue before shutdown.
    msg = sink_q.get(timeout=0.5)
    assert msg.kind == "alert"
    assert msg.verified.true_alert is True


def test_bounded_queue_preserves_items_across_close_for_workers_to_drain():
    """Contract: a closed, non-empty queue still delivers its backlog. The
    SinkWorker relies on this so alerts committed to the sink_q just
    before stop() still land on disk."""
    q = BoundedQueue(maxsize=8)
    for i in range(3):
        q.put(i)
    q.close()
    assert q.get(timeout=0.1) == 0
    assert q.get(timeout=0.1) == 1
    assert q.get(timeout=0.1) == 2
    with pytest.raises(StopIteration):
        q.get(timeout=0.1)


def test_sink_worker_drains_backlog_after_stop_and_queue_close(tmp_path):
    """A stop request should not discard alerts already committed to the
    per-stream sink queue. Pipeline.stop() closes queues during shutdown;
    SinkWorker must drain the closed queue before exiting."""
    stop = threading.Event()
    q = BoundedQueue(maxsize=16)
    verified = VerifiedAlert(
        candidate=_cand(),
        true_alert=True,
        confidence=0.8,
        false_negative_class=None,
        rationale="queued-before-stop",
    )
    q.put(_SinkMsg(kind="alert", verified=verified))
    stop.set()
    q.close()

    worker = SinkWorker(
        stream_id="s1", out_dir=tmp_path, fps=4.0,
        write_annotated=False, jsonl_name="alerts.jsonl", mp4_name="x.mp4",
        sink_q=q, stop_event=stop,
    )
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "queued-before-stop" in lines[0]


def test_pipeline_stop_keeps_sink_queues_open_until_verifier_finishes():
    """Regression: closing sink queues before joining the verifier could
    make an in-flight verifier lose its final alert with `queue is closed`."""
    from vrs.multistream.pipeline import MultiStreamPipeline

    p = MultiStreamPipeline.__new__(MultiStreamPipeline)
    p._stop = threading.Event()
    p._frame_q = BoundedQueue(maxsize=4)
    p._candidate_q = BoundedQueue(maxsize=4)
    sink_q = BoundedQueue(maxsize=4)
    p._sink_qs = {"s1": sink_q}
    p._decoders = []
    p._detector_worker = None
    p._sinks = []
    p._calibrator = None

    def _finish_in_flight():
        time.sleep(0.05)
        sink_q.put("final-alert")

    p._verifier_worker = threading.Thread(
        target=_finish_in_flight, name="verifier", daemon=True,
    )
    p._verifier_worker.start()

    p.stop()

    assert sink_q.get(timeout=0.1) == "final-alert"
    with pytest.raises(StopIteration):
        sink_q.get(timeout=0.1)


# ──────────────────────────────────────────────────────────────────────
# 5. Multi-stream config / manifest validation
# ──────────────────────────────────────────────────────────────────────

def test_manifest_rejects_non_mapping_yaml(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_multistream_spec(p)


def test_manifest_rejects_missing_streams_key(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("multistream:\n  frame_queue_size: 16\n")
    with pytest.raises(ValueError, match="missing required top-level 'streams'"):
        load_multistream_spec(p)


def test_manifest_rejects_non_list_streams(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("streams:\n  id: oops\n")
    with pytest.raises(ValueError, match="'streams' must be a list"):
        load_multistream_spec(p)


def test_manifest_rejects_stream_without_id(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("streams:\n  - rtsp: rtsp://demo/stream\n")
    with pytest.raises(ValueError, match="missing required key 'id'"):
        load_multistream_spec(p)


def test_manifest_rejects_stream_without_source(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("streams:\n  - id: cam01\n")
    with pytest.raises(ValueError, match="must define one of 'rtsp'"):
        load_multistream_spec(p)


def test_manifest_rejects_non_mapping_stream_entry(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("streams:\n  - just-a-string\n")
    with pytest.raises(ValueError, match=r"streams\[0\] must be a mapping"):
        load_multistream_spec(p)


def test_manifest_accepts_video_and_source_keys_as_aliases(tmp_path: Path):
    """'rtsp', 'source', and 'video' are all legal source keys — useful for
    file-replay and historical-format aliasing without renaming."""
    p = tmp_path / "ok.yaml"
    p.write_text(
        "streams:\n"
        "  - id: cam_a\n    rtsp: rtsp://a\n"
        "  - id: cam_b\n    source: /data/b.mp4\n"
        "  - id: cam_c\n    video: file:///c.mp4\n"
    )
    cfg = load_multistream_spec(p)
    assert len(cfg["streams"]) == 3


def test_multistream_pipeline_rejects_duplicate_ids():
    from vrs.multistream.pipeline import MultiStreamPipeline, StreamSpec
    cfg = {
        "ingest": {"target_fps": 4},
        "detector": {"model": "x", "backend": "ultralytics"},
        "event_state": {"window": 8},
        "verifier": {"enabled": False},
        "sink": {},
    }
    streams = [StreamSpec(id="dup", source="rtsp://a"),
               StreamSpec(id="dup", source="rtsp://b")]
    with pytest.raises(ValueError, match="duplicate stream id"):
        # MultiStreamPipeline actually builds the detector during __init__;
        # we can't fully construct without ultralytics. But the duplicate-id
        # check runs *before* detector construction so it still raises.
        MultiStreamPipeline(cfg, _policy(), streams, out_dir="/tmp/nope")


def test_multistream_pipeline_rejects_empty_streams_list():
    from vrs.multistream.pipeline import MultiStreamPipeline
    with pytest.raises(ValueError, match="no streams configured"):
        MultiStreamPipeline({}, _policy(), streams=[], out_dir="/tmp/nope")


def test_validate_multistream_spec_accepts_non_path_string_source():
    """Pure function, no tmp file — direct cfg dict."""
    _validate_multistream_spec({
        "streams": [{"id": "a", "rtsp": "rtsp://x"}]
    })
