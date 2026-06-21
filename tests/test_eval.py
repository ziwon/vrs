"""Eval-harness unit tests — pure Python, no GPU / video deps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vrs.eval import (
    SCHEMA_VERSION,
    ClassMetrics,
    EvalItem,
    EvalReport,
    GroundTruthEvent,
    HarnessResult,
    bbox_iou_xywh_norm,
    config_for_eval_mode,
    dataset_items_are_images,
    evaluate_detector_only_images,
    score_alerts_against_truth,
)
from vrs.eval.datasets import DFireDataset, LabeledDirDataset, build_dataset
from vrs.eval.metrics import aggregate_scores
from vrs.schemas import CandidateAlert, Detection, VerifiedAlert


def _alert(
    class_name: str,
    peak_pts_s: float,
    *,
    true_alert: bool = True,
    fn_cls: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    verifier_json_valid: bool | None = None,
) -> dict:
    out = {
        "class_name": class_name,
        "peak_pts_s": peak_pts_s,
        "true_alert": true_alert,
        "false_negative_class": fn_cls,
    }
    if bbox is not None:
        out["bbox_xywh_norm"] = bbox
    if verifier_json_valid is not None:
        out["verifier_json_valid"] = verifier_json_valid
    return out


def _event(cls: str, start: float, end: float) -> GroundTruthEvent:
    return GroundTruthEvent(class_name=cls, start_s=start, end_s=end)


# ─── ClassMetrics ──────────────────────────────────────────────────────


def test_class_metrics_handles_zero_denominators():
    cm = ClassMetrics()
    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.f1 == 0.0


def test_class_metrics_f1_from_p_and_r():
    cm = ClassMetrics(tp=3, fp=1, fn=1)
    assert cm.precision == pytest.approx(0.75)
    assert cm.recall == pytest.approx(0.75)
    assert cm.f1 == pytest.approx(0.75)


# ─── matching ─────────────────────────────────────────────────────────


def test_score_basic_tp_fp_fn():
    alerts = [_alert("fire", 3.0), _alert("fire", 20.0)]  # 2nd is FP
    events = [_event("fire", 2.0, 5.0), _event("fire", 40.0, 45.0)]  # 2nd unmatched → FN
    score = score_alerts_against_truth(alerts, events, tolerance_s=0.0)
    fire = score.per_class["fire"]
    assert (fire.tp, fire.fp, fire.fn) == (1, 1, 1)
    assert fire.precision == pytest.approx(0.5)
    assert fire.recall == pytest.approx(0.5)


def test_score_respects_tolerance_boundary():
    """An alert just outside the event window is FP; within tolerance is TP."""
    events = [_event("fire", 10.0, 12.0)]
    s_hit = score_alerts_against_truth([_alert("fire", 12.4)], events, tolerance_s=0.5)
    s_miss = score_alerts_against_truth([_alert("fire", 12.6)], events, tolerance_s=0.5)
    assert s_hit.per_class["fire"].tp == 1
    assert s_miss.per_class["fire"].fp == 1
    assert s_miss.per_class["fire"].fn == 1


def test_score_is_one_to_one_even_if_multiple_alerts_overlap_one_event():
    """Two alerts in the same event window → 1 TP + 1 FP (not 2 TP)."""
    alerts = [_alert("fire", 3.0), _alert("fire", 3.5)]
    events = [_event("fire", 2.0, 5.0)]
    score = score_alerts_against_truth(alerts, events, tolerance_s=0.0)
    fire = score.per_class["fire"]
    assert (fire.tp, fire.fp, fire.fn) == (1, 1, 0)


def test_score_cross_class_mismatch_does_not_credit():
    """A fire alert inside a smoke event must not match."""
    alerts = [_alert("fire", 3.0)]
    events = [_event("smoke", 2.0, 5.0)]
    score = score_alerts_against_truth(alerts, events, tolerance_s=0.0)
    assert score.per_class["fire"].fp == 1
    assert score.per_class["smoke"].fn == 1


def test_score_ignores_verifier_flipped_alerts_for_prf():
    """true_alert=False alerts must NOT count as positives against ground
    truth — they are only reflected in flip_rate."""
    alerts = [
        _alert("fire", 3.0, true_alert=True),
        _alert("fire", 3.5, true_alert=False),  # verifier flipped — shouldn't become FP
    ]
    events = [_event("fire", 2.0, 5.0)]
    score = score_alerts_against_truth(alerts, events, tolerance_s=0.0)
    fire = score.per_class["fire"]
    assert (fire.tp, fire.fp, fire.fn) == (1, 0, 0)


def test_score_flip_rate_and_fn_flag_rate():
    alerts = [
        _alert("fire", 1.0, true_alert=True),
        _alert("fire", 2.0, true_alert=False, verifier_json_valid=False),
        _alert("smoke", 3.0, true_alert=True, fn_cls="fire"),
    ]
    score = score_alerts_against_truth(alerts, [], tolerance_s=0.0)
    assert score.n_alerts_total == 3
    assert score.n_alerts_true == 2
    assert score.n_fn_flagged == 1
    assert score.n_verifier_json_malformed == 1
    assert score.flip_rate == pytest.approx(1 / 3)
    assert score.fn_flag_rate == pytest.approx(1 / 3)
    assert score.malformed_json_rate == pytest.approx(1 / 3)


def test_score_empty_inputs_do_not_blow_up():
    score = score_alerts_against_truth([], [])
    assert score.per_class == {}
    assert score.flip_rate == 0.0
    assert score.overall().f1 == 0.0


def test_score_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        score_alerts_against_truth([], [], tolerance_s=-0.1)


def test_score_restricted_classes_excludes_others():
    alerts = [_alert("fire", 1.0), _alert("smoke", 2.0)]
    events = [_event("smoke", 1.8, 2.2)]
    score = score_alerts_against_truth(alerts, events, classes={"fire"})
    assert set(score.per_class.keys()) == {"fire"}
    # smoke event not in scored set — n_events should reflect that
    assert score.n_events == 0


def test_bbox_iou_xywh_norm_uses_vrs_coordinates():
    assert bbox_iou_xywh_norm((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)
    assert bbox_iou_xywh_norm((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0


def test_bbox_threshold_requires_matching_alert_box():
    events = [
        GroundTruthEvent(
            class_name="fire",
            start_s=0.0,
            end_s=0.0,
            bbox_xywh_norm=(0.5, 0.5, 0.2, 0.2),
        )
    ]
    image_level = score_alerts_against_truth(
        [_alert("fire", 0.0, bbox=(0.1, 0.1, 0.1, 0.1))],
        events,
        tolerance_s=0.0,
    )
    bbox_level = score_alerts_against_truth(
        [_alert("fire", 0.0, bbox=(0.1, 0.1, 0.1, 0.1))],
        events,
        tolerance_s=0.0,
        bbox_iou_threshold=0.5,
    )
    assert image_level.per_class["fire"].tp == 1
    assert (bbox_level.per_class["fire"].tp, bbox_level.per_class["fire"].fn) == (0, 1)


def test_detector_only_alert_serializes_peak_detector_bbox():
    alert = VerifiedAlert(
        candidate=CandidateAlert(
            class_name="fire",
            severity="critical",
            start_pts_s=0.0,
            peak_pts_s=0.0,
            peak_frame_index=0,
            peak_detections=[
                Detection("fire", 0.9, (20.0, 10.0, 60.0, 50.0)),
            ],
            keyframes=[np.zeros((100, 200, 3), dtype=np.uint8)],
        ),
        true_alert=True,
        confidence=1.0,
        false_negative_class=None,
        rationale="verifier disabled",
    )

    assert alert.to_json()["bbox_xywh_norm"] == pytest.approx([0.1, 0.1, 0.2, 0.4])


# ─── aggregation ──────────────────────────────────────────────────────


def test_aggregate_scores_sums_counts_and_recomputes_ratios():
    a = score_alerts_against_truth(
        [_alert("fire", 3.0)],
        [_event("fire", 2.0, 5.0)],
        tolerance_s=0.0,
    )
    b = score_alerts_against_truth(
        [_alert("fire", 99.0)],
        [_event("fire", 10.0, 12.0)],
        tolerance_s=0.0,
    )
    agg = aggregate_scores([a, b])
    fire = agg.per_class["fire"]
    assert (fire.tp, fire.fp, fire.fn) == (1, 1, 1)
    assert agg.n_events == 2


# ─── LabeledDirDataset ────────────────────────────────────────────────


def test_labeled_dir_yields_items_with_events(tmp_path: Path):
    # create two fake mp4s (content doesn't matter — adapter never decodes)
    (tmp_path / "a.mp4").write_bytes(b"\x00" * 32)
    (tmp_path / "b.mp4").write_bytes(b"\x00" * 32)
    (tmp_path / "a.json").write_text(
        json.dumps({"events": [{"class": "fire", "start_s": 1.0, "end_s": 4.0}]})
    )
    # b.mp4 has no sidecar → empty events list (quiet footage)

    items = list(LabeledDirDataset(tmp_path))
    assert [i.video_path.name for i in items] == ["a.mp4", "b.mp4"]
    assert items[0].events == [GroundTruthEvent("fire", 1.0, 4.0)]
    assert items[1].events == []


def test_labeled_dir_accepts_common_video_suffixes(tmp_path: Path):
    (tmp_path / "clip.avi").write_bytes(b"\x00" * 32)
    (tmp_path / "ignore.txt").write_text("nope")
    (tmp_path / "clip.json").write_text(
        json.dumps({"events": [{"class": "fire", "start_s": 0.0, "end_s": 1.0}]})
    )

    items = list(LabeledDirDataset(tmp_path))

    assert [item.video_path.name for item in items] == ["clip.avi"]
    assert items[0].events == [GroundTruthEvent("fire", 0.0, 1.0)]


def test_labeled_dir_rejects_nonexistent_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LabeledDirDataset(tmp_path / "does-not-exist")


# ─── DFireDataset ─────────────────────────────────────────────────────


def _make_dfire_root(tmp_path: Path) -> Path:
    root = tmp_path / "dfire-mini"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    return root


def test_dfire_yields_image_items_with_bbox_events(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    (root / "images" / "a.jpg").write_bytes(b"fake image bytes")
    (root / "images" / "b.png").write_bytes(b"fake image bytes")
    (root / "labels" / "a.txt").write_text(
        "1 0.50 0.50 0.20 0.30\n0 0.25 0.25 0.10 0.10\n",
        encoding="utf-8",
    )
    (root / "labels" / "b.txt").write_text("", encoding="utf-8")

    items = list(DFireDataset(root))

    assert [item.video_path.name for item in items] == ["a.jpg", "b.png"]
    assert items[0].events == [
        GroundTruthEvent("fire", 0.0, 0.0, (0.40, 0.35, 0.20, 0.30)),
        GroundTruthEvent("smoke", 0.0, 0.0, (0.20, 0.20, 0.10, 0.10)),
    ]
    assert items[1].events == []


def test_dfire_missing_label_file_is_quiet_by_default(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    (root / "images" / "quiet.jpg").write_bytes(b"fake image bytes")

    [item] = list(DFireDataset(root))

    assert item.video_path.name == "quiet.jpg"
    assert item.events == []


def test_dfire_can_require_label_files(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    (root / "images" / "quiet.jpg").write_bytes(b"fake image bytes")

    with pytest.raises(FileNotFoundError):
        list(DFireDataset(root, require_labels=True))


def test_dfire_rejects_malformed_yolo_rows(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    (root / "images" / "bad.jpg").write_bytes(b"fake image bytes")
    (root / "labels" / "bad.txt").write_text("1 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 5 YOLO fields"):
        list(DFireDataset(root))


def test_dataset_registry_builds_dfire(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    (root / "images" / "a.jpg").write_bytes(b"fake image bytes")
    (root / "labels" / "a.txt").write_text("", encoding="utf-8")

    dataset = build_dataset("dfire", root)

    assert isinstance(dataset, DFireDataset)


def test_evaluate_detector_only_images_reuses_detector_and_scores_boxes(tmp_path: Path):
    root = _make_dfire_root(tmp_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / "images" / "a.jpg"), image)
    assert cv2.imwrite(str(root / "images" / "b.jpg"), image)
    (root / "labels" / "a.txt").write_text("1 0.50 0.50 0.20 0.20\n", encoding="utf-8")
    (root / "labels" / "b.txt").write_text("", encoding="utf-8")

    class _Detector:
        def __init__(self):
            self.calls = 0

        def __call__(self, frame):
            self.calls += 1
            if self.calls == 1:
                return [
                    Detection(
                        class_name="fire",
                        score=0.9,
                        xyxy=(40.0, 40.0, 60.0, 60.0),
                    )
                ]
            return []

        def batch(self, frames):
            return [self(frame) for frame in frames]

    detector = _Detector()
    result = evaluate_detector_only_images(
        dataset=DFireDataset(root),
        detector=detector,
        out_dir=tmp_path / "out",
        bbox_iou_threshold=0.5,
    )

    assert detector.calls == 2
    assert (tmp_path / "out" / "a" / "alerts.jsonl").exists()
    assert (tmp_path / "out" / "b" / "alerts.jsonl").exists()
    assert result.aggregate.per_class["fire"].tp == 1
    assert result.aggregate.per_class["fire"].fn == 0
    assert result.aggregate.per_class.get("smoke", ClassMetrics()).tp == 0


def test_stream_reader_yields_single_frame_for_image(tmp_path: Path):
    from vrs.ingest import StreamReader

    image_path = tmp_path / "frame.jpg"
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    frames = list(StreamReader(str(image_path), target_fps=4))

    assert len(frames) == 1
    assert frames[0].index == 0
    assert frames[0].pts_s == 0.0
    assert frames[0].image.shape == (8, 12, 3)


def test_dataset_items_are_images_detects_media_kind():
    class _Dataset:
        def __init__(self, names):
            self.names = names

        def __iter__(self):
            for name in self.names:
                yield EvalItem(Path(name), [])

    assert dataset_items_are_images(_Dataset(["a.jpg", "b.png"])) is True
    assert dataset_items_are_images(_Dataset(["a.jpg", "b.mp4"])) is False
    assert dataset_items_are_images(_Dataset(["clip.mp4"])) is False
    assert dataset_items_are_images(_Dataset([])) is True


# ─── harness integration (stubbed pipeline) ────────────────────────────


class _StubPipeline:
    """Pipeline stand-in: writes a caller-supplied alerts.jsonl in its out dir."""

    def __init__(self, out_dir: Path, alerts_by_video: dict[str, list[dict]]):
        self.out_dir = Path(out_dir)
        self._alerts_by_video = alerts_by_video

    def run(self, source: str) -> None:
        src_stem = Path(source).stem
        alerts = self._alerts_by_video.get(src_stem, [])
        with (self.out_dir / "alerts.jsonl").open("w", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps(a) + "\n")


def test_harness_scores_per_video_and_aggregates(tmp_path: Path):
    from vrs.eval import evaluate

    root = tmp_path / "dataset"
    root.mkdir()
    # Two videos: one with a correctly-flagged fire, one with a flipped alert
    # whose ground-truth smoke event is never reported.
    (root / "v1.mp4").write_bytes(b"\0")
    (root / "v1.json").write_text(
        json.dumps({"events": [{"class": "fire", "start_s": 1.0, "end_s": 3.0}]})
    )
    (root / "v2.mp4").write_bytes(b"\0")
    (root / "v2.json").write_text(
        json.dumps({"events": [{"class": "smoke", "start_s": 5.0, "end_s": 8.0}]})
    )

    alerts_per_video = {
        "v1": [_alert("fire", 2.0, true_alert=True)],
        "v2": [_alert("fire", 6.0, true_alert=False, fn_cls="smoke")],
    }

    def factory(od):
        return _StubPipeline(od, alerts_per_video)

    result = evaluate(
        dataset=LabeledDirDataset(root),
        pipeline_factory=factory,
        out_dir=tmp_path / "out",
        tolerance_s=0.5,
    )

    assert len(result.per_video) == 2
    # Per-video dirs were created and populated
    assert (tmp_path / "out" / "v1" / "alerts.jsonl").exists()
    assert (tmp_path / "out" / "v2" / "alerts.jsonl").exists()

    agg = result.aggregate
    fire = agg.per_class["fire"]
    smoke = agg.per_class["smoke"]
    # v1 fire alert matches the fire event → 1 TP
    assert (fire.tp, fire.fp, fire.fn) == (1, 0, 0)
    # v2 smoke event unmatched (only a flipped fire alert existed) → 1 FN
    assert smoke.fn == 1
    # Counters
    assert agg.n_alerts_total == 2
    assert agg.n_alerts_true == 1
    assert agg.n_fn_flagged == 1
    assert agg.flip_rate == pytest.approx(0.5)

    # to_dict is JSON-serializable — smoke-test the CLI report shape
    blob = json.dumps(result.to_dict())
    assert "aggregate" in blob and "per_video" in blob


def test_eval_report_round_trip_is_stable():
    score = score_alerts_against_truth(
        [
            _alert("smoke", 8.0, true_alert=False, fn_cls="smoke"),
            _alert("fire", 2.0, true_alert=True),
        ],
        [
            _event("fire", 1.0, 3.0),
            _event("smoke", 7.0, 9.0),
        ],
        tolerance_s=0.0,
    )
    result = HarnessResult(
        aggregate=score,
        per_video=[
            (Path("b.mp4"), score),
            (Path("a.mp4"), score),
        ],
    )
    result.aggregate.detector_latencies_ms.extend([10.0, 20.0, 30.0])
    result.aggregate.verifier_latencies_ms.extend([100.0, 200.0, 300.0])
    result.aggregate.verifier_tokens_per_second.extend([5.0, 10.0, 20.0])
    report = EvalReport.from_harness_result(
        result,
        dataset="fixtures/mini-dataset",
        config_path="configs/default.yaml",
        policy_path="configs/policies/safety.yaml",
        config={
            "detector": {"backend": "ultralytics", "model": "yoloe-11l-seg.pt"},
            "verifier": {
                "enabled": True,
                "backend": "transformers",
                "model_id": "nvidia/Cosmos-Reason2-2B",
            },
        },
        created_at=datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
    )

    assert report.schema_version == SCHEMA_VERSION
    assert report.run.mode == "full_cascade"
    assert report.run.run_id == "2026-04-26-mini-dataset-yoloe-11l-seg-cosmos-reason2-2b"
    assert [entry.video for entry in report.per_video] == ["a.mp4", "b.mp4"]
    assert report.metrics.overall.tp == 1
    assert report.metrics.per_class["fire"].f1 == pytest.approx(1.0)
    assert report.detector_quality is None
    assert report.full_cascade_quality == report.metrics
    assert report.quality_signals.verifier_flip_rate == pytest.approx(0.5)
    assert report.quality_signals.false_negative_flag_rate == pytest.approx(0.5)
    assert report.quality_signals.malformed_json_rate == pytest.approx(0.0)
    assert report.latency.detector_p50_ms == pytest.approx(20.0)
    assert report.latency.detector_p95_ms == pytest.approx(29.0)
    assert report.latency.detector_p99_ms == pytest.approx(29.8)
    assert report.latency.verifier_p50_ms == pytest.approx(200.0)
    assert report.latency.verifier_p95_ms == pytest.approx(290.0)
    assert report.latency.verifier_p99_ms == pytest.approx(298.0)
    assert report.latency.verifier_tokens_per_second_p50 == pytest.approx(10.0)
    assert report.latency.verifier_tokens_per_second_p95 == pytest.approx(19.0)

    payload = report.to_dict()
    assert list(payload["metrics"]["per_class"].keys()) == ["fire", "smoke"]
    assert payload["detector_quality"] is None
    assert payload["full_cascade_quality"] == payload["metrics"]
    assert EvalReport.from_dict(payload) == report
    assert json.loads(report.to_json()) == payload


def test_config_for_eval_mode_marks_detector_only_without_mutating_source():
    config = {
        "detector": {"backend": "ultralytics", "model": "yoloe-11l-seg.pt"},
        "verifier": {
            "enabled": True,
            "backend": "transformers",
            "model_id": "nvidia/Cosmos-Reason2-2B",
        },
    }

    effective = config_for_eval_mode(config, "detector_only")

    assert config["verifier"]["enabled"] is True
    assert effective["verifier"]["enabled"] is False

    report = EvalReport.from_harness_result(
        HarnessResult(aggregate=score_alerts_against_truth([], [])),
        dataset="fixtures/mini-dataset",
        config_path="configs/default.yaml",
        policy_path="configs/policies/safety.yaml",
        config=effective,
        created_at=datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
    )
    assert report.run.mode == "detector_only"
    assert report.models.detector is not None
    assert report.models.verifier is None
    assert report.detector_quality == report.metrics
    assert report.full_cascade_quality is None


def test_eval_cli_accepts_detector_only_mode():
    from scripts.eval import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "fixtures/mini-dataset",
            "--dataset-format",
            "dfire",
            "--config",
            "configs/default.yaml",
            "--policy",
            "configs/policies/safety.yaml",
            "--mode",
            "detector_only",
            "--out",
            "runs/eval-detector",
            "--bbox-iou-threshold",
            "0.5",
        ]
    )
    assert args.mode == "detector_only"
    assert args.dataset_format == "dfire"
    assert args.bbox_iou_threshold == 0.5


def test_dfire_threshold_sweep_scores_cached_alerts():
    from scripts.sweep_dfire_thresholds import (
        CachedItem,
        config_with_conf_floor,
        parse_thresholds,
        score_cached_items,
    )

    thresholds = parse_thresholds("0.30, 0.10, 0.10")
    assert thresholds == [0.10, 0.30]

    cached = [
        CachedItem(
            image_path=Path("a.jpg"),
            events=[GroundTruthEvent("fire", 0.0, 0.0, (0.4, 0.4, 0.2, 0.2))],
            alerts=[
                _alert("fire", 0.0, bbox=(0.4, 0.4, 0.2, 0.2)) | {"confidence": 0.25},
                _alert("smoke", 0.0, bbox=(0.1, 0.1, 0.1, 0.1)) | {"confidence": 0.80},
            ],
        )
    ]

    loose = score_cached_items(
        cached,
        {"fire": 0.20, "smoke": 0.20},
        bbox_iou_threshold=0.5,
    )
    strict = score_cached_items(
        cached,
        {"fire": 0.30, "smoke": 0.20},
        bbox_iou_threshold=0.5,
    )

    assert loose.per_class["fire"].tp == 1
    assert strict.per_class["fire"].fn == 1
    assert strict.per_class["smoke"].fp == 1

    tuned = config_with_conf_floor(
        {"detector": {"conf_floor": 0.20, "model": "fake.pt"}},
        {"fire": 0.05, "smoke": 0.10},
    )
    assert tuned["detector"]["conf_floor"] == 0.05


def test_dfire_prompt_sweep_builds_policy_and_config():
    from scripts.sweep_dfire_prompts import (
        config_with_model_and_thresholds,
        load_prompt_sets,
        parse_models,
        policy_yaml_with_prompts_and_thresholds,
    )

    assert parse_models("a.pt, b.pt") == ["a.pt", "b.pt"]
    [baseline, *_] = load_prompt_sets(None)
    assert baseline["name"] == "baseline"

    policy = {
        "watch": [
            {
                "name": "fire",
                "detector": ["fire"],
                "verifier": "fire visible",
                "severity": "critical",
                "min_score": 0.30,
                "min_persist_frames": 1,
            },
            {
                "name": "smoke",
                "detector": ["smoke"],
                "verifier": "smoke visible",
                "severity": "high",
                "min_score": 0.25,
                "min_persist_frames": 1,
            },
        ]
    }
    prompts = {
        "name": "custom",
        "prompts": {"fire": ["flame"], "smoke": ["dark smoke", "white smoke"]},
    }

    tuned_policy = policy_yaml_with_prompts_and_thresholds(
        policy,
        prompts,
        {"fire": 0.05, "smoke": 0.10},
    )
    assert tuned_policy["watch"][0]["detector"] == ["flame"]
    assert tuned_policy["watch"][0]["min_score"] == 0.05
    assert tuned_policy["watch"][1]["detector"] == ["dark smoke", "white smoke"]

    tuned_config = config_with_model_and_thresholds(
        {"detector": {"model": "old.pt", "conf_floor": 0.20}},
        model="new.pt",
        thresholds={"fire": 0.05, "smoke": 0.10},
    )
    assert tuned_config["detector"]["model"] == "new.pt"
    assert tuned_config["detector"]["conf_floor"] == 0.05


def test_detector_model_refresh_helpers_choose_decision():
    from scripts.eval_detector_models import (
        add_baseline_deltas,
        choose_best,
        make_decision,
        parse_models,
        summarize_latency,
    )

    assert parse_models("yoloe-11l-seg.pt, yoloe-26l-seg.pt") == [
        "yoloe-11l-seg.pt",
        "yoloe-26l-seg.pt",
    ]
    assert summarize_latency([10.0, 20.0, 30.0]) == {
        "count": 3,
        "mean_ms": 20.0,
        "p50_ms": 20.0,
        "p95_ms": 29.0,
    }

    rows = add_baseline_deltas(
        [
            {
                "model": "yoloe-11l-seg.pt",
                "metrics": {
                    "macro_f1": 0.50,
                    "overall": {"f1": 0.50, "recall": 0.60, "precision": 0.45},
                },
                "latency": {"p95_ms": 20.0},
            },
            {
                "model": "yoloe-26l-seg.pt",
                "metrics": {
                    "macro_f1": 0.54,
                    "overall": {"f1": 0.55, "recall": 0.62, "precision": 0.50},
                },
                "latency": {"p95_ms": 21.0},
            },
        ],
        "yoloe-11l-seg.pt",
    )

    assert rows[1]["delta_vs_baseline"]["macro_f1"] == pytest.approx(0.04)
    assert choose_best(rows, "macro_f1")["model"] == "yoloe-26l-seg.pt"
    decision = make_decision(
        rows,
        baseline_model="yoloe-11l-seg.pt",
        optimize="macro_f1",
        min_metric_gain=0.01,
        max_p95_latency_ratio=1.10,
    )
    assert decision["action"] == "adopt_candidate"
    assert decision["p95_latency_ratio"] == pytest.approx(1.05)


def test_detector_model_refresh_config_sets_model_and_floor():
    from scripts.eval_detector_models import config_with_model
    from vrs.policy.watch_policy import WatchItem, WatchPolicy

    policy = WatchPolicy(
        [
            WatchItem("fire", ["fire"], "fire visible", "critical", 0.30, 1),
            WatchItem("smoke", ["smoke"], "smoke visible", "high", 0.25, 1),
        ]
    )
    cfg = config_with_model(
        {"detector": {"model": "old.pt", "conf_floor": 0.40}},
        model="new.pt",
        policy=policy,
    )

    assert cfg["detector"]["model"] == "new.pt"
    assert cfg["detector"]["conf_floor"] == 0.25


def test_detector_model_refresh_detects_half_dtype_mismatch():
    from scripts.eval_detector_models import _is_half_dtype_mismatch

    assert _is_half_dtype_mismatch(
        RuntimeError("expected mat1 and mat2 to have the same dtype, but got: c10::Half != float")
    )
    assert not _is_half_dtype_mismatch(RuntimeError("CUDA out of memory"))


def test_verifier_bakeoff_summarizes_existing_reports(tmp_path: Path):
    from scripts.eval_verifier_backends import parse_candidate, run_bakeoff

    candidate = parse_candidate("cosmos=configs/default.yaml")
    score = score_alerts_against_truth(
        [_alert("fire", 2.0, true_alert=True, verifier_json_valid=False)],
        [_event("fire", 1.0, 3.0)],
        tolerance_s=0.0,
    )
    score.detector_latencies_ms.extend([1.0, 3.0])
    score.verifier_latencies_ms.extend([100.0, 300.0])
    report = EvalReport.from_harness_result(
        HarnessResult(aggregate=score, per_video=[(Path("clip.mp4"), score)]),
        dataset="fixtures/verifier",
        config_path="configs/default.yaml",
        policy_path="configs/policies/safety.yaml",
        config={
            "detector": {"backend": "ultralytics", "model": "yoloe-11l-seg.pt"},
            "verifier": {
                "enabled": True,
                "backend": "transformers",
                "model_id": "nvidia/Cosmos-Reason2-2B",
            },
        },
        created_at=datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
    )
    report_dir = tmp_path / "bakeoff" / candidate.name
    report_dir.mkdir(parents=True)
    report.write(report_dir / "report.json")

    comparison = run_bakeoff(
        candidates=[candidate],
        dataset=Path("fixtures/verifier"),
        dataset_format="labeled_dir",
        policy=Path("configs/policies/safety.yaml"),
        out_dir=tmp_path / "bakeoff",
        tolerance_s=0.5,
        skip_run=True,
    )

    row = comparison["candidates"][0]
    assert row["name"] == "cosmos"
    assert row["metrics"]["overall"]["f1"] == pytest.approx(1.0)
    assert row["quality_signals"]["malformed_json_rate"] == pytest.approx(1.0)
    assert row["latency"]["verifier_p95_ms"] == pytest.approx(290.0)
    assert (tmp_path / "bakeoff" / "verifier_bakeoff.json").exists()


def test_prepare_verifier_eval_dataset_writes_sidecars(monkeypatch, tmp_path: Path):
    from scripts.prepare_verifier_eval_dataset import prepare_dataset

    positive_root = tmp_path / "positive"
    negative_root = tmp_path / "negative"
    positive_root.mkdir()
    negative_root.mkdir()
    (positive_root / "fire one.avi").write_bytes(b"video")
    (negative_root / "quiet.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "scripts.prepare_verifier_eval_dataset.video_duration_s", lambda path: 12.345
    )

    manifest = prepare_dataset(
        out_dir=tmp_path / "out",
        positive_roots=[positive_root],
        positive_class="fire",
        positive_limit=1,
        negative_roots=[negative_root],
        negative_limit=1,
        copy_mode="symlink",
        overwrite=True,
    )

    assert manifest["count"] == 2
    positive_sidecar = Path(manifest["entries"][0]["sidecar"])
    negative_sidecar = Path(manifest["entries"][1]["sidecar"])
    assert json.loads(positive_sidecar.read_text()) == {
        "events": [{"class": "fire", "start_s": 0.0, "end_s": 12.345}]
    }
    assert json.loads(negative_sidecar.read_text()) == {"events": []}
    assert Path(manifest["entries"][0]["video"]).is_symlink()


def test_smoke_verifier_backend_writes_result(monkeypatch, tmp_path: Path):
    from scripts import smoke_verifier_backend

    config_path = tmp_path / "config.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "smoke.json"
    config_path.write_text(
        "\n".join(
            [
                "ingest:",
                "  target_fps: 4",
                "detector:",
                "  model: fake.pt",
                "event_state:",
                "  window: 2",
                "verifier:",
                "  enabled: true",
                "  backend: openai_compatible",
                "  model_id: qwen-vl-served",
                "  base_url: http://127.0.0.1:1/v1",
                "sink: {}",
            ]
        )
    )
    policy_path.write_text(
        "\n".join(
            [
                "watch:",
                "  - name: fire",
                "    detector: ['fire']",
                "    verifier: 'open flames'",
                "    severity: critical",
                "    min_score: 0.3",
                "    min_persist_frames: 1",
            ]
        )
    )

    class _FakeVerifier:
        def verify(self, candidate):
            return VerifiedAlert(
                candidate=candidate,
                true_alert=True,
                confidence=0.8,
                false_negative_class=None,
                rationale="ok",
                verifier_raw='{"true_alert":true}',
                verifier_json_valid=True,
            )

    monkeypatch.setattr(
        smoke_verifier_backend, "build_verifier", lambda config, policy: _FakeVerifier()
    )

    payload = smoke_verifier_backend.run_smoke(
        config_path=config_path,
        policy_path=policy_path,
        class_name="fire",
        image_path=None,
        out_path=out_path,
    )

    assert payload["true_alert"] is True
    assert payload["verifier_json_valid"] is True
    assert payload["smoke"]["backend"] == "openai_compatible"
    assert json.loads(out_path.read_text()) == payload


def test_detector_only_pipeline_construction_skips_verifier(monkeypatch, tmp_path: Path):
    from vrs.pipeline import VRSPipeline
    from vrs.policy.watch_policy import WatchItem, WatchPolicy

    class _FakeDetector:
        def __call__(self, frame):
            return []

        def batch(self, frames):
            return [[] for _ in frames]

    def fail_build_vlm_backend(*args, **kwargs):
        raise AssertionError("detector-only eval must not construct a VLM backend")

    monkeypatch.setattr("vrs.pipeline.build_detector", lambda *args, **kwargs: _FakeDetector())
    monkeypatch.setattr("vrs.pipeline.build_vlm_backend", fail_build_vlm_backend)

    policy = WatchPolicy(
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
    config = config_for_eval_mode(
        {
            "ingest": {"target_fps": 4},
            "detector": {"backend": "ultralytics", "model": "fake.pt"},
            "event_state": {"window": 2},
            "tracker": {"backend": "none"},
            "verifier": {
                "enabled": True,
                "backend": "transformers",
                "model_id": "nvidia/Cosmos-Reason2-2B",
            },
            "sink": {"write_thumbnails": False, "write_annotated": False},
            "calibration": {"enabled": False},
        },
        "detector_only",
    )

    pipeline = VRSPipeline(config, policy, tmp_path)

    assert pipeline.verifier is None


def test_vllm_pipeline_constructs_verifier_before_detector(monkeypatch, tmp_path: Path):
    from vrs.pipeline import VRSPipeline
    from vrs.policy.watch_policy import WatchItem, WatchPolicy

    calls: list[str] = []

    class _FakeDetector:
        def __call__(self, frame):
            return []

        def batch(self, frames):
            return [[] for _ in frames]

    class _FakeVLM:
        def __init__(self):
            self.last_generation_stats = {}

        def chat_video(self, *args, **kwargs):
            return "{}"

    def fake_build_detector(*args, **kwargs):
        calls.append("detector")
        return _FakeDetector()

    def fake_build_vlm_backend(*args, **kwargs):
        calls.append("verifier")
        return _FakeVLM()

    monkeypatch.setattr("vrs.pipeline.build_detector", fake_build_detector)
    monkeypatch.setattr("vrs.pipeline.build_vlm_backend", fake_build_vlm_backend)

    policy = WatchPolicy(
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
    config = {
        "ingest": {"target_fps": 4},
        "detector": {"backend": "ultralytics", "model": "fake.pt"},
        "event_state": {"window": 2},
        "tracker": {"backend": "none"},
        "verifier": {
            "enabled": True,
            "backend": "vllm",
            "model_id": "nvidia/Cosmos-Reason2-2B",
        },
        "sink": {"write_thumbnails": False, "write_annotated": False},
        "calibration": {"enabled": False},
    }

    VRSPipeline(config, policy, tmp_path)

    assert calls == ["verifier", "detector"]


def _report(
    per_class: dict[str, float],
    overall_f1: float = 0.0,
    flip_rate: float = 0.0,
    fn_flag_rate: float = 0.0,
) -> dict:
    """Minimal report.json-shaped fixture — f1-only per class, all other
    ClassMetrics fields zeroed out since the gate only reads f1."""
    return {
        "aggregate": {
            "per_class": {
                cls: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": f1}
                for cls, f1 in per_class.items()
            },
            "overall": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": overall_f1,
            },
            "flip_rate": flip_rate,
            "fn_flag_rate": fn_flag_rate,
        },
        "per_video": [],
    }


# ─── CI regression gate ────────────────────────────────────────────────


def test_gate_passes_when_reports_are_equal():
    from vrs.eval.ci import compare_reports

    r = _report({"fire": 0.80, "smoke": 0.70}, overall_f1=0.75)
    result = compare_reports(r, r, max_f1_drop=0.02)
    assert result.passed is True
    assert result.regressions() == []


def test_gate_passes_on_improvement():
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.70}, overall_f1=0.70)
    current = _report({"fire": 0.85}, overall_f1=0.85)
    assert compare_reports(baseline, current).passed is True


def test_gate_passes_within_tolerance():
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80}, overall_f1=0.80)
    current = _report({"fire": 0.785}, overall_f1=0.785)  # drop of 0.015 < 0.02
    assert compare_reports(baseline, current, max_f1_drop=0.02).passed is True


def test_gate_fails_on_per_class_regression():
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80, "smoke": 0.70}, overall_f1=0.75)
    current = _report({"fire": 0.50, "smoke": 0.70}, overall_f1=0.60)
    result = compare_reports(baseline, current, max_f1_drop=0.02)
    assert result.passed is False
    regressed = [d.class_name for d in result.regressions()]
    assert "fire" in regressed
    assert "smoke" not in regressed


def test_gate_fails_on_overall_regression_even_if_per_class_pass():
    """Per-class values are rounded in the report but overall still shifts;
    the gate must look at overall independently."""
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80}, overall_f1=0.80)
    current = _report({"fire": 0.80}, overall_f1=0.60)  # overall dropped 0.20
    result = compare_reports(baseline, current, max_f1_drop=0.02)
    assert result.passed is False
    assert result.overall.regressed is True


def test_gate_treats_missing_class_in_current_as_regression():
    """A class present in baseline but absent from current → implicit F1=0 →
    regression (unless baseline had F1=0 too)."""
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80, "smoke": 0.70}, overall_f1=0.75)
    current = _report({"fire": 0.80}, overall_f1=0.80)
    result = compare_reports(baseline, current)
    smoke = next(d for d in result.per_class if d.class_name == "smoke")
    assert smoke.regressed is True
    assert "missing" in smoke.note
    assert result.passed is False


def _schema_v1_report(
    per_class: dict[str, float],
    *,
    overall_f1: float = 0.0,
    flip_rate: float = 0.0,
    fn_flag_rate: float = 0.0,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": "2026-04-26-mini-dataset-yoloe-11l-seg-cosmos-reason2-2b",
            "created_at": "2026-04-26T00:00:00Z",
            "dataset": "mini-dataset",
            "mode": "full_cascade",
            "policy_path": "configs/policies/safety.yaml",
            "config_path": "configs/default.yaml",
        },
        "models": {
            "detector": {"backend": "ultralytics", "model": "yoloe-11l-seg.pt"},
            "verifier": {"backend": "transformers", "model": "nvidia/Cosmos-Reason2-2B"},
        },
        "metrics": {
            "per_class": {
                cls: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": f1}
                for cls, f1 in per_class.items()
            },
            "overall": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": overall_f1,
            },
        },
        "latency": {
            "detector_p50_ms": None,
            "detector_p95_ms": None,
            "verifier_p50_ms": None,
            "verifier_p95_ms": None,
        },
        "runtime": {
            "python": None,
            "torch": None,
            "cuda": None,
            "gpu_name": None,
            "peak_vram_mb": None,
        },
        "quality_signals": {
            "verifier_flip_rate": flip_rate,
            "false_negative_flag_rate": fn_flag_rate,
            "malformed_json_rate": None,
            "queue_drops": None,
        },
        "per_video": [],
    }


def test_gate_reads_schema_v1_reports():
    from vrs.eval.ci import compare_reports

    baseline = _schema_v1_report({"fire": 0.80}, overall_f1=0.80, flip_rate=0.10)
    current = _schema_v1_report({"fire": 0.78}, overall_f1=0.78, flip_rate=0.12)
    result = compare_reports(baseline, current, max_f1_drop=0.05)
    assert result.passed is True
    assert result.baseline_flip_rate == pytest.approx(0.10)
    assert result.current_flip_rate == pytest.approx(0.12)


def test_eval_report_infers_quality_sections_for_legacy_payloads():
    full_payload = _schema_v1_report({"fire": 0.80}, overall_f1=0.80)
    full_report = EvalReport.from_dict(full_payload)
    assert full_report.detector_quality is None
    assert full_report.full_cascade_quality == full_report.metrics

    detector_payload = _schema_v1_report({"fire": 0.80}, overall_f1=0.80)
    detector_payload["run"]["mode"] = "detector_only"
    detector_payload["models"]["verifier"] = None
    detector_report = EvalReport.from_dict(detector_payload)
    assert detector_report.detector_quality == detector_report.metrics
    assert detector_report.full_cascade_quality is None


def test_committed_eval_baseline_matches_regeneration_script():
    from scripts.write_eval_baseline import build_baseline_report

    baseline_path = Path("baselines/eval/report.json")
    committed = json.loads(baseline_path.read_text(encoding="utf-8"))
    regenerated = build_baseline_report().to_dict()

    assert committed == regenerated


def test_gate_accepts_committed_eval_baseline():
    from vrs.eval.ci import compare_reports

    baseline = json.loads(Path("baselines/eval/report.json").read_text(encoding="utf-8"))
    current = _schema_v1_report(
        {"fire": 0.74, "smoke": 0.79},
        overall_f1=0.76,
        flip_rate=0.125,
        fn_flag_rate=0.125,
    )

    result = compare_reports(baseline, current, max_f1_drop=0.02)

    assert result.passed is True
    assert result.overall.regressed is False
    assert {d.class_name for d in result.per_class} == {"fire", "smoke"}


def test_gate_supports_legacy_vs_schema_v1_reports():
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80}, overall_f1=0.80, flip_rate=0.10, fn_flag_rate=0.01)
    current = _schema_v1_report({"fire": 0.70}, overall_f1=0.70, flip_rate=0.20, fn_flag_rate=0.03)
    result = compare_reports(baseline, current, max_f1_drop=0.05)
    assert result.passed is False
    assert result.overall.regressed is True
    assert result.current_fn_flag_rate == pytest.approx(0.03)


def test_gate_welcomes_new_class_in_current():
    """A class that appears only in the current report is informational,
    never a regression."""
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80}, overall_f1=0.80)
    current = _report({"fire": 0.80, "weapon": 0.50}, overall_f1=0.80)
    result = compare_reports(baseline, current)
    weapon = next(d for d in result.per_class if d.class_name == "weapon")
    assert weapon.regressed is False
    assert weapon.note == "new class"
    assert result.passed is True


def test_gate_respects_classes_filter():
    from vrs.eval.ci import compare_reports

    baseline = _report({"fire": 0.80, "smoke": 0.70}, overall_f1=0.75)
    current = _report({"fire": 0.80, "smoke": 0.30}, overall_f1=0.75)
    # smoke tanked but we're only gating on fire → pass
    result = compare_reports(baseline, current, classes={"fire"})
    assert result.passed is True
    assert {d.class_name for d in result.per_class} == {"fire"}


def test_gate_rejects_malformed_report():
    from vrs.eval.ci import compare_reports

    with pytest.raises(ValueError):
        compare_reports({}, _report({"fire": 0.8}))


def test_gate_rejects_non_object_report():
    from vrs.eval.ci import compare_reports

    with pytest.raises(ValueError, match="expected object at 'baseline'"):
        compare_reports([], _report({"fire": 0.8}))


def test_gate_rejects_structural_metric_errors():
    from vrs.eval.ci import compare_reports

    baseline = _schema_v1_report({"fire": 0.80}, overall_f1=0.80)
    current = _schema_v1_report({"fire": 0.79}, overall_f1=0.79)
    del current["metrics"]["per_class"]["fire"]["f1"]

    with pytest.raises(ValueError, match=r"metrics\.per_class\.fire\.f1"):
        compare_reports(baseline, current)


def test_gate_treats_null_quality_signals_as_zero():
    from vrs.eval.ci import compare_reports

    baseline = _schema_v1_report({"fire": 0.80}, overall_f1=0.80)
    current = _schema_v1_report({"fire": 0.80}, overall_f1=0.80)
    current["quality_signals"]["verifier_flip_rate"] = None
    current["quality_signals"]["false_negative_flag_rate"] = None

    result = compare_reports(baseline, current)

    assert result.passed is True
    assert result.current_flip_rate == 0.0
    assert result.current_fn_flag_rate == 0.0


def test_gate_cli_exit_codes(tmp_path: Path):
    import json as _json

    from vrs.eval.ci import main as ci_main

    baseline = tmp_path / "baseline.json"
    current_pass = tmp_path / "current_pass.json"
    current_fail = tmp_path / "current_fail.json"
    current_invalid = tmp_path / "current_invalid.json"
    current_non_object = tmp_path / "current_non_object.json"

    baseline.write_text(_json.dumps(_report({"fire": 0.80}, overall_f1=0.80)))
    current_pass.write_text(_json.dumps(_report({"fire": 0.79}, overall_f1=0.79)))
    current_fail.write_text(_json.dumps(_report({"fire": 0.50}, overall_f1=0.50)))
    current_invalid.write_text(_json.dumps({"metrics": {"per_class": {"fire": {}}}}))
    current_non_object.write_text(_json.dumps([]))

    assert ci_main(["--baseline", str(baseline), "--current", str(current_pass)]) == 0
    assert ci_main(["--baseline", str(baseline), "--current", str(current_fail)]) == 1
    assert ci_main(["--baseline", str(baseline), "--current", str(current_invalid)]) == 2
    assert ci_main(["--baseline", str(baseline), "--current", str(current_non_object)]) == 2
    assert (
        ci_main(["--baseline", str(baseline), "--current", str(tmp_path / "nonexistent.json")]) == 2
    )


def test_harness_survives_a_failing_pipeline_run(tmp_path: Path):
    """One bad clip must not abort the entire eval — it scores as no alerts
    and the harness keeps going."""
    from vrs.eval import evaluate

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "good.mp4").write_bytes(b"\0")
    (root / "good.json").write_text(
        json.dumps({"events": [{"class": "fire", "start_s": 1.0, "end_s": 3.0}]})
    )
    (root / "bad.mp4").write_bytes(b"\0")
    (root / "bad.json").write_text(json.dumps({"events": []}))

    class _Factory:
        def __call__(self, od: Path):
            return _GoodOrBad(od)

    class _GoodOrBad:
        def __init__(self, od: Path):
            self.od = Path(od)

        def run(self, source: str) -> None:
            if Path(source).stem == "bad":
                raise RuntimeError("decoder exploded")
            with (self.od / "alerts.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(_alert("fire", 2.0)) + "\n")

    result = evaluate(
        dataset=LabeledDirDataset(root),
        pipeline_factory=_Factory(),
        out_dir=tmp_path / "out",
        tolerance_s=0.5,
    )
    assert len(result.per_video) == 2  # both items scored
    assert result.aggregate.per_class["fire"].tp == 1  # good clip still scored
    assert result.aggregate.n_alerts_total == 1  # bad clip contributed no alerts
