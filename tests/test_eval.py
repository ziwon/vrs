"""Eval-harness unit tests — pure Python, no GPU / video deps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrs.eval import (
    ClassMetrics,
    EvalItem,
    GroundTruthEvent,
    RunScore,
    score_alerts_against_truth,
)
from vrs.eval.datasets import LabeledDirDataset
from vrs.eval.metrics import aggregate_scores


def _alert(class_name: str, peak_pts_s: float, *, true_alert: bool = True,
           fn_cls: str | None = None) -> dict:
    return {
        "class_name": class_name,
        "peak_pts_s": peak_pts_s,
        "true_alert": true_alert,
        "false_negative_class": fn_cls,
    }


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
    alerts = [_alert("fire", 3.0), _alert("fire", 20.0)]    # 2nd is FP
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
        _alert("fire", 3.5, true_alert=False),   # verifier flipped — shouldn't become FP
    ]
    events = [_event("fire", 2.0, 5.0)]
    score = score_alerts_against_truth(alerts, events, tolerance_s=0.0)
    fire = score.per_class["fire"]
    assert (fire.tp, fire.fp, fire.fn) == (1, 0, 0)


def test_score_flip_rate_and_fn_flag_rate():
    alerts = [
        _alert("fire", 1.0, true_alert=True),
        _alert("fire", 2.0, true_alert=False),
        _alert("smoke", 3.0, true_alert=True, fn_cls="fire"),
    ]
    score = score_alerts_against_truth(alerts, [], tolerance_s=0.0)
    assert score.n_alerts_total == 3
    assert score.n_alerts_true == 2
    assert score.n_fn_flagged == 1
    assert score.flip_rate == pytest.approx(1 / 3)
    assert score.fn_flag_rate == pytest.approx(1 / 3)


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


# ─── aggregation ──────────────────────────────────────────────────────

def test_aggregate_scores_sums_counts_and_recomputes_ratios():
    a = score_alerts_against_truth(
        [_alert("fire", 3.0)], [_event("fire", 2.0, 5.0)], tolerance_s=0.0,
    )
    b = score_alerts_against_truth(
        [_alert("fire", 99.0)], [_event("fire", 10.0, 12.0)], tolerance_s=0.0,
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
    (tmp_path / "a.json").write_text(json.dumps({
        "events": [{"class": "fire", "start_s": 1.0, "end_s": 4.0}]
    }))
    # b.mp4 has no sidecar → empty events list (quiet footage)

    items = list(LabeledDirDataset(tmp_path))
    assert [i.video_path.name for i in items] == ["a.mp4", "b.mp4"]
    assert items[0].events == [GroundTruthEvent("fire", 1.0, 4.0)]
    assert items[1].events == []


def test_labeled_dir_rejects_nonexistent_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LabeledDirDataset(tmp_path / "does-not-exist")


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
    (root / "v1.json").write_text(json.dumps({
        "events": [{"class": "fire", "start_s": 1.0, "end_s": 3.0}]
    }))
    (root / "v2.mp4").write_bytes(b"\0")
    (root / "v2.json").write_text(json.dumps({
        "events": [{"class": "smoke", "start_s": 5.0, "end_s": 8.0}]
    }))

    alerts_per_video = {
        "v1": [_alert("fire", 2.0, true_alert=True)],
        "v2": [_alert("fire", 6.0, true_alert=False, fn_cls="smoke")],
    }
    factory = lambda od: _StubPipeline(od, alerts_per_video)

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


def _report(per_class: dict[str, float], overall_f1: float = 0.0,
            flip_rate: float = 0.0, fn_flag_rate: float = 0.0) -> dict:
    """Minimal report.json-shaped fixture — f1-only per class, all other
    ClassMetrics fields zeroed out since the gate only reads f1."""
    return {
        "aggregate": {
            "per_class": {
                cls: {"tp": 0, "fp": 0, "fn": 0,
                      "precision": 0.0, "recall": 0.0, "f1": f1}
                for cls, f1 in per_class.items()
            },
            "overall": {"tp": 0, "fp": 0, "fn": 0,
                        "precision": 0.0, "recall": 0.0, "f1": overall_f1},
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
    current = _report({"fire": 0.785}, overall_f1=0.785)   # drop of 0.015 < 0.02
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
    current = _report({"fire": 0.80}, overall_f1=0.60)    # overall dropped 0.20
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


def test_gate_cli_exit_codes(tmp_path: Path):
    import json as _json
    from vrs.eval.ci import main as ci_main

    baseline = tmp_path / "baseline.json"
    current_pass = tmp_path / "current_pass.json"
    current_fail = tmp_path / "current_fail.json"

    baseline.write_text(_json.dumps(_report({"fire": 0.80}, overall_f1=0.80)))
    current_pass.write_text(_json.dumps(_report({"fire": 0.79}, overall_f1=0.79)))
    current_fail.write_text(_json.dumps(_report({"fire": 0.50}, overall_f1=0.50)))

    assert ci_main(["--baseline", str(baseline), "--current", str(current_pass)]) == 0
    assert ci_main(["--baseline", str(baseline), "--current", str(current_fail)]) == 1
    assert ci_main(["--baseline", str(baseline),
                    "--current", str(tmp_path / "nonexistent.json")]) == 2


def test_harness_survives_a_failing_pipeline_run(tmp_path: Path):
    """One bad clip must not abort the entire eval — it scores as no alerts
    and the harness keeps going."""
    from vrs.eval import evaluate

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "good.mp4").write_bytes(b"\0")
    (root / "good.json").write_text(json.dumps({
        "events": [{"class": "fire", "start_s": 1.0, "end_s": 3.0}]
    }))
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
    assert len(result.per_video) == 2                 # both items scored
    assert result.aggregate.per_class["fire"].tp == 1  # good clip still scored
    assert result.aggregate.n_alerts_total == 1        # bad clip contributed no alerts
