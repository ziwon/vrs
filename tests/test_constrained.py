"""Unit tests for constrained-decoding glue.

We can't exercise the full xgrammar + transformers roundtrip without a GPU
and model weights, so these tests cover the parts that *can* be verified
without the model:

  * schema construction matches the parser's expected field set,
  * ``false_negative_class`` is restricted to the policy's classes + null,
  * ``build_logits_processor`` degrades gracefully when xgrammar is absent
    or setup raises.
"""

from __future__ import annotations

import logging
import sys
import types

from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.verifier import constrained as C


def _policy() -> WatchPolicy:
    return WatchPolicy(
        [
            WatchItem(
                name="fire",
                detector_prompts=["fire"],
                verifier_prompt="flames",
                severity="critical",
                min_score=0.3,
                min_persist_frames=2,
            ),
            WatchItem(
                name="smoke",
                detector_prompts=["smoke"],
                verifier_prompt="smoke",
                severity="high",
                min_score=0.3,
                min_persist_frames=2,
            ),
        ]
    )


# ─── schema ───────────────────────────────────────────────────────────


def test_schema_has_expected_top_level_shape():
    schema = C.build_verifier_schema(["fire", "smoke"])
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    req = set(schema["required"])
    # All core verdict fields required regardless of bbox/trajectory flags
    assert {"true_alert", "confidence", "false_negative_class", "rationale"} <= req


def test_schema_fn_class_enum_matches_policy():
    schema = C.build_verifier_schema(["fire", "smoke", "weapon"])
    fn = schema["properties"]["false_negative_class"]
    assert None in fn["enum"]
    assert set(fn["enum"]) == {None, "fire", "smoke", "weapon"}


def test_schema_confidence_range_is_zero_to_one():
    schema = C.build_verifier_schema(["fire"])
    conf = schema["properties"]["confidence"]
    assert conf["type"] == "number"
    assert conf["minimum"] == 0.0
    assert conf["maximum"] == 1.0


def test_schema_bbox_and_trajectory_follow_request_flags():
    full = C.build_verifier_schema(["fire"], request_bbox=True, request_trajectory=True)
    assert "bbox_xywh_norm" in full["properties"]
    assert "trajectory_xy_norm" in full["properties"]
    assert "bbox_xywh_norm" in full["required"]
    assert "trajectory_xy_norm" in full["required"]

    minimal = C.build_verifier_schema(["fire"], request_bbox=False, request_trajectory=False)
    assert "bbox_xywh_norm" not in minimal["properties"]
    assert "trajectory_xy_norm" not in minimal["properties"]


def test_schema_bbox_is_nullable_array_of_four():
    schema = C.build_verifier_schema(["fire"], request_bbox=True)
    bbox = schema["properties"]["bbox_xywh_norm"]
    variants = bbox["oneOf"]
    types_ = [v.get("type") for v in variants]
    assert "null" in types_
    arr = next(v for v in variants if v.get("type") == "array")
    assert arr["minItems"] == 4
    assert arr["maxItems"] == 4


def test_schema_trajectory_is_array_of_xy_pairs():
    schema = C.build_verifier_schema(["fire"], request_trajectory=True)
    traj = schema["properties"]["trajectory_xy_norm"]
    assert traj["type"] == "array"
    item = traj["items"]
    assert item["minItems"] == 2
    assert item["maxItems"] == 2


# ─── logits-processor fallback ────────────────────────────────────────


def test_build_logits_processor_returns_none_when_xgrammar_missing(monkeypatch, caplog):
    """Simulate a host without xgrammar installed — must return None with
    exactly one INFO log per process."""
    # Mask any real xgrammar import so the function hits the ImportError path.
    monkeypatch.setitem(sys.modules, "xgrammar", None)
    monkeypatch.setattr(C, "_XGRAMMAR_UNAVAILABLE_LOGGED", False, raising=False)

    with caplog.at_level(logging.INFO, logger="vrs.verifier.constrained"):
        result = C.build_logits_processor({"type": "object"}, tokenizer=object())

    assert result is None
    info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("xgrammar not installed" in r.getMessage() for r in info_logs)


def test_build_logits_processor_only_logs_missing_xgrammar_once(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "xgrammar", None)
    monkeypatch.setattr(C, "_XGRAMMAR_UNAVAILABLE_LOGGED", False, raising=False)

    with caplog.at_level(logging.INFO, logger="vrs.verifier.constrained"):
        C.build_logits_processor({"type": "object"}, tokenizer=object())
        C.build_logits_processor({"type": "object"}, tokenizer=object())
        C.build_logits_processor({"type": "object"}, tokenizer=object())

    missing_logs = [r for r in caplog.records if "xgrammar not installed" in r.getMessage()]
    assert len(missing_logs) == 1


def test_build_logits_processor_falls_back_on_setup_error(monkeypatch, caplog):
    """If xgrammar is present but raises during setup, we fall back to None
    and log a WARNING naming the error rather than crashing the verifier."""
    fake = types.ModuleType("xgrammar")

    class _BrokenTokenizerInfo:
        @staticmethod
        def from_huggingface(*a, **kw):
            raise RuntimeError("tokenizer has no vocab")

    fake.TokenizerInfo = _BrokenTokenizerInfo
    monkeypatch.setitem(sys.modules, "xgrammar", fake)

    with caplog.at_level(logging.WARNING, logger="vrs.verifier.constrained"):
        result = C.build_logits_processor({"type": "object"}, tokenizer=object())

    assert result is None
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tokenizer has no vocab" in r.getMessage() for r in warns)


def test_build_logits_processor_returns_processor_when_xgrammar_succeeds(monkeypatch):
    """Happy path: the function returns whatever xgrammar's HF contrib yields,
    unchanged."""
    fake = types.ModuleType("xgrammar")
    fake_contrib = types.ModuleType("xgrammar.contrib")
    fake_hf = types.ModuleType("xgrammar.contrib.hf")

    class _FakeTokenizerInfo:
        @staticmethod
        def from_huggingface(*a, **kw):
            return "tok-info"

    class _FakeCompiler:
        def __init__(self, tok_info):
            self.tok_info = tok_info

        def compile_json_schema(self, schema):
            return ("compiled", schema)

    class _FakeLogitsProcessor:
        def __init__(self, compiled):
            self.compiled = compiled

    fake.TokenizerInfo = _FakeTokenizerInfo
    fake.GrammarCompiler = _FakeCompiler
    fake_hf.LogitsProcessor = _FakeLogitsProcessor
    fake.contrib = fake_contrib
    fake_contrib.hf = fake_hf
    monkeypatch.setitem(sys.modules, "xgrammar", fake)
    monkeypatch.setitem(sys.modules, "xgrammar.contrib", fake_contrib)
    monkeypatch.setitem(sys.modules, "xgrammar.contrib.hf", fake_hf)

    schema = C.build_verifier_schema(["fire"])
    result = C.build_logits_processor(schema, tokenizer=object(), vocab_size=32000)

    assert isinstance(result, _FakeLogitsProcessor)
    assert result.compiled == ("compiled", schema)


# ─── end-to-end wiring: verifier uses schema built from policy ─────────


def test_verifier_builds_schema_from_policy_on_init():
    """AlertVerifier must compute its response schema from the policy's
    class names so the fn_enum matches what the pipeline can dispatch."""
    from vrs.verifier.alert_verifier import AlertVerifier

    class _FakeCosmos:  # minimal shape AlertVerifier reads at init
        class processor:
            tokenizer = None

        class model:
            class config:
                vocab_size = 0

    verifier = AlertVerifier(
        cosmos=_FakeCosmos(),
        policy=_policy(),
        request_bbox=True,
        request_trajectory=True,
    )
    fn_enum = verifier._response_schema["properties"]["false_negative_class"]["enum"]
    assert set(fn_enum) == {None, "fire", "smoke"}
