"""Tests for the Cosmos verifier backend abstraction.

Structural tests only — nothing GPU-bound. The vLLM backend is exercised
via a fake ``vllm`` module substituted into ``sys.modules`` so we can
assert Protocol conformance and wiring without installing vLLM or
spinning up a GPU."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vrs.runtime import CosmosBackend, build_cosmos_backend
from vrs.runtime.backends import _KNOWN_BACKENDS


def _fake_vllm_module(captured: dict):
    """Return a ``vllm`` module stub that records calls for assertions."""
    fake = types.ModuleType("vllm")
    fake_sp = types.ModuleType("vllm.sampling_params")

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def chat(self, messages, sampling_params):
            captured["messages"] = messages
            captured["sampling_params"] = sampling_params

            class _Out:
                def __init__(self, text):
                    self.text = text

            class _Result:
                def __init__(self, text):
                    self.outputs = [_Out(text)]

            return [
                _Result(
                    '{"true_alert": true, "confidence": 0.9, '
                    '"false_negative_class": null, "rationale": "ok"}'
                )
            ]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sp_kwargs"] = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _FakeGuidedDecodingParams:
        def __init__(self, json):
            captured["guided_json_schema"] = json
            self.json = json

    fake.LLM = _FakeLLM
    fake.SamplingParams = _FakeSamplingParams
    fake_sp.GuidedDecodingParams = _FakeGuidedDecodingParams
    fake.sampling_params = fake_sp
    return fake, fake_sp


# ─── factory ──────────────────────────────────────────────────────────


def test_build_cosmos_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown verifier backend"):
        build_cosmos_backend(object(), backend="banana")


def test_build_cosmos_backend_trtllm_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="trtllm"):
        build_cosmos_backend(object(), backend="trtllm")


def test_known_backends_set_matches_factory_branches():
    """If we add a backend name we must also update the _KNOWN_BACKENDS
    advertisement — this pin catches silent drift."""
    assert {"transformers", "vllm", "trtllm"} == _KNOWN_BACKENDS


# ─── vLLM backend ─────────────────────────────────────────────────────


def test_vllm_backend_raises_clean_importerror_when_vllm_missing(monkeypatch):
    # simulate missing dep
    monkeypatch.setitem(sys.modules, "vllm", None)
    from vrs.runtime.vllm_cosmos import VLLMCosmosBackend

    with pytest.raises(ImportError, match="vLLM backend requires"):
        VLLMCosmosBackend(cfg=object())


def test_vllm_backend_conforms_to_protocol(monkeypatch):
    captured: dict = {}
    fake, fake_sp = _fake_vllm_module(captured)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sp)
    monkeypatch.setitem(sys.modules, "PIL", types.ModuleType("PIL"))
    # PIL.Image is needed by _bgr_to_pil
    pil_image = types.ModuleType("PIL.Image")
    pil_image.fromarray = lambda arr: ("pil_stub", arr.shape)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)

    from vrs.runtime.cosmos_loader import CosmosConfig
    from vrs.runtime.vllm_cosmos import VLLMCosmosBackend

    cfg = CosmosConfig(
        model_id="nvidia/Cosmos-Reason2-2B",
        dtype="bf16",
        max_new_tokens=128,
        temperature=0.2,
        clip_fps=4,
    )
    backend = VLLMCosmosBackend(cfg)

    # Protocol conformance — isinstance works because CosmosBackend is
    # @runtime_checkable.
    assert isinstance(backend, CosmosBackend)

    # LLM got the right keyword args
    kw = captured["llm_kwargs"]
    assert kw["model"] == "nvidia/Cosmos-Reason2-2B"
    assert kw["dtype"] == "bfloat16"
    assert kw["trust_remote_code"] is True


def test_vllm_backend_passes_schema_as_guided_decoding(monkeypatch):
    captured: dict = {}
    fake, fake_sp = _fake_vllm_module(captured)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sp)
    # stub cv2 + PIL so _bgr_to_pil runs without the real libraries. The
    # ``from PIL import Image`` form imports the ``PIL`` package and reads
    # ``Image`` off it, so we have to attach Image to the fake PIL module.
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 4
    fake_cv2.cvtColor = lambda arr, code: arr
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    pil_image = types.ModuleType("PIL.Image")
    pil_image.fromarray = lambda arr: object()
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = pil_image
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)

    from vrs.runtime.cosmos_loader import CosmosConfig
    from vrs.runtime.vllm_cosmos import VLLMCosmosBackend

    backend = VLLMCosmosBackend(
        CosmosConfig(
            model_id="nvidia/Cosmos-Reason2-2B",
            dtype="bf16",
            max_new_tokens=64,
            temperature=0.1,
        )
    )
    schema = {"type": "object", "properties": {"true_alert": {"type": "boolean"}}}
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    out = backend.chat_video("sys", "user", frames, response_schema=schema)

    # guided decoding was constructed from the schema and passed into SP
    assert captured["guided_json_schema"] == schema
    assert captured["sp_kwargs"]["max_tokens"] == 64
    assert captured["sp_kwargs"]["temperature"] == pytest.approx(0.1)
    assert captured["sp_kwargs"]["guided_decoding"].json == schema
    # messages include one image item per frame + the text prompt
    user_content = captured["messages"][1]["content"]
    image_items = [c for c in user_content if c.get("type") == "image"]
    text_items = [c for c in user_content if c.get("type") == "text"]
    assert len(image_items) == 1
    assert text_items[0]["text"] == "user"
    assert out.startswith("{")


def test_vllm_backend_empty_frames_raises():
    fake, fake_sp = _fake_vllm_module({})
    sys.modules["vllm"] = fake
    sys.modules["vllm.sampling_params"] = fake_sp
    try:
        from vrs.runtime.cosmos_loader import CosmosConfig
        from vrs.runtime.vllm_cosmos import VLLMCosmosBackend

        backend = VLLMCosmosBackend(
            CosmosConfig(
                model_id="x",
                dtype="bf16",
                max_new_tokens=1,
                temperature=0.0,
            )
        )
        with pytest.raises(ValueError, match="at least one frame"):
            backend.chat_video("sys", "user", [], response_schema=None)
    finally:
        sys.modules.pop("vllm", None)
        sys.modules.pop("vllm.sampling_params", None)


# ─── verifier integrates cleanly through the Protocol ─────────────────


def test_alert_verifier_only_depends_on_chat_video_surface():
    """Protocol contract: AlertVerifier must accept anything with a
    ``chat_video`` method — no transformers-specific attributes snuck in
    when we removed ``_fresh_logits_processors``."""
    from vrs.policy.watch_policy import WatchItem, WatchPolicy
    from vrs.schemas import CandidateAlert, Detection
    from vrs.verifier.alert_verifier import AlertVerifier

    class _StubBackend:
        def __init__(self):
            self.last_schema = None

        def chat_video(self, system, user, frames, *, clip_fps=None, response_schema=None):
            self.last_schema = response_schema
            return (
                '{"true_alert": true, "confidence": 0.8, '
                '"false_negative_class": null, "rationale": "ok"}'
            )

    policy = WatchPolicy(
        [
            WatchItem(
                name="fire",
                detector_prompts=["fire"],
                verifier_prompt="flames",
                severity="critical",
                min_score=0.3,
                min_persist_frames=2,
            ),
        ]
    )
    stub = _StubBackend()
    verifier = AlertVerifier(cosmos=stub, policy=policy)

    fake_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=1.0,
        peak_pts_s=2.0,
        peak_frame_index=8,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(0, 0, 1, 1))],
        keyframes=[fake_frame],
        keyframe_pts=[2.0],
    )
    result = verifier.verify(cand)

    # Verifier forwarded the schema the AlertVerifier built at init
    assert stub.last_schema is not None
    assert "true_alert" in stub.last_schema["properties"]
    assert result.true_alert is True
    assert result.confidence == pytest.approx(0.8)
