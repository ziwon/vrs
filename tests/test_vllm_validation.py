from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_vllm_backend


def _config(path: Path, *, backend: str = "vllm") -> Path:
    path.write_text(
        "\n".join(
            [
                "ingest:",
                "  target_fps: 4",
                "detector:",
                "  model: yoloe-11l-seg.pt",
                "event_state:",
                "  window: 8",
                "verifier:",
                "  enabled: true",
                f"  backend: {backend}",
                "  model_id: nvidia/Cosmos-Reason2-2B",
                "sink: {}",
            ]
        )
    )
    return path


def _policy(path: Path) -> Path:
    path.write_text(
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
    return path


def test_validate_vllm_backend_rejects_non_vllm_config(tmp_path: Path):
    cfg = _config(tmp_path / "config.yaml", backend="transformers")

    with pytest.raises(ValueError, match=r"verifier\.backend: vllm"):
        validate_vllm_backend.validate_vllm_backend(
            config_path=cfg,
            policy_path=_policy(tmp_path / "policy.yaml"),
            class_name="fire",
            image_path=None,
            out_path=tmp_path / "result.json",
        )


def test_validate_vllm_backend_writes_environment_errors(monkeypatch, tmp_path: Path):
    cfg = _config(tmp_path / "config.yaml")
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        validate_vllm_backend,
        "validate_environment",
        lambda require_cuda=True: ["vllm is not installed"],
    )
    monkeypatch.setattr(validate_vllm_backend, "environment_snapshot", lambda: {"vllm": None})

    payload = validate_vllm_backend.validate_vllm_backend(
        config_path=cfg,
        policy_path=_policy(tmp_path / "policy.yaml"),
        class_name="fire",
        image_path=None,
        out_path=out,
    )

    assert payload["passed"] is False
    assert payload["errors"] == ["vllm is not installed"]
    assert json.loads(out.read_text()) == payload


def test_validate_vllm_backend_accepts_mocked_live_smoke(monkeypatch, tmp_path: Path):
    cfg = _config(tmp_path / "config.yaml")
    out = tmp_path / "result.json"
    monkeypatch.setattr(validate_vllm_backend, "validate_environment", lambda require_cuda=True: [])
    monkeypatch.setattr(
        validate_vllm_backend,
        "environment_snapshot",
        lambda: {"vllm": "test", "cuda_available": True},
    )
    monkeypatch.setattr(
        validate_vllm_backend,
        "run_smoke",
        lambda **kwargs: {
            "true_alert": True,
            "confidence": 0.9,
            "verifier_json_valid": True,
            "smoke": {"generation_stats": {"tokens_per_second": 10.0}},
        },
    )

    payload = validate_vllm_backend.validate_vllm_backend(
        config_path=cfg,
        policy_path=_policy(tmp_path / "policy.yaml"),
        class_name="fire",
        image_path=None,
        out_path=out,
    )

    assert payload["passed"] is True
    assert payload["errors"] == []
    assert payload["smoke"]["smoke"]["generation_stats"]["tokens_per_second"] == 10.0
