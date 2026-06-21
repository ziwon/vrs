"""Live vLLM verifier validation.

This command is intentionally separate from unit tests: it requires a CUDA host,
the optional ``vllm`` extra, and model weights. It validates the backend through
the same ``AlertVerifier`` path used in production and records enough runtime
metadata to pin the validated environment in benchmark notes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from vrs.pipeline import load_config

try:
    from scripts.smoke_verifier_backend import run_smoke
except ModuleNotFoundError:
    from smoke_verifier_backend import run_smoke


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm": _version_or_none("vllm"),
        "torch": _version_or_none("torch"),
    }
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["torch_cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            snapshot["gpu_name"] = torch.cuda.get_device_name(idx)
            snapshot["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024),
                2,
            )
    except Exception as exc:
        snapshot["torch_error"] = str(exc)
    return snapshot


def validate_environment(*, require_cuda: bool = True) -> list[str]:
    errors: list[str] = []
    if _version_or_none("vllm") is None:
        errors.append("vllm is not installed; run `uv sync --extra vllm` on the validation host")
    if require_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("CUDA is not available to torch; vLLM validation needs a GPU host")
        except Exception as exc:
            errors.append(f"failed to inspect torch CUDA availability: {exc}")
    return errors


def validate_vllm_backend(
    *,
    config_path: Path,
    policy_path: Path,
    class_name: str | None,
    image_path: Path | None,
    out_path: Path,
    require_cuda: bool = True,
    require_json_valid: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    backend = str((config.get("verifier") or {}).get("backend", ""))
    if backend != "vllm":
        raise ValueError(f"{config_path} must set verifier.backend: vllm, got {backend!r}")

    env_errors = validate_environment(require_cuda=require_cuda)
    env = environment_snapshot()
    if env_errors:
        payload = {"passed": False, "environment": env, "errors": env_errors}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    smoke_t0 = time.perf_counter()
    smoke = run_smoke(
        config_path=config_path,
        policy_path=policy_path,
        class_name=class_name,
        image_path=image_path,
        out_path=None,
    )
    elapsed_s = time.perf_counter() - smoke_t0
    errors = []
    if require_json_valid and smoke.get("verifier_json_valid") is not True:
        errors.append("verifier output did not parse as valid JSON")

    payload = {
        "passed": not errors,
        "environment": env,
        "elapsed_s": round(elapsed_s, 4),
        "errors": errors,
        "smoke": smoke,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live-validate the vLLM verifier backend")
    parser.add_argument("--config", type=Path, default=Path("configs/vllm-cosmos.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("configs/policies/safety.yaml"))
    parser.add_argument("--class-name", default="fire")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/vllm-smoke/result.json"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-unparsed-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = validate_vllm_backend(
        config_path=args.config,
        policy_path=args.policy,
        class_name=args.class_name,
        image_path=args.image,
        out_path=args.out,
        require_cuda=not args.allow_cpu,
        require_json_valid=not args.allow_unparsed_json,
    )
    print(f"Result: {args.out}")
    if payload["passed"]:
        stats = payload.get("smoke", {}).get("smoke", {}).get("generation_stats", {})
        print(
            "vLLM validation passed: "
            f"json_valid={payload.get('smoke', {}).get('verifier_json_valid')} "
            f"tokens_per_second={stats.get('tokens_per_second', 'n/a')}"
        )
        return

    for error in payload.get("errors", []):
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)


def _version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
