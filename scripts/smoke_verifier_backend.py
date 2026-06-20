"""Smoke-test one VLM verifier backend through the AlertVerifier contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vrs.pipeline import load_config
from vrs.policy import load_watch_policy
from vrs.runtime import VLMConfig, build_vlm_backend
from vrs.schemas import CandidateAlert, Detection
from vrs.verifier import AlertVerifier


def load_frame(path: Path | None, *, width: int = 640, height: int = 360) -> np.ndarray:
    if path is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[height // 3 : (height * 2) // 3, width // 3 : (width * 2) // 3] = (0, 80, 255)
        return frame

    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return frame


def make_candidate(class_name: str, severity: str, frame: np.ndarray) -> CandidateAlert:
    height, width = frame.shape[:2]
    return CandidateAlert(
        class_name=class_name,
        severity=severity,
        start_pts_s=0.0,
        peak_pts_s=1.0,
        peak_frame_index=4,
        peak_detections=[
            Detection(
                class_name=class_name,
                score=0.99,
                xyxy=(width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                raw_label=class_name,
            )
        ],
        keyframes=[frame],
        keyframe_pts=[1.0],
    )


def build_verifier(config: dict[str, Any], policy_path: Path) -> AlertVerifier:
    policy = load_watch_policy(policy_path)
    ver_cfg = config["verifier"]
    backend = build_vlm_backend(
        VLMConfig(
            model_id=ver_cfg["model_id"],
            dtype=ver_cfg.get("dtype", "bf16"),
            device=ver_cfg.get("device", "cuda"),
            max_new_tokens=int(ver_cfg.get("max_new_tokens", 1024)),
            temperature=float(ver_cfg.get("temperature", 0.2)),
            clip_fps=int(ver_cfg.get("clip_fps", 4)),
            base_url=ver_cfg.get("base_url"),
            api_key_env=ver_cfg.get("api_key_env"),
            timeout_s=float(ver_cfg.get("timeout_s", 60.0)),
            max_frame_width=(
                int(ver_cfg["max_frame_width"])
                if ver_cfg.get("max_frame_width") is not None
                else None
            ),
        ),
        backend=ver_cfg.get("backend", "transformers"),
    )
    return AlertVerifier(
        vlm=backend,
        policy=policy,
        request_bbox=bool(ver_cfg.get("request_bbox", True)),
        request_trajectory=bool(ver_cfg.get("request_trajectory", True)),
        clip_fps=int(ver_cfg.get("clip_fps", 4)),
        failure_policy=ver_cfg.get("failure_policy"),
    )


def run_smoke(
    *,
    config_path: Path,
    policy_path: Path,
    class_name: str | None,
    image_path: Path | None,
    out_path: Path | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    policy = load_watch_policy(policy_path)
    selected_class = class_name or policy.names()[0]
    if selected_class not in policy:
        raise ValueError(f"class {selected_class!r} is not in {policy_path}")

    frame = load_frame(image_path)
    candidate = make_candidate(selected_class, policy[selected_class].severity, frame)
    verifier = build_verifier(config, policy_path)
    result = verifier.verify(candidate)
    payload = result.to_json()
    payload["smoke"] = {
        "config_path": str(config_path),
        "policy_path": str(policy_path),
        "backend": str((config.get("verifier") or {}).get("backend", "transformers")),
        "model_id": str((config.get("verifier") or {}).get("model_id", "")),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a configured verifier backend")
    parser.add_argument("--config", default="configs/qwen-openai-compatible.yaml", type=Path)
    parser.add_argument("--policy", default="configs/policies/safety.yaml", type=Path)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/verifier-smoke/result.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = run_smoke(
        config_path=args.config,
        policy_path=args.policy,
        class_name=args.class_name,
        image_path=args.image,
        out_path=args.out,
    )
    print(
        "verifier smoke: "
        f"true_alert={payload['true_alert']} "
        f"confidence={payload['confidence']:.2f} "
        f"json_valid={payload.get('verifier_json_valid')}"
    )
    print(f"Result: {args.out}")


if __name__ == "__main__":
    main()
