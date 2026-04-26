"""vLLM-backed Cosmos verifier runtime.

Why vLLM: paged KV cache + in-flight batching give 2-3x higher generation
throughput than ``transformers.generate`` on the same hardware, and the
server-side continuous-batching model means queued verify requests amortize
nicely. Qwen3-VL architecture (which Cosmos-Reason2-2B inherits) is
supported natively from vLLM ~0.6.5 onward.

**Status note.** The transformers backend in ``cosmos_loader.py`` is
exhaustively exercised by the repo's test suite. The vLLM backend is
structurally complete and conforms to the ``CosmosBackend`` Protocol
(pinned by unit tests that substitute a fake ``vllm`` module), but the
actual GPU round-trip against a live vLLM instance has not been validated
in this repo's CI — that validation requires a CUDA host with vLLM
installed and a Cosmos-Reason2-2B checkpoint mounted. The code below
targets the vLLM ≥ 0.6.5 public API; when you flip a deployment to this
backend, do a short smoke run and pin the exact version you validated
against.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _bgr_to_pil(frames_bgr: list[np.ndarray]):
    """Shared with cosmos_loader — kept local so this module doesn't depend
    on the transformers backend being importable."""
    import cv2
    from PIL import Image

    out = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(rgb))
    return out


class VLLMCosmosBackend:
    """``CosmosBackend`` backed by a vLLM ``LLM`` engine.

    Construction imports ``vllm`` lazily and raises a clear ImportError
    pointing at the install extra if it's missing. Per-call, ``chat_video``
    builds a Qwen3-VL-style chat message list, attaches
    ``GuidedDecodingParams(json=schema)`` when a ``response_schema`` is
    provided, and returns the decoded text.
    """

    def __init__(self, cfg):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM backend requires the 'vllm' package. Install with "
                "`uv sync --extra vllm`. "
                f"(underlying: {e})"
            ) from e

        self.cfg = cfg
        # Map our dtype names to vLLM's. vLLM does its own W4A16 loading
        # via the model's quantization_config, so we pass-through the
        # model_id and let vLLM pick up the pre-quantized weights.
        dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "w4a16": "auto"}
        llm_kwargs = {
            "model": cfg.model_id,
            "dtype": dtype_map.get(cfg.dtype, "auto"),
            "trust_remote_code": True,
            # Sensible defaults for a 2B VLM on a single GPU; operators can
            # tune via env vars if needed.
            "max_model_len": 8192,
            "enforce_eager": False,
        }
        self.llm = LLM(**llm_kwargs)

    def chat_video(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_bgr: list[np.ndarray],
        *,
        clip_fps: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        if not frames_bgr:
            raise ValueError("chat_video requires at least one frame")

        from vllm import SamplingParams

        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:  # very old vllm — graceful downgrade
            GuidedDecodingParams = None  # type: ignore[assignment]

        pil_frames = _bgr_to_pil(frames_bgr)

        # Qwen3-VL accepts a list of images in message content; vLLM's chat
        # API (``llm.chat``) resolves them through the model's processor.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_frames],
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        sp_kwargs: dict[str, Any] = dict(
            max_tokens=int(self.cfg.max_new_tokens),
            temperature=max(float(self.cfg.temperature), 1e-5),
        )
        if response_schema is not None and GuidedDecodingParams is not None:
            sp_kwargs["guided_decoding"] = GuidedDecodingParams(json=response_schema)
        elif response_schema is not None:
            logger.warning(
                "vLLM version lacks GuidedDecodingParams; verifier will run "
                "unconstrained. Upgrade to vLLM >= 0.6.5 for guided JSON."
            )

        sampling = SamplingParams(**sp_kwargs)

        # llm.chat auto-selects the model's chat template and accepts image
        # content items via the model's multi-modal processor.
        outputs = self.llm.chat(messages, sampling_params=sampling)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text
