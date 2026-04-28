"""VLM verifier backend Protocol + factory.

The slow path is the system's capacity ceiling, so we want to be able to
swap the generation engine without touching ``AlertVerifier``. Every
backend implements ``chat_video``; the verifier only ever calls that, and
the backend decides internally how to apply a JSON-schema constraint.

Backends shipped:

* ``transformers`` — the ``CosmosReason2`` class in ``cosmos_loader``.
  Default baseline implementation.
* ``vllm`` — higher generation throughput via paged KV cache + in-flight
  batching. Optional dep (``uv sync --extra vllm``); implementation
  lives in ``vllm_cosmos``.
* ``openai_compatible`` — served VLM over an OpenAI-compatible
  ``/chat/completions`` endpoint.
* ``trtllm`` — reserved (a future addition that produces the biggest
  latency win when paired with speculative decoding).

The Protocol is deliberately minimal. ``response_schema`` is passed
through ``chat_video`` so each backend can map it to its native
constraint surface (xgrammar logits processor for transformers,
``GuidedDecodingParams`` for vLLM, constraint engines in TRT-LLM).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VLMBackend(Protocol):
    """Minimum surface the verifier needs from a video-language-model runtime."""

    def chat_video(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_bgr: list[np.ndarray],
        *,
        clip_fps: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """One multi-modal turn over a short video clip. Returns completion text.

        If ``response_schema`` is provided, the backend must constrain the
        output to conform to it (no-op if the backend doesn't support
        constrained decoding — in which case the caller's parser fallback
        takes over).
        """
        ...


# Backward-compatible public alias. Existing imports and isinstance checks keep
# working while new code uses the model-family-neutral VLMBackend name.
CosmosBackend = VLMBackend


# ──────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────

_KNOWN_BACKENDS = {"transformers", "vllm", "openai_compatible", "trtllm"}


def build_vlm_backend(cfg, backend: str = "transformers") -> VLMBackend:
    """Construct the VLM backend named by ``backend``.

    Lazy-imports the backend module so hosts without vLLM / TRT-LLM
    installed can still run the transformers default unchanged.
    """
    name = (backend or "transformers").lower()
    if name == "transformers":
        # Local import — avoids circular dep (cosmos_loader imports from here
        # for Protocol registration only at type-check time).
        from .cosmos_loader import CosmosReason2

        return CosmosReason2(cfg)
    if name == "vllm":
        from .vllm_cosmos import VLLMCosmosBackend

        return VLLMCosmosBackend(cfg)
    if name in ("openai_compatible", "openai-compatible", "openai"):
        from .openai_compatible_vlm import OpenAICompatibleVLMBackend

        return OpenAICompatibleVLMBackend(cfg)
    if name == "trtllm":
        raise NotImplementedError(
            "the trtllm backend is a planned follow-on once the vllm path "
            "is validated end-to-end; track the #10 roadmap item."
        )
    raise ValueError(f"unknown verifier backend: {backend!r}. Valid: {sorted(_KNOWN_BACKENDS)}")


def build_cosmos_backend(cfg, backend: str = "transformers") -> VLMBackend:
    """Compatibility wrapper for the old Cosmos-named factory."""
    return build_vlm_backend(cfg, backend=backend)
