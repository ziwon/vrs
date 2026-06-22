"""vLLM-backed VLM verifier runtime for the Cosmos baseline.

Why vLLM: paged KV cache + in-flight batching give 2-3x higher generation
throughput than ``transformers.generate`` on the same hardware, and the
server-side continuous-batching model means queued verify requests amortize
nicely. Qwen3-VL architecture (which Cosmos-Reason2-2B inherits) is
supported natively from vLLM ~0.6.5 onward.

**Status note.** The transformers backend in ``cosmos_loader.py`` is
exhaustively exercised by the repo's test suite. The vLLM backend conforms to
the ``VLMBackend`` Protocol (pinned by unit tests that substitute a fake
``vllm`` module) and has a live RTX 5080 smoke/eval note under
``docs/benchmarks/``. That is still not a deployment-sizing claim: run the
smoke and throughput benchmark on the target CUDA host before flipping a
production profile to this backend.
"""

from __future__ import annotations

import base64
import gc
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _frame_to_jpeg_data_url(frame_bgr: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise ValueError("failed to JPEG-encode verifier frame")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


class VLLMCosmosBackend:
    """Cosmos baseline implementation of ``VLMBackend`` using vLLM ``LLM``.

    Construction imports ``vllm`` lazily and raises a clear ImportError
    pointing at the install extra if it's missing. Per-call, ``chat_video``
    builds a Qwen3-VL-style chat message list, attaches the response schema
    through vLLM's structured-output surface, and returns the decoded text.
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
        self.last_generation_stats: dict[str, float | int] = {}
        # Map our dtype names to vLLM's. vLLM does its own W4A16 loading
        # via the model's quantization_config, so we pass-through the
        # model_id and let vLLM pick up the pre-quantized weights.
        dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "w4a16": "auto"}
        llm_kwargs = {
            "model": cfg.model_id,
            "dtype": dtype_map.get(cfg.dtype, "auto"),
            "trust_remote_code": True,
            "max_model_len": int(cfg.max_model_len or 8192),
            "enforce_eager": False,
        }
        if cfg.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(cfg.gpu_memory_utilization)
        self.llm = LLM(**llm_kwargs)
        self._closed = False

    def close(self) -> None:
        """Release vLLM worker processes and cached CUDA memory."""
        if self._closed:
            return
        self._closed = True
        try:
            engine = getattr(self.llm, "llm_engine", None)
            engine_core = getattr(engine, "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown(timeout=5)
        except Exception as e:
            logger.warning("vLLM shutdown failed: %s", e)
        finally:
            self.llm = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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

        # vLLM's OpenAI chat parser accepts multimodal media as ``image_url``
        # parts. Plain ``image`` parts are rejected by current vLLM releases.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _frame_to_jpeg_data_url(frame),
                                "detail": "auto",
                            },
                        }
                        for frame in frames_bgr
                    ],
                ],
            },
        ]

        sp_kwargs: dict[str, Any] = dict(
            max_tokens=int(self.cfg.max_new_tokens),
            temperature=max(float(self.cfg.temperature), 1e-5),
        )
        if response_schema is not None:
            structured_output = _build_structured_output_params(response_schema)
            if structured_output is None:
                logger.warning(
                    "vLLM version lacks a supported structured-output API; "
                    "verifier will run unconstrained."
                )
            else:
                sp_kwargs.update(structured_output)

        sampling = SamplingParams(**sp_kwargs)

        # llm.chat auto-selects the model's chat template and accepts image
        # content items via the model's multi-modal processor.
        generate_t0 = time.perf_counter()
        if self.llm is None:
            raise RuntimeError("vLLM backend is closed")
        outputs = self.llm.chat(messages, sampling_params=sampling)
        elapsed_s = time.perf_counter() - generate_t0
        if not outputs or not outputs[0].outputs:
            self.last_generation_stats = {"elapsed_s": elapsed_s}
            return ""
        text = outputs[0].outputs[0].text
        token_ids = getattr(outputs[0].outputs[0], "token_ids", None)
        completion_tokens = len(token_ids) if token_ids is not None else None
        self.last_generation_stats = {
            "elapsed_s": elapsed_s,
            **({"completion_tokens": completion_tokens} if completion_tokens is not None else {}),
            **(
                {"tokens_per_second": completion_tokens / elapsed_s}
                if completion_tokens is not None and completion_tokens > 0 and elapsed_s > 0
                else {}
            ),
        }
        return text


def _build_structured_output_params(response_schema: dict[str, Any]) -> dict[str, Any] | None:
    """Map a JSON schema to the structured-output API exposed by this vLLM."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=response_schema)}
    except ImportError:
        pass

    try:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=response_schema)}
    except ImportError:
        return None
