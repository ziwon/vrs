"""TensorRT-LLM verifier backend.

This backend uses TensorRT-LLM's high-level ``LLM`` API behind the same
``VLMBackend`` contract as the transformers, vLLM, and served OpenAI-compatible
backends. It is intentionally opt-in: TensorRT-LLM installs and engine choices
are host-specific, so production configs still need a target-GPU smoke and
benchmark before adoption.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import numpy as np


def _frame_to_jpeg_data_url(frame_bgr: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise ValueError("failed to JPEG-encode verifier frame")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


class TensorRTLLMVLMBackend:
    """VLM backend backed by TensorRT-LLM's Python LLM API."""

    def __init__(self, cfg):
        try:
            from tensorrt_llm import LLM
        except ImportError as e:
            raise ImportError(
                "TRT-LLM backend requires the 'tensorrt_llm' package and a CUDA "
                "runtime matching the target host. Install TensorRT-LLM in the "
                "validation environment before using verifier.backend=trtllm. "
                f"(underlying: {e})"
            ) from e

        self.cfg = cfg
        self.last_generation_stats: dict[str, float | int] = {}
        llm_kwargs = _llm_kwargs(cfg)
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

        from tensorrt_llm import SamplingParams

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

        sp_kwargs: dict[str, Any] = {
            "max_tokens": int(self.cfg.max_new_tokens),
            "temperature": max(float(self.cfg.temperature), 1e-5),
        }
        if response_schema is not None:
            sp_kwargs["guided_decoding"] = _guided_decoding_params(response_schema)

        sampling = SamplingParams(**sp_kwargs)

        generate_t0 = time.perf_counter()
        if hasattr(self.llm, "chat"):
            outputs = self.llm.chat(messages, sampling_params=sampling)
        else:
            tokenizer = getattr(self.llm, "tokenizer", None)
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise RuntimeError(
                    "TensorRT-LLM backend needs llm.chat(...) or "
                    "llm.tokenizer.apply_chat_template(...)"
                )
            prompt = apply_chat_template(messages, tokenize=False)
            outputs = self.llm.generate(prompt, sampling_params=sampling)
        elapsed_s = time.perf_counter() - generate_t0

        text, completion_tokens = _extract_output(outputs)
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


def _llm_kwargs(cfg) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": cfg.model_id,
        "guided_decoding_backend": cfg.guided_decoding_backend or "xgrammar",
    }
    if cfg.tokenizer_id:
        kwargs["tokenizer"] = cfg.tokenizer_id
    if cfg.max_model_len is not None:
        kwargs["max_model_len"] = int(cfg.max_model_len)
    if cfg.dtype:
        kwargs["dtype"] = cfg.dtype

    speculative = _speculative_config(cfg)
    if speculative:
        kwargs["speculative_config"] = speculative

    if cfg.trtllm_extra_llm_kwargs:
        kwargs.update(cfg.trtllm_extra_llm_kwargs)
    return kwargs


def _speculative_config(cfg) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if cfg.draft_model_id:
        config["draft_model"] = cfg.draft_model_id
    if cfg.draft_engine_dir:
        config["draft_engine_dir"] = cfg.draft_engine_dir
    if cfg.speculative_tokens is not None:
        config["num_draft_tokens"] = int(cfg.speculative_tokens)
    return config


def _guided_decoding_params(response_schema: dict[str, Any]) -> Any:
    from tensorrt_llm.llmapi import GuidedDecodingParams

    return GuidedDecodingParams(json=json.dumps(response_schema))


def _extract_output(outputs: Any) -> tuple[str, int | None]:
    first = outputs[0] if isinstance(outputs, list) else outputs
    choices = getattr(first, "outputs", None)
    if not choices:
        return "", None
    item = choices[0]
    text = str(getattr(item, "text", ""))
    token_ids = getattr(item, "token_ids", None)
    return text, len(token_ids) if token_ids is not None else None
