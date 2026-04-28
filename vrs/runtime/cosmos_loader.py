"""Cosmos-Reason2-2B loader.

Cosmos-Reason2-2B is post-trained from Qwen3-VL-2B-Instruct, so it loads
through ``transformers`` with the standard Qwen-VL chat template and consumes
multi-frame video natively (FPS=4 is the recommended training distribution).

Three weight profiles, sized for different deployment targets:

    bf16   : accuracy-oriented default; validate target GPU memory. NVIDIA's
             2026 reference model card lists 24 GB minimum for this path.
    fp16   : ~5.0 GB — same footprint, prefer bf16 on Ampere+.
    w4a16  : ~1.7 GB — community quant ``embedl/Cosmos-Reason2-2B-W4A16``,
                       fits 8 GB cards / Jetson Orin.

The public runtime config is named ``VLMConfig`` because Cosmos is one backend
implementation, not the verifier architecture. ``CosmosConfig`` remains as a
compatibility alias for existing configs and imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VLMConfig:
    model_id: str = "nvidia/Cosmos-Reason2-2B"
    dtype: str = "bf16"  # bf16 | fp16 | w4a16
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.2
    clip_fps: int = 4  # Cosmos-Reason2-2B was trained at FPS=4


CosmosConfig = VLMConfig


def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _bgr_list_to_pil_rgb(frames_bgr: list[np.ndarray]):
    """Convert a list of OpenCV BGR uint8 frames to PIL RGB images."""
    import cv2
    from PIL import Image

    out = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(rgb))
    return out


class CosmosReason2:
    """Thin wrapper that hides the dtype / quant / chat-template choreography."""

    def __init__(self, cfg: VLMConfig):
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _Model
        except ImportError:
            from transformers import AutoModelForVision2Seq as _Model  # type: ignore

        self.cfg = cfg

        kwargs = {"trust_remote_code": True}

        if cfg.dtype == "w4a16":
            # Pre-quantized checkpoint already carries its own quant_config; just
            # let transformers materialize it.
            kwargs["device_map"] = cfg.device
        else:
            kwargs["torch_dtype"] = _torch_dtype(cfg.dtype)
            kwargs["device_map"] = cfg.device

        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
        self.model = _Model.from_pretrained(cfg.model_id, **kwargs).eval()

    @torch.inference_mode()
    def chat_video(
        self,
        system_prompt: str,
        user_prompt: str,
        frames_bgr: list[np.ndarray],
        *,
        clip_fps: int | None = None,
        response_schema: dict | None = None,
    ) -> str:
        """One multi-modal turn over a short video clip; returns the completion text.

        ``response_schema`` (if given) is compiled to an XGrammar
        ``LogitsProcessor`` right before ``generate`` — a fresh processor
        per call because xgrammar matchers are stateful. ``None`` means
        unconstrained generation.
        """
        if not frames_bgr:
            raise ValueError("chat_video requires at least one frame")

        pil_frames = _bgr_list_to_pil_rgb(frames_bgr)
        fps = int(clip_fps if clip_fps is not None else self.cfg.clip_fps)

        # Qwen3-VL / Cosmos-Reason2 chat schema accepts a "video" content item
        # whose payload is a list of PIL frames + a sampled fps for timestamping.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": fps},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Some processors still want videos passed via the explicit kwarg
        try:
            inputs = self.processor(
                text=[text],
                videos=[pil_frames],
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            # older processors: fall back to images=[grid]
            inputs = self.processor(
                text=[text],
                images=pil_frames,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0.0,
            temperature=max(self.cfg.temperature, 1e-5),
        )
        if response_schema is not None:
            proc = self._build_logits_processor(response_schema)
            if proc is not None:
                gen_kwargs["logits_processor"] = [proc]
        gen = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        out_ids = gen[:, prompt_len:]
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    def _build_logits_processor(self, schema: dict):
        """Compile a JSON-schema constraint into a transformers LogitsProcessor.

        Delegates to ``vrs.verifier.constrained.build_logits_processor`` so the
        xgrammar setup lives in one place. Returns ``None`` when xgrammar
        isn't installed (caller relies on the parser fallback).
        """
        from ..verifier.constrained import build_logits_processor  # avoid cycle

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", None)
        return build_logits_processor(schema, tokenizer, vocab_size)
