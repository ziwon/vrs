"""OpenAI-compatible served VLM verifier runtime.

This backend keeps the verifier's local ``chat_video`` contract but sends the
request to a ``/chat/completions`` endpoint. Frames are encoded as JPEG data
URLs and attached as OpenAI-style ``image_url`` content items, which is the
common multimodal surface exposed by vLLM, SGLang, and other compatible
servers.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np


def _frame_to_jpeg_data_url(frame_bgr: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise ValueError("failed to JPEG-encode verifier frame")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


class OpenAICompatibleVLMBackend:
    """VLM backend for OpenAI-compatible chat-completions servers."""

    def __init__(self, cfg):
        if not cfg.base_url:
            raise ValueError(
                "openai_compatible verifier backend requires verifier.base_url "
                "(for example http://localhost:8000/v1)"
            )
        self.cfg = cfg
        self.base_url = str(cfg.base_url).rstrip("/") + "/"
        self.api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        self.timeout_s = float(getattr(cfg, "timeout_s", 60.0))

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

        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _frame_to_jpeg_data_url(frame), "detail": "auto"},
            }
            for frame in frames_bgr
        )

        payload: dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": int(self.cfg.max_new_tokens),
            "temperature": float(self.cfg.temperature),
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vrs_verifier_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = Request(
            urljoin(self.base_url, "chat/completions"),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible VLM request failed with HTTP {e.code}: {detail}"
            ) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible VLM request failed: {e.reason}") from e

        try:
            response = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError("OpenAI-compatible VLM returned invalid JSON response") from e
        return _completion_text(response)
