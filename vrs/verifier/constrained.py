"""Constrained-decoding glue for the verifier.

When the optional ``xgrammar`` dependency is installed, we build a JSON-schema
constraint that forces Cosmos-Reason2-2B to emit *only* conformant JSON. The
schema mirrors the fields the parser in ``alert_verifier.py`` already expects,
with the additional guarantee that ``false_negative_class`` can only be one
of the watch-policy's class names (or ``null``) — the model can no longer
hallucinate a class that isn't on the list.

Graceful-fallback contract:

* ``build_verifier_schema(policy, …)`` is pure Python and always works.
* ``build_logits_processor(schema, tokenizer, vocab_size)`` returns
  ``None`` when ``xgrammar`` is unavailable or setup fails. Callers are
  expected to pass that ``None`` through — the verifier then runs
  unconstrained and relies on the existing balanced-brace parser fallback.

Why in-process ``LogitsProcessor`` instead of a separate server: it plugs
directly into ``transformers.generate``, needs no extra infra, and the
compiled grammar carries over when we later migrate to TRT-LLM guided
decoding (#9 on the roadmap).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_XGRAMMAR_UNAVAILABLE_LOGGED = False


def build_verifier_schema(
    class_names: Iterable[str],
    *,
    request_bbox: bool = True,
    request_trajectory: bool = True,
) -> Dict[str, Any]:
    """Return a JSON schema describing a valid verifier response.

    ``false_negative_class`` is constrained to an enum of the policy's class
    names plus ``null``, so the model cannot name a class that the pipeline
    doesn't know how to dispatch.
    """
    fn_enum: List[Any] = [None] + [str(n) for n in class_names]

    properties: Dict[str, Any] = {
        "true_alert": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "false_negative_class": {"enum": fn_enum},
        "rationale": {"type": "string"},
    }
    required = ["true_alert", "confidence", "false_negative_class", "rationale"]

    if request_bbox:
        properties["bbox_xywh_norm"] = {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": 4,
                    "maxItems": 4,
                },
            ]
        }
        required.append("bbox_xywh_norm")

    if request_trajectory:
        properties["trajectory_xy_norm"] = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "minItems": 2,
                "maxItems": 2,
            },
        }
        required.append("trajectory_xy_norm")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_logits_processor(
    schema: Dict[str, Any],
    tokenizer: Any,
    vocab_size: Optional[int] = None,
) -> Optional[Any]:
    """Compile the schema with XGrammar and return a ``LogitsProcessor``.

    Returns ``None`` if ``xgrammar`` is not installed, or if setup raises.
    The first "unavailable" outcome is logged at INFO once per process;
    repeated calls stay silent so multi-verify workloads don't spam.
    """
    global _XGRAMMAR_UNAVAILABLE_LOGGED

    try:
        import xgrammar as xgr
    except ImportError:
        if not _XGRAMMAR_UNAVAILABLE_LOGGED:
            logger.info(
                "xgrammar not installed — verifier runs unconstrained. "
                "Install the 'constrained' extra (pip install 'vrs[constrained]') "
                "to make JSON parse failures impossible."
            )
            _XGRAMMAR_UNAVAILABLE_LOGGED = True
        return None

    try:
        tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(tok_info)
        compiled = compiler.compile_json_schema(schema)
        # xgrammar's HF contrib is the transformers-friendly LogitsProcessor
        return xgr.contrib.hf.LogitsProcessor(compiled)
    except Exception as e:  # noqa: BLE001 — any xgrammar setup error → fall back
        logger.warning(
            "xgrammar setup failed (%s); verifier will run unconstrained this session.",
            e,
        )
        return None
