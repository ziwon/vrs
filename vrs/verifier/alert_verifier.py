"""Stage-2 verifier: CandidateAlert → VerifiedAlert (using Cosmos-Reason2-2B).

Hands the keyframe clip to Cosmos-Reason2-2B with a strict-JSON prompt; parses
the verdict, bounding box, and trajectory; returns a VerifiedAlert. On any
parsing or model failure we fall back to a pass-through verdict so the
pipeline never silently drops a detector hit.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from ..policy import WatchPolicy
from ..runtime import CosmosReason2
from ..schemas import CandidateAlert, VerifiedAlert
from .prompts import SYSTEM_PROMPT, build_user_prompt


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    for candidate in (blob, blob.replace("'", '"')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _coerce_bbox(value) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (x, y, w, h)


def _coerce_trajectory(value) -> List[Tuple[float, float]]:
    if not value:
        return []
    out: List[Tuple[float, float]] = []
    for pt in value:
        try:
            x, y = (float(v) for v in pt)
        except (TypeError, ValueError):
            continue
        out.append((x, y))
    return out


class AlertVerifier:
    def __init__(
        self,
        cosmos: CosmosReason2,
        policy: WatchPolicy,
        request_bbox: bool = True,
        request_trajectory: bool = True,
        clip_fps: int = 4,
    ):
        self.cosmos = cosmos
        self.policy = policy
        self.request_bbox = bool(request_bbox)
        self.request_trajectory = bool(request_trajectory)
        self.clip_fps = int(clip_fps)
        self._known_events = [(it.name, it.verifier_prompt) for it in policy]

    def verify(self, alert: CandidateAlert) -> VerifiedAlert:
        if not alert.keyframes:
            return VerifiedAlert(
                candidate=alert,
                true_alert=True,
                confidence=0.0,
                false_negative_class=None,
                rationale="no keyframes available for verification",
            )

        item = self.policy[alert.class_name]
        user_msg = build_user_prompt(
            detector_class=alert.class_name,
            detector_definition=item.verifier_prompt,
            start_pts_s=alert.start_pts_s,
            peak_pts_s=alert.peak_pts_s,
            keyframe_pts=alert.keyframe_pts,
            known_events=self._known_events,
            request_bbox=self.request_bbox,
            request_trajectory=self.request_trajectory,
        )

        try:
            raw = self.cosmos.chat_video(
                SYSTEM_PROMPT, user_msg, alert.keyframes, clip_fps=self.clip_fps
            )
        except Exception as e:  # noqa: BLE001 — surface in rationale, never crash
            return VerifiedAlert(
                candidate=alert,
                true_alert=True,
                confidence=0.0,
                false_negative_class=None,
                rationale=f"verifier error: {e}",
            )

        parsed = _safe_parse_json(raw) or {}

        true_alert = bool(parsed.get("true_alert", True))
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        fn_cls = parsed.get("false_negative_class")
        if fn_cls in (None, "null", "None", "", "<none>"):
            fn_cls = None
        elif fn_cls not in self.policy.names():
            fn_cls = None

        rationale = str(parsed.get("rationale", "")).strip() or raw.strip()[:240]

        return VerifiedAlert(
            candidate=alert,
            true_alert=true_alert,
            confidence=max(0.0, min(1.0, conf)),
            false_negative_class=fn_cls,
            rationale=rationale,
            bbox_xywh_norm=_coerce_bbox(parsed.get("bbox_xywh_norm")),
            trajectory_xy_norm=_coerce_trajectory(parsed.get("trajectory_xy_norm")),
            verifier_raw=raw,
        )
