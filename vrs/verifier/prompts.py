"""Prompt templates for Cosmos-Reason2-2B verification.

Cosmos-Reason2-2B is post-trained for *physical-world* reasoning with native
chain-of-thought, bounding-box, and trajectory output. We exploit all three:

  - CoT to actually reason about whether the detector's claim matches the
    physics of the scene (e.g. "is that orange glow a fire or a sunset?").
  - bbox to give a precise spatial answer that the annotator can draw.
  - trajectory to summarize moving subjects (e.g. the falling person's path).

We ask for strict JSON so parsing is deterministic. Coordinates are normalized
to [0, 1] in (x, y, w, h) to be robust against frame-resize.
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = (
    "You are a safety operator reviewing a CCTV alert. "
    "An automated detector has flagged a candidate event in the video clip you are watching. "
    "Use physical-world common sense — gravity, occupancy, scale, and motion continuity — "
    "to decide whether the event is real, and whether the detector missed any other listed event. "
    "Reason step by step internally, but reply with a single JSON object only. "
    "All bounding boxes and trajectory points are in normalized [0,1] coordinates relative to frame size."
)


def build_user_prompt(
    detector_class: str,
    detector_definition: str,
    start_pts_s: float,
    peak_pts_s: float,
    keyframe_pts: Sequence[float],
    known_events: Sequence[tuple[str, str]],  # (name, definition)
    request_bbox: bool = True,
    request_trajectory: bool = True,
) -> str:
    pts_str = ", ".join(f"{p:.2f}s" for p in keyframe_pts)
    other = "\n".join(
        f"  - {name}: {definition}" for name, definition in known_events if name != detector_class
    )
    bbox_field = '  "bbox_xywh_norm": [<x>, <y>, <w>, <h>] | null,\n' if request_bbox else ""
    traj_field = (
        '  "trajectory_xy_norm": [[<x1>,<y1>], [<x2>,<y2>], ...] | [],\n'
        if request_trajectory
        else ""
    )
    fn_options = ", ".join(f'"{n}"' for n, _ in known_events if n != detector_class) or "<none>"

    return (
        f'Detector claim: "{detector_class}" — {detector_definition}\n'
        f"Time range: t={start_pts_s:.2f}s .. t={peak_pts_s:.2f}s.\n"
        f"Frames sampled in order at: [{pts_str}].\n\n"
        f"Other listed safety events to watch for as false negatives:\n{other or '  (none)'}\n\n"
        "Reply with EXACTLY this JSON object — no prose, no code fence:\n"
        "{\n"
        '  "true_alert": <true|false>,\n'
        '  "confidence": <float in [0.0, 1.0]>,\n'
        f'  "false_negative_class": <null | one of [{fn_options}]>,\n'
        f"{bbox_field}{traj_field}"
        '  "rationale": "<one short sentence describing what you saw>"\n'
        "}\n\n"
        "Decision rules:\n"
        "  - true_alert=true ONLY if the detector's claimed event is actually visible "
        "AND consistent with physics across the frames.\n"
        "  - Set false_negative_class to one of the listed events when a different "
        "listed event is visible but the detector's claim is NOT — this tells the "
        "operator that the fast detector missed something.\n"
        "  - When the scene is benign, return true_alert=false and false_negative_class=null."
    )
