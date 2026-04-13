"""Per-class temporal-persistence queue.

Promotes streams of YOLOE detections into ``CandidateAlert`` records once an
event has been seen on enough frames inside a sliding window. Each class also
gets a debounce window (``cooldown_s``) so we don't re-alert on the same event.

The queue also maintains a small ring buffer of recent ``Frame`` images per
class so the verifier receives a true short clip rather than a single peak frame.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from ..policy import WatchPolicy
from ..schemas import CandidateAlert, Detection, Frame


@dataclass
class _ClassState:
    hits: Deque[bool]                 # last `window` frame outcomes
    fill_start_pts: Optional[float]   # pts of the first hit in the current run
    last_alert_pts: Optional[float]   # last time an alert fired (cooldown anchor)


class EventStateQueue:
    def __init__(
        self,
        policy: WatchPolicy,
        window: int = 8,
        cooldown_s: float = 10.0,
        keyframes: int = 6,
        context_window_s: float = 3.0,
        target_fps: float = 4.0,
    ):
        self.policy = policy
        self.window = int(window)
        self.cooldown_s = float(cooldown_s)
        self.keyframes = int(keyframes)
        self.context_window_s = float(context_window_s)

        self._state: Dict[str, _ClassState] = {
            it.name: _ClassState(
                hits=deque([False] * self.window, maxlen=self.window),
                fill_start_pts=None,
                last_alert_pts=None,
            )
            for it in policy
        }

        # rolling buffer of (frame, detections) for keyframe extraction
        ring_len = max(int(context_window_s * target_fps) * 2 + 4, 16)
        self._ring: Deque[Tuple[Frame, List[Detection]]] = deque(maxlen=ring_len)

    # ---- main api ---------------------------------------------------

    def step(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> List[CandidateAlert]:
        """Push one frame's detections; return a CandidateAlert per newly-fired class."""
        self._ring.append((frame, list(detections)))

        # group hits by class for this frame
        hit_classes = {d.class_name for d in detections}

        alerts: List[CandidateAlert] = []
        for name, state in self._state.items():
            hit = name in hit_classes
            state.hits.append(hit)

            if hit and state.fill_start_pts is None:
                state.fill_start_pts = frame.pts_s
            if sum(state.hits) == 0:
                state.fill_start_pts = None

            min_persist = self.policy[name].min_persist_frames
            if not hit or sum(state.hits) < min_persist:
                continue

            # cooldown debounce
            last = state.last_alert_pts
            if last is not None and (frame.pts_s - last) < self.cooldown_s:
                continue
            state.last_alert_pts = frame.pts_s

            peak_dets = [d for d in detections if d.class_name == name]
            keyframes, keyframe_pts = self._sample_keyframes(frame.pts_s)
            alerts.append(
                CandidateAlert(
                    class_name=name,
                    severity=self.policy[name].severity,
                    start_pts_s=state.fill_start_pts or frame.pts_s,
                    peak_pts_s=frame.pts_s,
                    peak_frame_index=frame.index,
                    peak_detections=peak_dets,
                    keyframes=keyframes,
                    keyframe_pts=keyframe_pts,
                )
            )
        return alerts

    # ---- helpers ----------------------------------------------------

    def _sample_keyframes(self, peak_pts_s: float):
        """Pick ``self.keyframes`` evenly-spaced frames around the peak."""
        if not self._ring:
            return [], []
        t_lo = peak_pts_s - self.context_window_s
        t_hi = peak_pts_s + self.context_window_s
        candidates = [pair for pair in self._ring if t_lo <= pair[0].pts_s <= t_hi]
        if not candidates:
            candidates = list(self._ring)

        n = min(self.keyframes, len(candidates))
        if n <= 0:
            return [], []
        step = max(1, len(candidates) // n)
        picked = candidates[:: step][:n]
        return (
            [pair[0].image.copy() for pair in picked],
            [float(pair[0].pts_s) for pair in picked],
        )
