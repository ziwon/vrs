"""Cross-camera incident correlation for verified multistream alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import VerifiedAlert


@dataclass(frozen=True)
class IncidentCorrelationConfig:
    enabled: bool = False
    window_s: float = 5.0
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    match_class: bool = True
    match_severity: bool = True
    include_false_alerts: bool = False
    min_bbox_iou: float | None = None

    @staticmethod
    def from_mapping(cfg: dict[str, Any] | None) -> IncidentCorrelationConfig:
        raw = cfg or {}
        adjacency = _parse_adjacency(raw.get("adjacency") or raw.get("overlap_map"))
        min_bbox_iou = raw.get("min_bbox_iou")
        if min_bbox_iou is not None:
            min_bbox_iou = float(min_bbox_iou)
            if not 0.0 <= min_bbox_iou <= 1.0:
                raise ValueError("multistream.incident_correlation.min_bbox_iou must be in [0, 1]")
        window_s = float(raw.get("window_s", 5.0))
        if window_s < 0:
            raise ValueError("multistream.incident_correlation.window_s must be >= 0")
        return IncidentCorrelationConfig(
            enabled=bool(raw.get("enabled", False)),
            window_s=window_s,
            adjacency=adjacency,
            match_class=bool(raw.get("match_class", True)),
            match_severity=bool(raw.get("match_severity", True)),
            include_false_alerts=bool(raw.get("include_false_alerts", False)),
            min_bbox_iou=min_bbox_iou,
        )


@dataclass
class _Incident:
    id: str
    primary_stream_id: str
    class_name: str
    severity: str
    last_pts_s: float
    stream_ids: set[str] = field(default_factory=set)
    last_bbox_xywh_norm: tuple[float, float, float, float] | None = None


class IncidentCorrelator:
    """Assign stable incident ids to related alerts from overlapping cameras."""

    def __init__(self, cfg: IncidentCorrelationConfig | dict[str, Any] | None = None):
        self.cfg = (
            cfg
            if isinstance(cfg, IncidentCorrelationConfig)
            else (IncidentCorrelationConfig.from_mapping(cfg))
        )
        self._next_id = 1
        self._active: list[_Incident] = []

    def assign(self, stream_id: str, alert: VerifiedAlert) -> VerifiedAlert:
        if not self.cfg.enabled:
            return alert
        if not alert.true_alert and not self.cfg.include_false_alerts:
            return alert

        pts_s = float(alert.candidate.peak_pts_s)
        self._expire(pts_s)
        incident = self._find_match(stream_id, alert)
        if incident is None:
            incident = self._new_incident(stream_id, alert)
            self._active.append(incident)
        else:
            incident.stream_ids.add(stream_id)
            incident.last_pts_s = max(incident.last_pts_s, pts_s)
            bbox = _alert_bbox(alert)
            if bbox is not None:
                incident.last_bbox_xywh_norm = bbox

        alert.incident_id = incident.id
        alert.incident_stream_ids = sorted(incident.stream_ids)
        alert.incident_primary_stream_id = incident.primary_stream_id
        return alert

    def _expire(self, pts_s: float) -> None:
        window = self.cfg.window_s
        self._active = [inc for inc in self._active if pts_s - inc.last_pts_s <= window]

    def _find_match(self, stream_id: str, alert: VerifiedAlert) -> _Incident | None:
        for incident in self._active:
            if abs(float(alert.candidate.peak_pts_s) - incident.last_pts_s) > self.cfg.window_s:
                continue
            if self.cfg.match_class and alert.candidate.class_name != incident.class_name:
                continue
            if self.cfg.match_severity and alert.candidate.severity != incident.severity:
                continue
            if not self._streams_overlap(stream_id, incident.stream_ids):
                continue
            if not self._geometry_matches(alert, incident):
                continue
            return incident
        return None

    def _streams_overlap(self, stream_id: str, incident_stream_ids: set[str]) -> bool:
        if stream_id in incident_stream_ids:
            return True
        if not self.cfg.adjacency:
            return False
        neighbors = self.cfg.adjacency.get(stream_id, set())
        return any(other in neighbors for other in incident_stream_ids)

    def _geometry_matches(self, alert: VerifiedAlert, incident: _Incident) -> bool:
        threshold = self.cfg.min_bbox_iou
        if threshold is None:
            return True
        bbox = _alert_bbox(alert)
        if bbox is None or incident.last_bbox_xywh_norm is None:
            return False
        return _xywh_iou(bbox, incident.last_bbox_xywh_norm) >= threshold

    def _new_incident(self, stream_id: str, alert: VerifiedAlert) -> _Incident:
        incident_id = f"inc-{self._next_id:06d}"
        self._next_id += 1
        return _Incident(
            id=incident_id,
            primary_stream_id=stream_id,
            class_name=alert.candidate.class_name,
            severity=alert.candidate.severity,
            last_pts_s=float(alert.candidate.peak_pts_s),
            stream_ids={stream_id},
            last_bbox_xywh_norm=_alert_bbox(alert),
        )


def _parse_adjacency(raw: Any) -> dict[str, set[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("multistream.incident_correlation.adjacency must be a mapping")
    out: dict[str, set[str]] = {}
    for stream_id, neighbors in raw.items():
        sid = str(stream_id)
        if neighbors is None:
            out.setdefault(sid, set())
            continue
        if isinstance(neighbors, str):
            raise ValueError("multistream.incident_correlation.adjacency values must be lists")
        try:
            neighbor_set = {str(item) for item in neighbors}
        except TypeError as exc:
            raise ValueError(
                "multistream.incident_correlation.adjacency values must be iterable"
            ) from exc
        out.setdefault(sid, set()).update(neighbor_set)
        for neighbor in neighbor_set:
            out.setdefault(neighbor, set()).add(sid)
    return out


def _alert_bbox(alert: VerifiedAlert) -> tuple[float, float, float, float] | None:
    return alert.bbox_xywh_norm or alert.candidate.detector_bbox_xywh_norm()


def _xywh_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union
