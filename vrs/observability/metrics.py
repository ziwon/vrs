"""Prometheus-compatible runtime metrics for VRS.

The implementation intentionally uses only the Python standard library. VRS
can expose a scrape endpoint on appliances without adding another runtime
dependency, and tests can exercise metric behavior without importing GPU
libraries.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DEFAULT_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def _check_metric_name(name: str) -> None:
    if not _METRIC_NAME_RE.match(name):
        raise ValueError(f"invalid metric name: {name!r}")


def _check_label_names(labelnames: tuple[str, ...]) -> None:
    for label in labelnames:
        if not _LABEL_NAME_RE.match(label):
            raise ValueError(f"invalid label name: {label!r}")


def _escape_label_value(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return f"{value:g}"


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label_value(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


class _Metric:
    def __init__(self, name: str, help_text: str, kind: str, labelnames: tuple[str, ...]):
        _check_metric_name(name)
        _check_label_names(labelnames)
        self.name = name
        self.help_text = help_text
        self.kind = kind
        self.labelnames = labelnames
        self._lock = threading.RLock()

    def _labels_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        expected = set(self.labelnames)
        actual = set(labels)
        if expected != actual:
            raise ValueError(
                f"{self.name}: expected labels {sorted(expected)}, got {sorted(actual)}"
            )
        return tuple(str(labels[name]) for name in self.labelnames)

    def render(self) -> list[str]:
        raise NotImplementedError


class _Counter(_Metric):
    def __init__(self, name: str, help_text: str, labelnames: tuple[str, ...]):
        super().__init__(name, help_text, "counter", labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counter increment must be non-negative")
        with self._lock:
            key = self._labels_key(labels)
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def set_total(self, value: float, **labels: str) -> None:
        if value < 0:
            raise ValueError("counter total must be non-negative")
        with self._lock:
            key = self._labels_key(labels)
            self._values[key] = max(self._values.get(key, 0.0), float(value))

    def render(self) -> list[str]:
        with self._lock:
            lines = []
            for key, value in sorted(self._values.items()):
                labels = dict(zip(self.labelnames, key, strict=True))
                lines.append(f"{self.name}{_format_labels(labels)} {_format_number(value)}")
            return lines


class _Gauge(_Metric):
    def __init__(self, name: str, help_text: str, labelnames: tuple[str, ...]):
        super().__init__(name, help_text, "gauge", labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[self._labels_key(labels)] = float(value)

    def render(self) -> list[str]:
        with self._lock:
            lines = []
            for key, value in sorted(self._values.items()):
                labels = dict(zip(self.labelnames, key, strict=True))
                lines.append(f"{self.name}{_format_labels(labels)} {_format_number(value)}")
            return lines


class _Histogram(_Metric):
    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...],
        buckets: tuple[float, ...],
    ):
        super().__init__(name, help_text, "histogram", labelnames)
        self._buckets = tuple(sorted(float(b) for b in buckets))
        self._values: dict[tuple[str, ...], tuple[list[int], int, float]] = {}

    def observe(self, value: float, **labels: str) -> None:
        with self._lock:
            key = self._labels_key(labels)
            counts, total_count, total_sum = self._values.get(
                key, ([0 for _ in self._buckets], 0, 0.0)
            )
            value = float(value)
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    counts[i] += 1
            self._values[key] = (counts, total_count + 1, total_sum + value)

    def render(self) -> list[str]:
        with self._lock:
            lines = []
            for key, (counts, total_count, total_sum) in sorted(self._values.items()):
                base_labels = dict(zip(self.labelnames, key, strict=True))
                for bound, count in zip(self._buckets, counts, strict=True):
                    labels = {**base_labels, "le": _format_number(bound)}
                    lines.append(
                        f"{self.name}_bucket{_format_labels(labels)} {_format_number(count)}"
                    )
                labels = {**base_labels, "le": "+Inf"}
                lines.append(
                    f"{self.name}_bucket{_format_labels(labels)} {_format_number(total_count)}"
                )
                lines.append(
                    f"{self.name}_count{_format_labels(base_labels)} {_format_number(total_count)}"
                )
                lines.append(
                    f"{self.name}_sum{_format_labels(base_labels)} {_format_number(total_sum)}"
                )
            return lines


class MetricsRegistry:
    """Small thread-safe registry that renders Prometheus text exposition."""

    def __init__(self):
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.RLock()

    def counter(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> _Counter:
        return self._register(_Counter(name, help_text, labelnames), _Counter)

    def gauge(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> _Gauge:
        return self._register(_Gauge(name, help_text, labelnames), _Gauge)

    def histogram(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = _DEFAULT_LATENCY_BUCKETS,
    ) -> _Histogram:
        return self._register(_Histogram(name, help_text, labelnames, buckets), _Histogram)

    def _register(self, metric: _Metric, expected_type: type) -> Any:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if not isinstance(existing, expected_type):
                    raise ValueError(f"metric already registered with another type: {metric.name}")
                if existing.labelnames != metric.labelnames:
                    raise ValueError(
                        f"metric already registered with different labels: {metric.name}"
                    )
                return existing
            self._metrics[metric.name] = metric
            return metric

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for metric in self._metrics.values():
                lines.append(f"# HELP {metric.name} {metric.help_text}")
                lines.append(f"# TYPE {metric.name} {metric.kind}")
                lines.extend(metric.render())
            return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9108
    path: str = "/metrics"

    @classmethod
    def from_app_config(cls, cfg: dict[str, Any]) -> MetricsConfig:
        obs = cfg.get("observability") or {}
        raw = obs.get("metrics") or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            host=str(raw.get("host", "127.0.0.1")),
            port=int(raw.get("port", 9108)),
            path=str(raw.get("path", "/metrics")),
        )


class MetricsServer:
    def __init__(self, registry: MetricsRegistry, config: MetricsConfig):
        self.registry = registry
        self.config = config
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        registry = self.registry
        path = self.config.path

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != path:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = registry.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="metrics-server",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str | None:
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address
        return f"http://{host}:{port}{self.config.path}"

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None


class NullVRSMetrics:
    enabled = False
    registry: MetricsRegistry | None = None
    url: str | None = None

    def set_queue_depth(self, queue: str, stream_id: str, depth: int) -> None:
        return

    def set_queue_dropped_total(self, queue: str, stream_id: str, dropped: int) -> None:
        return

    def inc_candidates(self, stream_id: str, class_name: str, amount: int = 1) -> None:
        return

    def inc_verified_alerts(
        self, stream_id: str, class_name: str, verdict: str, amount: int = 1
    ) -> None:
        return

    def observe_detector_latency(self, seconds: float) -> None:
        return

    def observe_verifier_latency(self, seconds: float) -> None:
        return

    def inc_verifier_errors(self, backend: str, amount: int = 1) -> None:
        return

    def inc_sink_write_errors(self, stream_id: str, amount: int = 1) -> None:
        return

    def close(self) -> None:
        return


class VRSMetrics:
    enabled = True

    def __init__(
        self, registry: MetricsRegistry | None = None, server: MetricsServer | None = None
    ):
        self.registry = registry or MetricsRegistry()
        self._server = server
        self.queue_depth = self.registry.gauge(
            "vrs_queue_depth",
            "Current VRS queue depth.",
            ("queue", "stream_id"),
        )
        self.queue_dropped = self.registry.counter(
            "vrs_queue_dropped_total",
            "Total VRS queue items dropped by bounded queues.",
            ("queue", "stream_id"),
        )
        self.candidates = self.registry.counter(
            "vrs_candidates_total",
            "Total detector candidates promoted for verification.",
            ("stream_id", "class"),
        )
        self.verified_alerts = self.registry.counter(
            "vrs_verified_alerts_total",
            "Total verified alerts by verdict.",
            ("stream_id", "class", "verdict"),
        )
        self.detector_latency = self.registry.histogram(
            "vrs_detector_latency_seconds",
            "Detector inference latency in seconds.",
        )
        self.verifier_latency = self.registry.histogram(
            "vrs_verifier_latency_seconds",
            "Verifier latency in seconds.",
        )
        self.verifier_errors = self.registry.counter(
            "vrs_verifier_errors_total",
            "Total verifier errors by backend.",
            ("backend",),
        )
        self.sink_write_errors = self.registry.counter(
            "vrs_sink_write_errors_total",
            "Total sink write errors by stream.",
            ("stream_id",),
        )

    @property
    def url(self) -> str | None:
        return self._server.url if self._server is not None else None

    def set_queue_depth(self, queue: str, stream_id: str, depth: int) -> None:
        self.queue_depth.set(float(depth), queue=queue, stream_id=stream_id)

    def set_queue_dropped_total(self, queue: str, stream_id: str, dropped: int) -> None:
        self.queue_dropped.set_total(float(dropped), queue=queue, stream_id=stream_id)

    def inc_candidates(self, stream_id: str, class_name: str, amount: int = 1) -> None:
        self.candidates.inc(float(amount), stream_id=stream_id, **{"class": class_name})

    def inc_verified_alerts(
        self, stream_id: str, class_name: str, verdict: str, amount: int = 1
    ) -> None:
        self.verified_alerts.inc(
            float(amount), stream_id=stream_id, verdict=verdict, **{"class": class_name}
        )

    def observe_detector_latency(self, seconds: float) -> None:
        self.detector_latency.observe(float(seconds))

    def observe_verifier_latency(self, seconds: float) -> None:
        self.verifier_latency.observe(float(seconds))

    def inc_verifier_errors(self, backend: str, amount: int = 1) -> None:
        self.verifier_errors.inc(float(amount), backend=backend)

    def inc_sink_write_errors(self, stream_id: str, amount: int = 1) -> None:
        self.sink_write_errors.inc(float(amount), stream_id=stream_id)

    def close(self) -> None:
        if self._server is not None:
            self._server.close()


def build_metrics(config: dict[str, Any]) -> VRSMetrics | NullVRSMetrics:
    metrics_config = MetricsConfig.from_app_config(config)
    if not metrics_config.enabled:
        return NullVRSMetrics()

    registry = MetricsRegistry()
    server = MetricsServer(registry, metrics_config)
    metrics = VRSMetrics(registry=registry, server=server)
    server.start()
    return metrics
