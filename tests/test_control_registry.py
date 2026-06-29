from vrs.control import (
    HealthStatus,
    StreamInput,
    StreamRegistry,
    summarize_queue_pressure,
    worker_health_from_queue_stats,
)


def test_stream_registry_upserts_lists_and_disables() -> None:
    registry = StreamRegistry()
    stream = StreamInput(stream_id="cam-01", source_uri="rtsp://example.local/1")

    entry = registry.upsert(stream, assigned_worker_id="deepstream-0")
    disabled = registry.disable("cam-01")

    assert entry.to_dict()["stream"]["schema_version"] == "stream.v1"
    assert registry.get("cam-01").enabled is False
    assert disabled.assigned_worker_id == "deepstream-0"
    assert registry.list(enabled_only=True) == []


def test_queue_pressure_summarizes_global_and_per_stream_stats() -> None:
    rows = summarize_queue_pressure(
        {
            "frame_q": {"size": 2, "dropped": 0},
            "sink": {
                "by_stream": {
                    "cam-01": {"size": 3, "dropped": 1},
                    "cam-02": {"size": 4, "dropped": 100},
                }
            },
        }
    )

    by_key = {(row.queue, row.stream_id): row for row in rows}
    assert by_key[("frame_q", "all")].status == HealthStatus.HEALTHY
    assert by_key[("sink", "cam-01")].status == HealthStatus.DEGRADED
    assert by_key[("sink", "cam-02")].status == HealthStatus.UNHEALTHY


def test_worker_health_rolls_up_queue_pressure() -> None:
    health = worker_health_from_queue_stats(
        worker_id="deepstream-0",
        gpu_role="deepstream",
        queue_stats={"candidate_q": {"size": 0, "dropped": 2}},
        details={"streams": 4},
    )

    payload = health.to_dict()
    assert health.status == HealthStatus.DEGRADED
    assert payload["worker_id"] == "deepstream-0"
    assert payload["gpu_role"] == "deepstream"
    assert payload["details"]["streams"] == 4
    assert payload["queue_pressure"][0]["status"] == "degraded"
