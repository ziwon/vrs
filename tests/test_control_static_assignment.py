import pytest

from vrs.control import StreamInput, assign_streams_static, render_worker_configs


def _streams(n: int) -> list[StreamInput]:
    return [
        StreamInput(stream_id=f"cam-{idx}", source_uri=f"rtsp://example.local/{idx}")
        for idx in range(n)
    ]


def test_static_assignment_is_deterministic_round_robin() -> None:
    assignments = assign_streams_static(_streams(5), worker_count=2)

    assert [(a.worker_id, a.stream.stream_id) for a in assignments] == [
        ("deepstream-0", "cam-0"),
        ("deepstream-1", "cam-1"),
        ("deepstream-0", "cam-2"),
        ("deepstream-1", "cam-3"),
        ("deepstream-0", "cam-4"),
    ]


def test_static_assignment_rejects_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="exceed static capacity"):
        assign_streams_static(_streams(3), worker_count=1, max_streams_per_worker=2)


def test_render_worker_configs_emits_stream_contracts() -> None:
    assignments = assign_streams_static(_streams(2), worker_count=2)

    configs = render_worker_configs(
        assignments,
        transport={"type": "redis-streams", "stream_prefix": "edge"},
        object_store={"type": "local-pvc", "mountPath": "/data"},
    )

    assert [cfg.worker_id for cfg in configs] == ["deepstream-0", "deepstream-1"]
    first = configs[0].to_dict()
    assert first["gpu_role"] == "deepstream"
    assert first["streams"][0]["schema_version"] == "stream.v1"
    assert first["streams"][0]["stream_id"] == "cam-0"
    assert first["transport"]["stream_prefix"] == "edge"
    assert first["object_store"]["mountPath"] == "/data"


def test_stream_input_accepts_existing_manifest_aliases() -> None:
    stream = StreamInput.from_mapping({"id": "cam-a", "rtsp": "rtsp://example.local/a"})

    assert stream.to_contract()["source_uri"] == "rtsp://example.local/a"
