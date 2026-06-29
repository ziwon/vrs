from vrs.transport import EventMessage, InMemoryEventTransport, KafkaConfig, RedisStreamsConfig


def test_in_memory_transport_preserves_stream_order_and_cursor() -> None:
    transport = InMemoryEventTransport()
    first = EventMessage(
        stream="detections", key="cam-1", payload={"schema_version": "detection.v1"}
    )
    second = EventMessage(
        stream="detections",
        key="cam-1",
        payload={"schema_version": "candidate_alert.v1"},
    )

    first_id, second_id = transport.publish_many([first, second])

    assert first_id == "1"
    assert second_id == "2"
    assert transport.read("detections") == [first, second]
    assert transport.read("detections", after_id=first_id) == [second]
    assert transport.read("detections", limit=1) == [first]


def test_redis_and_kafka_configs_define_logical_name_mapping() -> None:
    assert RedisStreamsConfig(stream_prefix="edge").stream_name("alerts") == "edge.alerts"
    assert (
        KafkaConfig(bootstrap_servers="localhost:9092", topic_prefix="prod").topic_name("alerts")
        == "prod.alerts"
    )
