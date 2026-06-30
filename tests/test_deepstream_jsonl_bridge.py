import json

from vrs.deepstream.jsonl_bridge import detection_message, publish_jsonl_file


def test_detection_message_uses_idempotency_key_and_headers() -> None:
    record = {
        "schema_version": "detection.v1",
        "record_type": "detection",
        "idempotency_key": "det-1",
        "source_runtime": "deepstream",
    }

    message = detection_message(record, stream="detections")

    assert message.stream == "detections"
    assert message.key == "det-1"
    assert message.payload == record
    assert message.headers["schema_version"] == "detection.v1"
    assert message.headers["source_runtime"] == "deepstream"


def test_publish_jsonl_file_publishes_each_detection(tmp_path) -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message):
            self.messages.append(message)
            return str(len(self.messages))

    path = tmp_path / "detections.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"schema_version": "detection.v1", "detection_id": "det-1"}),
                json.dumps({"schema_version": "detection.v1", "detection_id": "det-2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    transport = FakeTransport()

    count = publish_jsonl_file(path, transport, stream="detections")

    assert count == 2
    assert [message.key for message in transport.messages] == ["det-1", "det-2"]
