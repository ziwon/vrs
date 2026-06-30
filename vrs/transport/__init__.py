"""Event transport contracts and adapters for VRS runtime records."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class EventMessage:
    """One canonical contract payload ready for a bus."""

    stream: str
    key: str
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class EventTransport(Protocol):
    """Minimal publish/read interface shared by edge and production buses."""

    def publish(self, message: EventMessage) -> str:
        """Publish one message and return the transport-assigned message id."""

    def read(
        self, stream: str, *, after_id: str | None = None, limit: int = 100
    ) -> list[EventMessage]:
        """Read messages from a logical stream."""


@dataclass(frozen=True)
class RedisStreamsConfig:
    """Configuration shape for the future edge-mode Redis Streams adapter."""

    url: str = "redis://localhost:6379/0"
    stream_prefix: str = "vrs"
    consumer_group: str = "vrs-runtime"
    max_len: int | None = 100_000

    def stream_name(self, logical_stream: str) -> str:
        return f"{self.stream_prefix}.{logical_stream}"


@dataclass(frozen=True)
class KafkaConfig:
    """Configuration shape for the future production Kafka adapter."""

    bootstrap_servers: str
    topic_prefix: str = "vrs"
    client_id: str = "vrs-runtime"
    acks: str = "all"

    def topic_name(self, logical_stream: str) -> str:
        return f"{self.topic_prefix}.{logical_stream}"


class InMemoryEventTransport:
    """Deterministic test transport implementing the production interface."""

    def __init__(self) -> None:
        self._streams: dict[str, deque[tuple[str, EventMessage]]] = defaultdict(deque)
        self._next_id = 1

    def publish(self, message: EventMessage) -> str:
        message_id = str(self._next_id)
        self._next_id += 1
        self._streams[message.stream].append((message_id, message))
        return message_id

    def publish_many(self, messages: Iterable[EventMessage]) -> list[str]:
        return [self.publish(message) for message in messages]

    def read(
        self, stream: str, *, after_id: str | None = None, limit: int = 100
    ) -> list[EventMessage]:
        if limit <= 0:
            return []
        after = int(after_id) if after_id is not None else 0
        out: list[EventMessage] = []
        for message_id, message in self._streams.get(stream, ()):
            if int(message_id) > after:
                out.append(message)
                if len(out) >= limit:
                    break
        return out


class RedisStreamsTransport:
    """Redis Streams transport for edge-mode worker handoff.

    The Redis client is imported lazily so unit tests and CPU-only development do
    not require a running Redis service. Tests can inject a small fake client that
    implements ``xadd``.
    """

    def __init__(self, config: RedisStreamsConfig | None = None, *, client: Any | None = None):
        self.config = config or RedisStreamsConfig()
        self.client = client or self._build_client(self.config.url)

    def publish(self, message: EventMessage) -> str:
        stream_name = self.config.stream_name(message.stream)
        fields: dict[str, str] = {
            "key": message.key,
            "payload": json.dumps(message.payload, ensure_ascii=False, separators=(",", ":")),
        }
        for name, value in message.headers.items():
            fields[f"header.{name}"] = value
        kwargs: dict[str, Any] = {}
        if self.config.max_len is not None:
            kwargs["maxlen"] = int(self.config.max_len)
            kwargs["approximate"] = True
        message_id = self.client.xadd(stream_name, fields, **kwargs)
        if isinstance(message_id, bytes):
            return message_id.decode("utf-8")
        return str(message_id)

    def read(
        self, stream: str, *, after_id: str | None = None, limit: int = 100
    ) -> list[EventMessage]:
        stream_name = self.config.stream_name(stream)
        start = after_id or "0-0"
        rows = self.client.xrange(stream_name, min=start, count=limit)
        out: list[EventMessage] = []
        for message_id, fields in rows:
            normalized = {_decode(key): _decode(value) for key, value in dict(fields).items()}
            if after_id is not None and _decode(message_id) == after_id:
                continue
            headers = {
                key.removeprefix("header."): value
                for key, value in normalized.items()
                if key.startswith("header.")
            }
            out.append(
                EventMessage(
                    stream=stream,
                    key=normalized.get("key", ""),
                    payload=json.loads(normalized["payload"]),
                    headers=headers,
                )
            )
        return out

    @staticmethod
    def _build_client(url: str) -> Any:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - only exercised without optional dep.
            raise RuntimeError("redis is required for RedisStreamsTransport") from exc
        return redis.Redis.from_url(url)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
