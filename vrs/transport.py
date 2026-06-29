"""Event transport contracts for VRS runtime records.

The production adapters are intentionally not implemented here; this module
defines the stable boundary that Redis Streams and Kafka implementations must
match while keeping unit tests service-free.
"""

from __future__ import annotations

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
