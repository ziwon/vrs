"""Thread-safe bounded queue with explicit drop policies.

The standard ``queue.Queue`` only offers block-on-full, which is the wrong
default for live video — one slow consumer should never freeze a producer
that's reading a real-time RTSP stream. We support three policies:

  * DROP_OLDEST  — when full, evict the oldest item to make room (default).
  * DROP_NEWEST  — when full, refuse the put silently and return False.
  * BLOCK        — classic blocking semantics.

Returns the number of *dropped* items via ``puts_dropped`` so the caller can
expose backpressure metrics.
"""
from __future__ import annotations

import threading
from collections import deque
from enum import Enum
from typing import Any, Deque, Optional


class DropPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class BoundedQueue:
    def __init__(self, maxsize: int, policy: DropPolicy = DropPolicy.DROP_OLDEST):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self.policy = DropPolicy(policy)

        self._buf: Deque[Any] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

        self.puts_total = 0
        self.puts_dropped = 0

    # ---- producer side ---------------------------------------------

    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Insert an item.

        Returns True if the item is in the queue, False if it was dropped per
        policy. With BLOCK + timeout=None, always returns True (or raises).
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("queue is closed")

            self.puts_total += 1
            if len(self._buf) < self.maxsize:
                self._buf.append(item)
                self._not_empty.notify()
                return True

            if self.policy is DropPolicy.DROP_OLDEST:
                # ring-buffer: evict oldest, append newest
                self._buf.popleft()
                self._buf.append(item)
                self.puts_dropped += 1
                self._not_empty.notify()
                return True

            if self.policy is DropPolicy.DROP_NEWEST:
                self.puts_dropped += 1
                return False

            # BLOCK
            ok = self._not_full.wait_for(
                lambda: len(self._buf) < self.maxsize or self._closed,
                timeout=timeout,
            )
            if self._closed:
                raise RuntimeError("queue is closed")
            if not ok:
                self.puts_dropped += 1
                return False
            self._buf.append(item)
            self._not_empty.notify()
            return True

    # ---- consumer side ---------------------------------------------

    def get(self, timeout: Optional[float] = None) -> Any:
        """Pop the oldest item; raises ``TimeoutError`` if the wait elapses."""
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._buf or self._closed,
                timeout=timeout,
            )
            if not ok:
                raise TimeoutError("BoundedQueue.get timed out")
            if not self._buf and self._closed:
                raise StopIteration("queue closed and drained")
            item = self._buf.popleft()
            self._not_full.notify()
            return item

    def get_batch(self, max_items: int, timeout: float) -> list:
        """Drain up to ``max_items`` items, waiting at most ``timeout`` seconds
        for the *first* item then taking whatever else is already buffered.
        Returns [] if the timeout elapses with nothing in the queue."""
        out = []
        try:
            out.append(self.get(timeout=timeout))
        except TimeoutError:
            return out
        with self._lock:
            while self._buf and len(out) < max_items:
                out.append(self._buf.popleft())
            if out:
                self._not_full.notify_all()
        return out

    # ---- lifecycle -------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def qsize(self) -> int:
        with self._lock:
            return len(self._buf)

    def __len__(self) -> int:  # for tests / introspection
        return self.qsize()
