"""Stream source/sink protocols. An in-memory asyncio.Queue implementation
is used for tests and demos. redis_source.py (a Redis Streams adapter) is
a documented but not-yet-implemented stretch item -- see docs/ASSUMPTIONS.md.
Nothing in this repo's test suite requires it.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np


class StreamSource(Protocol):
    async def get(self) -> np.ndarray: ...


class AlertSink(Protocol):
    async def put(self, alert: dict) -> None: ...


class InMemoryStreamSource:
    """Backed by an asyncio.Queue -- push points in with `feed`, consume
    them with `get`. Good enough for tests and local demos; a real
    deployment would swap this for redis_source.py's Redis Streams
    adapter without changing anything downstream (same Protocol).
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    async def feed(self, x_t: np.ndarray) -> None:
        await self._queue.put(x_t)

    async def get(self) -> np.ndarray:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()


class InMemoryAlertSink:
    def __init__(self):
        self.alerts: list[dict] = []

    async def put(self, alert: dict) -> None:
        self.alerts.append(alert)
