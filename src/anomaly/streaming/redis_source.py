"""Redis Streams adapter for streaming/sources.py's StreamSource/AlertSink
Protocols -- an alternative to InMemoryStreamSource for a real multi-process
deployment, where the in-memory asyncio.Queue can't be shared across
processes.

STRETCH ITEM: not required by anything in this repo's test suite. Import
is guarded so the rest of the service works with zero redis dependency
installed; only touches the network if you actually construct one of these
classes. See docker-compose.yml's `redis` profile.
"""

from __future__ import annotations

import json

import numpy as np

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover -- exercised only when redis is installed
    redis = None


def _require_redis() -> None:
    if redis is None:
        raise ImportError(
            "the 'redis' package is not installed. Run `pip install redis` to use "
            "RedisStreamSource / RedisAlertSink."
        )


class RedisStreamSource:
    """Backed by a Redis Stream: `feed` XADDs a point, `get` XREADs the
    next one (blocking). Same interface as InMemoryStreamSource, so
    swapping one for the other requires no downstream changes.
    """

    def __init__(self, url: str = "redis://localhost:6379", stream_key: str = "anomaly:input"):
        _require_redis()
        self._client = redis.from_url(url)
        self.stream_key = stream_key
        self._last_id = "$"  # "$" -- only new entries, not the stream's history

    async def feed(self, x_t: np.ndarray) -> None:
        await self._client.xadd(self.stream_key, {"data": json.dumps(np.asarray(x_t).tolist())})

    async def get(self) -> np.ndarray:
        # block indefinitely for the next entry after self._last_id
        result = await self._client.xread({self.stream_key: self._last_id}, count=1, block=0)
        _stream_key, entries = result[0]
        entry_id, fields = entries[0]
        self._last_id = entry_id
        return np.array(json.loads(fields[b"data"]))


class RedisAlertSink:
    def __init__(self, url: str = "redis://localhost:6379", stream_key: str = "anomaly:alerts"):
        _require_redis()
        self._client = redis.from_url(url)
        self.stream_key = stream_key

    async def put(self, alert: dict) -> None:
        await self._client.xadd(self.stream_key, {"data": json.dumps(alert)})
