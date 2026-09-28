"""Bounded staging queue implementing high/low watermark backpressure."""

import asyncio

from warden_ingestion.models import ChunkPayload


class BoundedChunkQueue:
    """Bounded chunk staging buffer enforcing memory containment via watermark backpressure."""

    def __init__(self, maxsize: int = 256, low_watermark: int = 128) -> None:
        self.maxsize = maxsize
        self.low_watermark = low_watermark
        self._queue: asyncio.Queue[ChunkPayload] = asyncio.Queue()
        self._can_push = asyncio.Event()
        self._can_push.set()

    def qsize(self) -> int:
        """Return current depth of queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return self._queue.empty()

    async def push(self, chunk: ChunkPayload) -> None:
        """Push a chunk onto the queue, blocking if high watermark is reached."""
        while self.qsize() >= self.maxsize:
            self._can_push.clear()
            await self._can_push.wait()

        await self._queue.put(chunk)
        if self.qsize() >= self.maxsize:
            self._can_push.clear()

    async def pop(self) -> ChunkPayload:
        """Pop a single chunk, unblocking producers if drained below low watermark."""
        chunk = await self._queue.get()
        if self.qsize() <= self.low_watermark and not self._can_push.is_set():
            self._can_push.set()
        return chunk

    async def pop_batch(self, max_batch: int = 64) -> list[ChunkPayload]:
        """Pop up to max_batch chunks atomically from queue."""
        batch: list[ChunkPayload] = []
        for _ in range(max_batch):
            if self._queue.empty():
                break
            batch.append(self._queue.get_nowait())

        if self.qsize() <= self.low_watermark and not self._can_push.is_set():
            self._can_push.set()

        return batch
