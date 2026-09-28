"""Cache invalidation event publisher over Redis Pub/Sub."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger("warden.ingestion.events")


class CacheInvalidationPublisher:
    """Publishes ingestion cache invalidation notifications to Redis."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        redis_url: str = "redis://warden-cache-redis:6379",
    ) -> None:
        self.redis_url = redis_url
        self._client = redis_client
        self._owned_client = False

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._owned_client = True
        return self._client

    async def publish_invalidation(self, affected_roles: list[str], run_id: str) -> None:
        """Broadcast cache invalidation event to warden:cache:invalidate topic."""
        if not affected_roles:
            return

        client = await self._get_client()
        event_payload = {
            "event": "INGESTION_COMPLETED",
            "run_id": run_id,
            "affected_roles": sorted(list(set(affected_roles))),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        message = json.dumps(event_payload)
        channel = "warden:cache:invalidate"
        try:
            await client.publish(channel, message)
            logger.info("Published cache invalidation event to %s: %s", channel, message)
        except Exception as e:
            logger.warning("Failed to publish cache invalidation event to Redis: %s", e)

    async def close(self) -> None:
        """Close connection if owned by this publisher instance."""
        if self._owned_client and self._client is not None:
            await self._client.close()
            self._client = None
