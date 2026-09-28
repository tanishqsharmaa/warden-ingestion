"""Streaming gRPC client for warden-retrieval IndexBatch RPC with exponential backoff."""

import asyncio
import logging
from typing import Any, Optional

import grpc
from warden_shared.proto.v1 import retrieval_pb2, retrieval_pb2_grpc

from warden_ingestion.models import VectorizedPoint

logger = logging.getLogger("warden.ingestion.grpc_client")


class RetrievalGrpcClient:
    """High-throughput gRPC client communicating with warden-retrieval."""

    def __init__(
        self,
        target_url: str = "warden-retrieval:50051",
        stub: Optional[Any] = None,
        retry_backoff_base: float = 1.0,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self.target_url = target_url
        self.retry_backoff_base = retry_backoff_base
        self.max_retries = max_retries
        self.timeout = timeout
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub = stub

    def _get_stub(self) -> retrieval_pb2_grpc.RetrievalServiceStub:
        if self._stub is not None:
            return self._stub
        if self._channel is None:
            options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.http2.max_pings_without_data", 0),
            ]
            self._channel = grpc.aio.insecure_channel(self.target_url, options=options)
        return retrieval_pb2_grpc.RetrievalServiceStub(self._channel)

    async def index_batch(
        self,
        points: list[VectorizedPoint],
        wait: bool = False,
    ) -> retrieval_pb2.IndexBatchResponse:
        """Transmit a micro-batch of points to RetrievalService.IndexBatch."""
        proto_points = [
            retrieval_pb2.PointData(
                point_id=p.point_id,
                doc_id=p.doc_id,
                chunk_index=p.chunk_index,
                content=p.content,
                dense_vector=p.dense_vector,
                role_tags=p.role_tags,
                redacted=p.redacted,
                source_url=p.source_url,
                token_count=p.token_count,
            )
            for p in points
        ]
        request = retrieval_pb2.IndexBatchRequest(points=proto_points, wait=wait)
        stub = self._get_stub()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await stub.IndexBatch(request, timeout=self.timeout)
                return response
            except grpc.RpcError as e:
                last_error = e
                status_code = e.code() if hasattr(e, "code") else None
                if status_code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    backoff = self.retry_backoff_base * (2**attempt)
                    logger.warning(
                        "gRPC IndexBatch transient failure (attempt %d/%d): %s. Backoff %.2fs",
                        attempt + 1,
                        self.max_retries,
                        e,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("gRPC IndexBatch failed with unknown error state")

    async def close(self) -> None:
        """Close gRPC channel connection."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
