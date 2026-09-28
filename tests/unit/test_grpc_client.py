from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from warden_shared.proto.v1 import retrieval_pb2

from warden_ingestion.grpc_client import RetrievalGrpcClient
from warden_ingestion.models import VectorizedPoint


@pytest.mark.asyncio
async def test_index_batch_grpc_client_success():
    mock_stub = MagicMock()
    mock_response = retrieval_pb2.IndexBatchResponse(
        status="UPSERTED",
        points_count=1,
        execution_time_ms=4.5,
    )
    mock_stub.IndexBatch = AsyncMock(return_value=mock_response)

    client = RetrievalGrpcClient(target_url="localhost:50051", stub=mock_stub)
    points = [
        VectorizedPoint(
            point_id="pid-1",
            doc_id="doc-1",
            chunk_index=0,
            content="text 1",
            dense_vector=[0.1] * 768,
            role_tags=["Employee"],
            redacted=True,
            source_url="file:///doc.md",
            token_count=10,
        )
    ]

    resp = await client.index_batch(points, wait=False)
    assert resp.status == "UPSERTED"
    assert resp.points_count == 1
    mock_stub.IndexBatch.assert_called_once()
    req = mock_stub.IndexBatch.call_args[0][0]
    assert req.wait is False
    assert len(req.points) == 1
    assert req.points[0].point_id == "pid-1"
    assert req.points[0].doc_id == "doc-1"
    assert req.points[0].chunk_index == 0
    assert len(req.points[0].dense_vector) == 768


@pytest.mark.asyncio
async def test_index_batch_retry_on_unavailable():
    mock_stub = MagicMock()
    mock_response = retrieval_pb2.IndexBatchResponse(
        status="UPSERTED",
        points_count=1,
        execution_time_ms=2.1,
    )

    # First call raises UNAVAILABLE, second call succeeds
    error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Transient network error",
    )
    mock_stub.IndexBatch = AsyncMock(side_effect=[error, mock_response])

    client = RetrievalGrpcClient(
        target_url="localhost:50051", stub=mock_stub, retry_backoff_base=0.01
    )
    points = [
        VectorizedPoint(
            point_id="pid-1",
            doc_id="doc-1",
            chunk_index=0,
            content="test",
            dense_vector=[0.0] * 768,
            role_tags=["Employee"],
            redacted=True,
            source_url="url",
            token_count=5,
        )
    ]

    resp = await client.index_batch(points)
    assert resp.status == "UPSERTED"
    assert mock_stub.IndexBatch.call_count == 2
