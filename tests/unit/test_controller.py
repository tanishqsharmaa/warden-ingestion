import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from warden_ingestion.controller import IngestionController
from warden_ingestion.events import CacheInvalidationPublisher
from warden_ingestion.models import ChunkPayload, DocumentRecord, ScrubbedResult, VectorizedPoint


@pytest.mark.asyncio
async def test_cache_invalidation_publisher():
    mock_redis = AsyncMock()
    publisher = CacheInvalidationPublisher(redis_client=mock_redis)
    await publisher.publish_invalidation(["Employee", "Manager"], "run-123")

    mock_redis.publish.assert_called_once()
    channel, message = mock_redis.publish.call_args[0]
    assert channel == "warden:cache:invalidate"
    payload = json.loads(message)
    assert payload["event"] == "INGESTION_COMPLETED"
    assert payload["run_id"] == "run-123"
    assert "Employee" in payload["affected_roles"]
    assert "Manager" in payload["affected_roles"]


@pytest.mark.asyncio
async def test_controller_skips_duplicate_document():
    mock_ledger = AsyncMock()
    mock_ledger.get_document_by_hash.return_value = DocumentRecord(
        id="DOC-001",
        source_url="url",
        content_hash="abc123hash",
        title="PTO",
        role_tags=["Employee"],
        raw_character_count=100,
        scrubbed_character_count=100,
        redaction_hits_json="{}",
        chunk_count=1,
        status="INDEXED",
    )

    controller = IngestionController(
        ledger=mock_ledger,
        scrubber=MagicMock(),
        chunker=MagicMock(),
        embedder=MagicMock(),
        grpc_client=AsyncMock(),
        publisher=AsyncMock(),
    )

    manifest_data = [
        {
            "doc_id": "DOC-001",
            "title": "PTO",
            "role_tags": ["Employee"],
            "content": "Policy text",
            "source_url": "url",
        }
    ]

    summary = await controller.process_manifest_documents("run-1", manifest_data, force_reindex=False)
    assert summary["skipped_documents"] == 1
    assert summary["processed_documents"] == 0
    assert summary["failed_documents"] == 0


@pytest.mark.asyncio
async def test_controller_processes_valid_document():
    mock_ledger = AsyncMock()
    mock_ledger.get_document_by_hash.return_value = None

    mock_scrubber = MagicMock()
    mock_scrubber.scrub_text.return_value = ScrubbedResult(
        text="Scrubbed policy text",
        redaction_counts={"EMAIL_ADDRESS": 1},
        character_count_delta=5,
    )

    mock_chunker = MagicMock()
    mock_chunker.split_text.return_value = [
        ChunkPayload("DOC-001", 0, "Chunk 0", ["Employee"], "url", 10, True)
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = [
        VectorizedPoint(
            point_id="pid-1",
            doc_id="DOC-001",
            chunk_index=0,
            content="Chunk 0",
            dense_vector=[0.1] * 768,
            role_tags=["Employee"],
            redacted=True,
            source_url="url",
            token_count=10,
        )
    ]

    mock_grpc = AsyncMock()
    mock_publisher = AsyncMock()

    controller = IngestionController(
        ledger=mock_ledger,
        scrubber=mock_scrubber,
        chunker=mock_chunker,
        embedder=mock_embedder,
        grpc_client=mock_grpc,
        publisher=mock_publisher,
    )

    manifest_data = [
        {
            "doc_id": "DOC-001",
            "title": "PTO",
            "role_tags": ["Employee"],
            "content": "Raw policy text",
            "source_url": "url",
        }
    ]

    summary = await controller.process_manifest_documents("run-1", manifest_data, force_reindex=False)
    assert summary["processed_documents"] == 1
    assert summary["skipped_documents"] == 0
    assert summary["failed_documents"] == 0

    mock_ledger.record_document.assert_called_once()
    mock_grpc.index_batch.assert_called_once()
    mock_publisher.publish_invalidation.assert_called_once_with(["Employee"], "run-1")


@pytest.mark.asyncio
async def test_controller_poison_pill_quarantine():
    mock_ledger = AsyncMock()
    mock_ledger.get_document_by_hash.return_value = None

    mock_scrubber = MagicMock()
    # Scrubber raises exception
    mock_scrubber.scrub_text.side_effect = RuntimeError("Malformed encoding crash")

    controller = IngestionController(
        ledger=mock_ledger,
        scrubber=mock_scrubber,
        chunker=MagicMock(),
        embedder=MagicMock(),
        grpc_client=AsyncMock(),
        publisher=AsyncMock(),
    )

    manifest_data = [
        {
            "doc_id": "DOC-BAD",
            "title": "Bad Policy",
            "role_tags": ["Manager"],
            "content": "Corrupt text",
            "source_url": "bad_url",
        }
    ]

    summary = await controller.process_manifest_documents("run-1", manifest_data, force_reindex=False)
    assert summary["failed_documents"] == 1
    assert summary["processed_documents"] == 0
    mock_ledger.record_poison_pill.assert_called_once()
    pill = mock_ledger.record_poison_pill.call_args[0][0]
    assert pill.document_id == "DOC-BAD"
    assert "Malformed encoding crash" in pill.stack_trace
