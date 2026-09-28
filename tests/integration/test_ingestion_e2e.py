import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from warden_ingestion.chunker import TableAwareChunker
from warden_ingestion.controller import IngestionController
from warden_ingestion.embedder import OnnxEmbedder
from warden_ingestion.events import CacheInvalidationPublisher
from warden_ingestion.grpc_client import RetrievalGrpcClient
from warden_ingestion.ledger import IngestionLedger
from warden_ingestion.models import ChunkPayload
from warden_ingestion.pii import PresidioScrubberPool
from warden_ingestion.queue import BoundedChunkQueue
from warden_shared.proto.v1 import retrieval_pb2


class FaultInjectingChunker(TableAwareChunker):
    """Chunker that injects a fatal parsing failure for poison-pill documents."""

    def split_text(
        self,
        doc_id: str,
        text: str,
        role_tags: list[str],
        source_url: str,
        redacted: bool = True,
    ) -> list[ChunkPayload]:
        if "CRITICAL_TRIGGER_POISON_PILL" in text:
            raise ValueError(f"Fatal unparseable poison-pill encoding in document {doc_id}")
        return super().split_text(doc_id, text, role_tags, source_url, redacted)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_batch_ingestion_with_poison_pill_and_deduplication(tmp_path):
    manifest_path = Path("tests/fixtures/sample_20_policies_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert len(manifest_data) == 20

    # 1. Initialize test components
    db_path = str(tmp_path / "e2e_ingestion.db")
    ledger = IngestionLedger(db_path=db_path)
    await ledger.initialize()

    scrubber = PresidioScrubberPool(workers=2, threshold=0.75)
    chunker = FaultInjectingChunker(chunk_size_tokens=512, chunk_overlap_tokens=50)

    # Mock ONNX session for deterministic 768-dim embeddings without downloading external weights
    class FastMockOnnxSession:
        def get_inputs(self):
            class InputMeta:
                def __init__(self, name):
                    self.name = name
            return [InputMeta("input_ids"), InputMeta("attention_mask")]

        def run(self, output_names, input_feed):
            batch_size = len(input_feed["input_ids"])
            seq_len = len(input_feed["input_ids"][0])
            return [np.ones((batch_size, seq_len, 768), dtype=np.float32)]

    class FastMockTokenizer:
        def encode_batch(self, texts):
            class Encoding:
                def __init__(self):
                    self.ids = [101] * 16
                    self.attention_mask = [1] * 16
            return [Encoding() for _ in texts]

    embedder = OnnxEmbedder(
        session=FastMockOnnxSession(),
        tokenizer=FastMockTokenizer(),
        batch_size=16,
    )

    # Mock gRPC retrieval client returning valid UPSERTED status
    mock_grpc_stub = MagicMock()
    mock_grpc_stub.IndexBatch = AsyncMock(
        return_value=retrieval_pb2.IndexBatchResponse(
            status="UPSERTED",
            points_count=1,
            execution_time_ms=1.5,
        )
    )
    grpc_client = RetrievalGrpcClient(target_url="localhost:50051", stub=mock_grpc_stub)

    mock_redis = AsyncMock()
    publisher = CacheInvalidationPublisher(redis_client=mock_redis)
    queue = BoundedChunkQueue(maxsize=256, low_watermark=128)

    controller = IngestionController(
        ledger=ledger,
        scrubber=scrubber,
        chunker=chunker,
        embedder=embedder,
        grpc_client=grpc_client,
        publisher=publisher,
        queue=queue,
    )

    try:
        # =========================================================================
        # PHASE 1: Full Batch Run (20 documents: 19 valid, 1 poison pill)
        # =========================================================================
        run_1_summary = await controller.process_manifest_documents(
            run_id="run-e2e-001",
            documents=manifest_data,
            force_reindex=False,
        )

        assert run_1_summary["processed_documents"] == 19
        assert run_1_summary["failed_documents"] == 1
        assert run_1_summary["skipped_documents"] == 0
        assert run_1_summary["total_chunks_created"] >= 19

        # Verify SQLite Ledger: 19 documents INDEXED, 1 POISON_PILL
        pills = await ledger.get_poison_pills()
        assert len(pills) == 1
        assert pills[0].document_id == "DOC-POISON-020"
        assert pills[0].failure_stage == "CHUNKING"
        assert "Fatal unparseable poison-pill encoding" in pills[0].stack_trace

        poison_doc = await ledger.get_document("DOC-POISON-020")
        assert poison_doc is not None
        assert poison_doc.status == "POISON_PILL"

        valid_doc = await ledger.get_document("DOC-HR-001")
        assert valid_doc is not None
        assert valid_doc.status == "INDEXED"

        # Verify Redis cache invalidation event emitted
        mock_redis.publish.assert_called_once()
        inv_channel, inv_msg = mock_redis.publish.call_args[0]
        assert inv_channel == "warden:cache:invalidate"
        inv_payload = json.loads(inv_msg)
        assert inv_payload["event"] == "INGESTION_COMPLETED"
        assert set(inv_payload["affected_roles"]).issuperset({"Employee", "Manager", "HR-Admin"})

        # =========================================================================
        # PHASE 2: Immediate Re-Ingestion (Assert 100% SHA-256 Deduplication < 1.0s)
        # =========================================================================
        t_start = time.perf_counter()
        run_2_summary = await controller.process_manifest_documents(
            run_id="run-e2e-002",
            documents=manifest_data,
            force_reindex=False,
        )
        t_elapsed = time.perf_counter() - t_start

        # All 19 previously indexed documents should be skipped via SHA-256 match
        assert run_2_summary["skipped_documents"] == 19
        assert run_2_summary["processed_documents"] == 0
        # Re-run takes less than 1.0s total wall clock
        assert t_elapsed < 1.0, f"Deduplication re-run took {t_elapsed:.4f}s, expected < 1.0s"

    finally:
        scrubber.shutdown()
        await grpc_client.close()
        await publisher.close()
