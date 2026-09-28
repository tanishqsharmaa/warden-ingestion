"""Pipeline controller orchestrating ledger checks, PII scrubbing, chunking, embedding, and hand-off."""

import hashlib
import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from warden_ingestion.chunker import TableAwareChunker
from warden_ingestion.embedder import OnnxEmbedder
from warden_ingestion.events import CacheInvalidationPublisher
from warden_ingestion.grpc_client import RetrievalGrpcClient
from warden_ingestion.ledger import IngestionLedger
from warden_ingestion.models import DocumentRecord, PoisonPillRecord
from warden_ingestion.pii import PresidioScrubberPool
from warden_ingestion.queue import BoundedChunkQueue

logger = logging.getLogger("warden.ingestion.controller")


class IngestionController:
    """Orchestrates end-to-end batch ingestion workflow with backpressure and DLQ handling."""

    def __init__(
        self,
        ledger: IngestionLedger,
        scrubber: PresidioScrubberPool,
        chunker: TableAwareChunker,
        embedder: OnnxEmbedder,
        grpc_client: RetrievalGrpcClient,
        publisher: CacheInvalidationPublisher,
        queue: Optional[BoundedChunkQueue] = None,
    ) -> None:
        self.ledger = ledger
        self.scrubber = scrubber
        self.chunker = chunker
        self.embedder = embedder
        self.grpc_client = grpc_client
        self.publisher = publisher
        self.queue = queue or BoundedChunkQueue(maxsize=256, low_watermark=128)

    async def process_manifest_documents(
        self,
        run_id: str,
        documents: list[dict[str, Any]],
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Process list of document dictionaries through the complete pipeline."""
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        total_chunks = 0
        affected_roles: set[str] = set()
        pii_hits: dict[str, int] = {}

        for doc_item in documents:
            doc_id = doc_item.get("doc_id") or doc_item.get("id") or str(uuid.uuid4())
            title = doc_item.get("title", "Untitled Document")
            content = doc_item.get("content", "")
            source_url = doc_item.get("source_url", "")
            role_tags = doc_item.get("role_tags") or ["Employee"]

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Idempotency check: skip identical document hash if already indexed
            if not force_reindex:
                existing = await self.ledger.get_document_by_hash(content_hash)
                if existing is not None and existing.status == "INDEXED":
                    logger.info("Document %s (%s) unchanged, skipping", doc_id, content_hash[:8])
                    skipped_count += 1
                    continue

            # Process document with retry loop (max 3 attempts before DLQ quarantine)
            success = False
            last_stage = "INIT"
            last_trace = ""

            for attempt in range(1, 4):
                try:
                    # Stage 1: PII Scrubbing
                    last_stage = "PRESIDIO_SCRUB"
                    scrub_res = self.scrubber.scrub_text(content)
                    for k, v in scrub_res.redaction_counts.items():
                        pii_hits[k] = pii_hits.get(k, 0) + v

                    # Stage 2: Chunking
                    last_stage = "CHUNKING"
                    chunks = self.chunker.split_text(
                        doc_id=doc_id,
                        text=scrub_res.text,
                        role_tags=role_tags,
                        source_url=source_url,
                        redacted=bool(scrub_res.redaction_counts),
                    )

                    # Stage 3: Embedding
                    last_stage = "EMBEDDING"
                    points = self.embedder.embed_chunks(chunks)

                    # Stage 4: gRPC Indexing Hand-off
                    last_stage = "INDEXING"
                    if points:
                        await self.grpc_client.index_batch(points, wait=False)

                    # Stage 5: Ledger status commit
                    doc_record = DocumentRecord(
                        id=doc_id,
                        source_url=source_url,
                        content_hash=content_hash,
                        title=title,
                        role_tags=role_tags,
                        raw_character_count=len(content),
                        scrubbed_character_count=len(scrub_res.text),
                        redaction_hits_json=json.dumps(scrub_res.redaction_counts),
                        chunk_count=len(chunks),
                        status="INDEXED",
                    )
                    await self.ledger.record_document(doc_record)

                    processed_count += 1
                    total_chunks += len(chunks)
                    affected_roles.update(role_tags)
                    success = True
                    break

                except Exception as e:
                    last_trace = traceback.format_exc()
                    logger.warning(
                        "Document %s failed at %s (attempt %d/3): %s",
                        doc_id,
                        last_stage,
                        attempt,
                        e,
                    )
                    if attempt < 3:
                        time.sleep(0.05 * (2**attempt))

            if not success:
                # Quarantined to poison_pills DLQ table and ledger
                failed_count += 1
                pill = PoisonPillRecord(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    source_url=source_url,
                    content_hash=content_hash,
                    failure_stage=last_stage,
                    retry_count=3,
                    stack_trace=last_trace,
                )
                await self.ledger.record_poison_pill(pill)

                failed_doc = DocumentRecord(
                    id=doc_id,
                    source_url=source_url,
                    content_hash=content_hash,
                    title=title,
                    role_tags=role_tags,
                    raw_character_count=len(content),
                    scrubbed_character_count=0,
                    redaction_hits_json="{}",
                    chunk_count=0,
                    status="POISON_PILL",
                    error_message=f"Failed at {last_stage} after 3 retries",
                )
                await self.ledger.record_document(failed_doc)

        # Broadcast cache invalidation if any document was successfully indexed
        if affected_roles:
            await self.publisher.publish_invalidation(sorted(list(affected_roles)), run_id)

        return {
            "run_id": run_id,
            "processed_documents": processed_count,
            "skipped_documents": skipped_count,
            "failed_documents": failed_count,
            "total_chunks_created": total_chunks,
            "pii_redaction_hits": pii_hits,
            "affected_roles": sorted(list(affected_roles)),
        }

    async def run_manifest(
        self,
        manifest_url: str,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Load manifest from file or URL and execute batch ingestion pipeline."""
        start_time = time.time()
        run_id = f"ingest-run-{uuid.uuid4()}"

        # Parse manifest payload
        documents: list[dict[str, Any]] = []
        if manifest_url.startswith("file://") or Path(manifest_url).exists():
            file_path = manifest_url.replace("file://", "")
            if not Path(file_path).exists() and manifest_url.startswith("file:///"):
                # Handle Windows file:/// path
                file_path = manifest_url[8:] if manifest_url[9:11] == ":/" else manifest_url[7:]
            raw_text = Path(file_path).read_text(encoding="utf-8")
            documents = json.loads(raw_text)
        elif manifest_url.startswith("http://") or manifest_url.startswith("https://"):
            async with httpx.AsyncClient() as client:
                resp = await client.get(manifest_url, timeout=30.0)
                resp.raise_for_status()
                documents = resp.json()
        else:
            raw_text = Path(manifest_url).read_text(encoding="utf-8")
            documents = json.loads(raw_text)

        await self.ledger.record_run_start(run_id, manifest_url, len(documents))

        summary = await self.process_manifest_documents(
            run_id=run_id,
            documents=documents,
            force_reindex=force_reindex,
        )

        duration = time.time() - start_time
        summary["duration_seconds"] = round(duration, 2)
        summary["total_documents"] = len(documents)

        await self.ledger.record_run_completion(
            run_id=run_id,
            processed=summary["processed_documents"],
            skipped=summary["skipped_documents"],
            failed=summary["failed_documents"],
            duration_seconds=duration,
        )

        return summary
