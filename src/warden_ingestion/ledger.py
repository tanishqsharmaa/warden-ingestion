"""High-concurrency SQLite Write-Ahead Logging (WAL) ledger for document idempotency and DLQ."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import aiosqlite

from warden_ingestion.models import DocumentRecord, IngestionRunRecord, PoisonPillRecord


class IngestionLedger:
    """Manages SQLite WAL persistence for document status, idempotency, and poison pills."""

    def __init__(self, db_path: str = "/data/ingestion.db") -> None:
        self.db_path = db_path
        self._ensure_parent_directory()

    def _ensure_parent_directory(self) -> None:
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")
            await db.execute("PRAGMA busy_timeout = 5000;")
            await db.execute("PRAGMA mmap_size = 268435456;")
            await db.execute("PRAGMA cache_size = -64000;")
            yield db

    async def initialize(self) -> None:
        """Initialize database schema, tables, and indexes."""
        async with self._get_connection() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY NOT NULL,
                    source_url TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    role_tags TEXT NOT NULL,
                    raw_character_count INTEGER NOT NULL,
                    scrubbed_character_count INTEGER NOT NULL,
                    redaction_hits_json TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(
                        status IN ('PENDING', 'PROCESSING', 'INDEXED', 'FAILED', 'POISON_PILL')
                    ),
                    error_message TEXT,
                    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);"
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    manifest_name TEXT NOT NULL,
                    total_documents INTEGER NOT NULL,
                    processed_documents INTEGER NOT NULL,
                    skipped_documents INTEGER NOT NULL,
                    failed_documents INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP NOT NULL
                );
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS poison_pills (
                    id TEXT PRIMARY KEY NOT NULL,
                    document_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    failure_stage TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    stack_trace TEXT NOT NULL,
                    quarantined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_poison_pills_hash ON poison_pills(content_hash);"
            )
            await db.commit()

    async def get_document_by_hash(self, content_hash: str) -> Optional[DocumentRecord]:
        """Look up document by its SHA-256 content hash."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE content_hash = ? LIMIT 1;", (content_hash,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return self._row_to_document(row)

    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Look up document by primary document ID."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE id = ? LIMIT 1;", (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return self._row_to_document(row)

    async def record_document(self, doc: DocumentRecord) -> None:
        """Insert or replace document record in ledger."""
        async with self._get_connection() as db:
            await db.execute(
                """
                INSERT INTO documents (
                    id, source_url, content_hash, title, role_tags,
                    raw_character_count, scrubbed_character_count,
                    redaction_hits_json, chunk_count, status, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    error_message=excluded.error_message,
                    updated_at=CURRENT_TIMESTAMP;
                """,
                (
                    doc.id,
                    doc.source_url,
                    doc.content_hash,
                    doc.title,
                    json.dumps(doc.role_tags),
                    doc.raw_character_count,
                    doc.scrubbed_character_count,
                    doc.redaction_hits_json,
                    doc.chunk_count,
                    doc.status,
                    doc.error_message,
                ),
            )
            await db.commit()

    async def update_document_status(
        self, doc_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Update processing status and optional error message for document."""
        async with self._get_connection() as db:
            await db.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (status, error_message, doc_id),
            )
            await db.commit()

    async def record_poison_pill(self, pill: PoisonPillRecord) -> None:
        """Quarantine a corrupted or repeatedly failing document to the DLQ."""
        async with self._get_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO poison_pills (
                    id, document_id, source_url, content_hash, failure_stage,
                    retry_count, stack_trace, quarantined_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (
                    pill.id,
                    pill.document_id,
                    pill.source_url,
                    pill.content_hash,
                    pill.failure_stage,
                    pill.retry_count,
                    pill.stack_trace,
                ),
            )
            await db.commit()

    async def get_poison_pills(self) -> list[PoisonPillRecord]:
        """Fetch all quarantined poison pills."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM poison_pills ORDER BY quarantined_at DESC;"
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [
                    PoisonPillRecord(
                        id=row["id"],
                        document_id=row["document_id"],
                        source_url=row["source_url"],
                        content_hash=row["content_hash"],
                        failure_stage=row["failure_stage"],
                        retry_count=row["retry_count"],
                        stack_trace=row["stack_trace"],
                        quarantined_at=str(row["quarantined_at"]),
                    )
                    for row in rows
                ]

    async def record_run_start(self, run_id: str, manifest_name: str, total_docs: int) -> None:
        """Record the beginning of a batch ingestion run."""
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._get_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO ingestion_runs (
                    run_id, manifest_name, total_documents, processed_documents,
                    skipped_documents, failed_documents, duration_seconds,
                    started_at, completed_at
                ) VALUES (?, ?, ?, 0, 0, 0, 0.0, ?, ?);
                """,
                (run_id, manifest_name, total_docs, now_iso, now_iso),
            )
            await db.commit()

    async def record_run_completion(
        self,
        run_id: str,
        processed: int,
        skipped: int,
        failed: int,
        duration_seconds: float,
    ) -> None:
        """Finalize an ingestion run record with completed metrics."""
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self._get_connection() as db:
            await db.execute(
                """
                UPDATE ingestion_runs
                SET processed_documents = ?,
                    skipped_documents = ?,
                    failed_documents = ?,
                    duration_seconds = ?,
                    completed_at = ?
                WHERE run_id = ?;
                """,
                (processed, skipped, failed, duration_seconds, now_iso, run_id),
            )
            await db.commit()

    async def get_run(self, run_id: str) -> Optional[IngestionRunRecord]:
        """Fetch metadata for an ingestion run by run_id."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ingestion_runs WHERE run_id = ? LIMIT 1;", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return IngestionRunRecord(
                    run_id=row["run_id"],
                    manifest_name=row["manifest_name"],
                    total_documents=row["total_documents"],
                    processed_documents=row["processed_documents"],
                    skipped_documents=row["skipped_documents"],
                    failed_documents=row["failed_documents"],
                    duration_seconds=row["duration_seconds"],
                    started_at=str(row["started_at"]),
                    completed_at=str(row["completed_at"]),
                )

    def _row_to_document(self, row: aiosqlite.Row) -> DocumentRecord:
        role_tags_raw = row["role_tags"]
        try:
            if isinstance(role_tags_raw, str):
                role_tags = json.loads(role_tags_raw)
            else:
                role_tags = list(role_tags_raw)
        except Exception:
            role_tags = []

        return DocumentRecord(
            id=row["id"],
            source_url=row["source_url"],
            content_hash=row["content_hash"],
            title=row["title"],
            role_tags=role_tags,
            raw_character_count=row["raw_character_count"],
            scrubbed_character_count=row["scrubbed_character_count"],
            redaction_hits_json=row["redaction_hits_json"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            error_message=row["error_message"],
            ingested_at=str(row["ingested_at"]),
            updated_at=str(row["updated_at"]),
        )
