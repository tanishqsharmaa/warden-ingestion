"""Data models and entity records for warden-ingestion."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentRecord:
    """Document metadata and status ledger record."""

    id: str
    source_url: str
    content_hash: str
    title: str
    role_tags: list[str]
    raw_character_count: int
    scrubbed_character_count: int
    redaction_hits_json: str
    chunk_count: int
    status: str  # PENDING, PROCESSING, INDEXED, FAILED, POISON_PILL
    error_message: Optional[str] = None
    ingested_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class IngestionRunRecord:
    """Batch ingestion execution metrics record."""

    run_id: str
    manifest_name: str
    total_documents: int
    processed_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    duration_seconds: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class PoisonPillRecord:
    """Dead-letter queue (DLQ) entry for corrupt documents."""

    id: str
    document_id: str
    source_url: str
    content_hash: str
    failure_stage: str  # PRESIDIO_SCRUB, CHUNKING, EMBEDDING, INDEXING
    stack_trace: str
    retry_count: int = 0
    quarantined_at: Optional[str] = None


@dataclass
class ChunkPayload:
    """Single extracted text chunk prior to vectorization."""

    doc_id: str
    chunk_index: int
    content: str
    role_tags: list[str]
    source_url: str
    token_count: int
    redacted: bool


@dataclass
class VectorizedPoint:
    """Vectorized chunk ready for gRPC indexing hand-off."""

    point_id: str
    doc_id: str
    chunk_index: int
    content: str
    dense_vector: list[float]  # 768-dim float32
    role_tags: list[str]
    redacted: bool
    source_url: str
    token_count: int


@dataclass
class ScrubbedResult:
    """Output from Presidio PII sanitization."""

    text: str
    redaction_counts: dict[str, int]
    character_count_delta: int
