# Project Warden: Document Ingestion Subsystem (`warden-ingestion`)

Autonomous Enterprise Single Source of Truth (SSOT) — Document Ingestion, PII Sanitization & Preprocessing Subsystem (Tier 2).

---

## 1. Overview

`warden-ingestion` is the Tier 2 microservice and Kubernetes batch job for **Project Warden**. It executes asynchronous, idempotent, backpressure-governed batch transformations of raw HR policy documents into PII-sanitized, table-aware text chunks and dense 768-dimensional embeddings.

### Core Architectural Invariants:
1. **MANDATE-01 (Database-Per-Service Isolation)**: Exclusively owns its private SQLite WAL idempotency ledger (`ingestion.db`). Initiates zero direct network connections to Qdrant.
2. **MANDATE-02 (Explicit Internal gRPC Data Plane)**: Transmits vectorized points via streaming HTTP/2 gRPC (`RetrievalService.IndexBatch`) to `warden-retrieval:50051`.
3. **MANDATE-03 (Early-Binding ACLs & Sanitization)**: Tags point payloads with `role_tags` and enforces 100% PII redaction (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`) using Microsoft Presidio before indexing.
4. **Bounded Backpressure & Memory Envelope**: Governed by an `asyncio.Queue(maxsize=256)` pausing chunk generation at 256 items and resuming below 128 items, strictly guaranteeing pod memory stays $< 1.2\text{GB}$ RSS.
5. **Dead-Letter Queue (DLQ) Resilience**: Failing or corrupt documents are retried 3 times with exponential backoff and quarantined to the `poison_pills` table without terminating batch processing.

---

## 2. Directory Structure

```
warden-ingestion/
├── src/
│   └── warden_ingestion/
│       ├── __init__.py          # Package metadata & exports
│       ├── api.py               # FastAPI REST endpoints & RFC 7807 error handlers
│       ├── chunker.py           # Table-aware recursive character chunker
│       ├── config.py            # Typed Settings & environment validation
│       ├── controller.py        # End-to-end ingestion orchestrator & DLQ handler
│       ├── embedder.py          # Batched INT8 ONNX vectorizer & UUIDv5 point IDs
│       ├── events.py            # Redis Pub/Sub cache invalidation publisher
│       ├── grpc_client.py       # Streaming gRPC client for RetrievalService.IndexBatch
│       ├── ledger.py            # SQLite WAL ledger (documents, runs, poison_pills)
│       ├── main.py              # Uvicorn application entrypoint
│       ├── models.py            # Typed dataclasses & entity schemas
│       └── queue.py             # Bounded backpressure chunk staging buffer
├── scripts/
│   └── download_onnx_model.py   # CLI tool to prepare and quantize ONNX models
├── docker/
│   └── Dockerfile.ingestion     # Production non-root (10001:10001) container
├── tests/
│   ├── fixtures/
│   │   ├── synthetic_pii_corpus.json          # 100% recall PII verification fixture
│   │   └── sample_20_policies_manifest.json   # 20-doc batch manifest with poison pill
│   ├── unit/                    # Fast isolated unit tests
│   └── integration/             # End-to-end integration and DLQ tests
└── pyproject.toml               # Package definition and build configuration
```

---

## 3. Quickstart & Verification

```bash
# 1. Initialize Python 3.12 virtual environment
uv venv .venv --python 3.12
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# 2. Install package in editable mode with development dependencies
uv pip install -e ".[dev]" -e "../warden-shared"

# 3. Execute unit and integration test suite
pytest tests/ -v

# 4. Verify Master Build Sequence GATE-2
pytest tests/ -m "pii or wal or integration" -v
```

---

## 4. API Endpoints

- `GET /health`: Health and readiness probe returning DB connection, queue depth, and worker status.
- `POST /ingest/run`: Accepts `manifest_url`, `force_reindex`, `chunk_size_tokens`, and triggers asynchronous batch processing (returns HTTP 202 Accepted).
- `GET /ingest/status/{run_id}`: Returns real-time metrics (processed, skipped, failed documents, duration, PII hit counts).

---

## 5. Master Build Sequence Integration Gate: GATE-2

Completion of `warden-ingestion` satisfies **GATE-2** of the Master Build Sequence (`Docs/BUILD_SEQUENCE.md`), unblocking Tier 3 (`warden-retrieval`).
