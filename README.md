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
│       ├── main.py              # Uvicorn application entrypoint & lifespan manager
│       ├── models.py            # Typed dataclasses & entity schemas
│       └── queue.py             # Bounded backpressure chunk staging buffer
├── scripts/
│   └── download_onnx_model.py   # CLI tool to download, prepare and quantize ONNX models
├── docker/
│   └── Dockerfile.ingestion     # Production non-root (10001:10001) container
├── tests/
│   ├── fixtures/
│   │   ├── synthetic_pii_corpus.json          # 100% recall PII verification fixture
│   │   └── sample_20_policies_manifest.json   # 20-doc batch manifest with poison pill
│   ├── unit/                    # Fast isolated unit tests
│   └── integration/             # End-to-end integration and DLQ tests
├── pyproject.toml               # Package definition and build configuration
└── README.md                    # Subsystem documentation & session handoff
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

# 5. Execute static analysis and linting
ruff check src/ tests/ scripts/
```

---

## 4. API Endpoints

- `GET /health`: Health and readiness probe returning DB connection, queue depth, and worker status.
- `POST /ingest/run`: Accepts `manifest_url`, `force_reindex`, `chunk_size_tokens`, `chunk_overlap_tokens`, and triggers asynchronous batch processing (returns HTTP 202 Accepted with poll URL).
- `GET /ingest/status/{run_id}`: Returns real-time metrics (processed, skipped, failed documents, duration, PII hit counts).

---

## 5. Master Build Sequence Integration Gate: GATE-2

Completion of `warden-ingestion` satisfies **GATE-2** of the Master Build Sequence (`Docs/BUILD_SEQUENCE.md`), unblocking Tier 3 (`warden-retrieval`).

---

## 6. Session Handoff & Platform Engineering Context

### 6.1 Status & Delivery State
- **Tier Classification**: Tier 2 (`warden-ingestion`) — **100% COMPLETE & PRODUCTION HARDENED**.
- **Test Suite**: 24 passed (0 failures, 0 skipped) across unit and integration suites in 7.04s.
- **Static Analysis**: Zero lint errors (`ruff check src/ tests/ scripts/`).
- **Code Review**: Independent Senior Code Review completed; all 5 Critical and 6 Important findings resolved and verified.
- **Build & Packaging**:
  - Reproducible OCI container image definition (`docker/Dockerfile.ingestion`) with non-root security (`USER 10001:10001`) and automated health checks.
  - Python virtual environment `.venv` configured on Python 3.12.13.
- **Workspace Hygiene**: Working tree clean, `.gitignore` protects non-program files, databases, model weights, and build caches.

### 6.2 Key Architectural Invariants & Hardened Mechanisms
1. **MANDATE-01 (Database-Per-Service Isolation)**: `warden-ingestion` exclusively owns its SQLite WAL ledger (`ingestion.db`). It does not open connections to Qdrant directly.
2. **MANDATE-02 (gRPC Internal Data Plane)**: Points are streamed to `warden-retrieval:50051` via `RetrievalService.IndexBatch(wait=False)` with 3-attempt exponential backoff retries.
3. **MANDATE-03 (Early-Binding ACLs & PII Sanitization)**: Every chunk is tagged with its origin `role_tags` and sanitized of PII using Microsoft Presidio and high-precision pattern recognizers.
4. **Watermark Backpressure**: Bounded `asyncio.Queue(maxsize=256)` decouples document chunking from vectorization, pausing chunk producers at 256 items and resuming below 128 items to strictly contain RSS memory $< 1.2\text{GB}$.
5. **Thundering Herd Elimination**: On batch completion, invalidation events are broadcast to Redis topic `warden:cache:invalidate` containing the affected `role_tags`.
6. **Dead-Letter Queue (DLQ)**: Malformed or crashing documents are retried 3 times and quarantined to `poison_pills` with status `POISON_PILL` and stack traces without crashing the batch.
7. **Idempotency**: Documents are hashed with SHA-256; re-ingested identical documents are bypassed in $<1.0\text{s}$ total wall clock.

### 6.3 Immediate Next Steps (Tier 3: Storage Custodian & Hybrid Retrieval Subsystem)
Per `Docs/BUILD_SEQUENCE.md`, the next phase of Project Warden is **Tier 3 (`warden-retrieval`)**:
1. **Repository Target**: `F:\RAG\project_1\warden-retrieval`
2. **Core Responsibilities**:
   - Sole custodian of the Qdrant StatefulSet cluster.
   - Initialize Qdrant collection `warden_hr_policies` with 768-dim INT8 scalar quantization and FastEmbed BM25 sparse vectors.
   - Implement gRPC server exposing `RetrievalService` (`IndexBatch` and `Retrieve` RPCs) on port 50051.
   - Enforce early-binding pre-filtering on `role_tags` directly inside Qdrant HNSW graph traversal.
   - Implement native Reciprocal Rank Fusion (RRF) combining dense and sparse search rankings.
   - Implement dynamic candidate pruning (`score >= 0.40 * max_score`, capped at $\le 10$ chunks) before speculative reranking.
   - Implement speculative gRPC client invoking `warden-laya-service:50051` with 150ms deadline and raw RRF fallback.
3. **Integration Gate**: **GATE-3** (`pytest tests/integration/test_retrieval_acl.py -m "acl_isolation and qdrant"`).

### 6.4 Verification Quickstart for Incoming Engineers
```bash
# 1. Run complete automated test suite
.venv\Scripts\pytest.exe tests/ -v

# 2. Run static analysis
.venv\Scripts\ruff.exe check src/ tests/ scripts/

# 3. Test Dockerfile build syntax
.venv\Scripts\pytest.exe tests/unit/test_dockerfile.py -v
```
