"""FastAPI REST application, health probes, and batch ingestion triggers."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from warden_shared.errors import register_error_handlers

from warden_ingestion.config import settings
from warden_ingestion.controller import IngestionController
from warden_ingestion.ledger import IngestionLedger


class IngestRunRequest(BaseModel):
    manifest_url: str = Field(..., description="URI or path to the document manifest")
    force_reindex: bool = Field(False, description="Bypass idempotency ledger if True")
    chunk_size_tokens: int = Field(512, description="Target chunk token size")
    chunk_overlap_tokens: int = Field(50, description="Chunk overlap token size")
    target_roles_default: list[str] = Field(default_factory=lambda: ["Employee"])


class IngestRunResponse(BaseModel):
    run_id: str
    status: str
    manifest_url: str
    started_at: str
    poll_url: str


def create_app(
    ledger: Optional[IngestionLedger] = None,
    controller: Optional[IngestionController] = None,
) -> FastAPI:
    """FastAPI application factory with lifecycle management."""
    _ledger = ledger or IngestionLedger(db_path=settings.SQLITE_DB_PATH)
    _controller = controller

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: initialize database tables and WAL pragmas
        if hasattr(_ledger, "initialize"):
            await _ledger.initialize()
        yield
        # Shutdown: release network connections and worker pools
        if _controller:
            if hasattr(_controller.grpc_client, "close"):
                await _controller.grpc_client.close()
            if hasattr(_controller.publisher, "close"):
                await _controller.publisher.close()
            if hasattr(_controller.scrubber, "shutdown"):
                _controller.scrubber.shutdown()

    app = FastAPI(
        title="Warden Ingestion Service",
        version="1.0.0",
        description="Batch document ingestion and PII sanitization microservice (Tier 2)",
        lifespan=lifespan,
    )

    # Register standardized RFC 7807 problem details exception handlers
    register_error_handlers(app)

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health and readiness probe."""
        db_ok = True
        try:
            if hasattr(_ledger, "_get_connection"):
                async with _ledger._get_connection() as db:
                    await db.execute("SELECT 1;")
        except Exception:
            db_ok = False

        queue_depth = 0
        if _controller and hasattr(_controller, "queue"):
            queue_depth = _controller.queue.qsize()

        return {
            "status": "HEALTHY" if db_ok else "DEGRADED",
            "service": "warden-ingestion",
            "database_connected": db_ok,
            "model_loaded": settings.EMBEDDING_MODEL_ID,
            "queue_depth": queue_depth,
            "presidio_workers_alive": settings.PRESIDIO_PROCESS_WORKERS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/ingest/run", status_code=202, response_model=IngestRunResponse)
    async def trigger_ingest_run(
        request: IngestRunRequest,
        background_tasks: BackgroundTasks,
    ) -> IngestRunResponse:
        """Trigger an asynchronous batch ingestion run with coordinated run_id."""
        run_id = f"ingest-run-{uuid.uuid4()}"
        started_at = datetime.now(timezone.utc).isoformat()

        if _controller:
            background_tasks.add_task(
                _controller.run_manifest,
                manifest_url=request.manifest_url,
                force_reindex=request.force_reindex,
                run_id=run_id,
                chunk_size_tokens=request.chunk_size_tokens,
                chunk_overlap_tokens=request.chunk_overlap_tokens,
                default_roles=request.target_roles_default,
            )

        return IngestRunResponse(
            run_id=run_id,
            status="PROCESSING",
            manifest_url=request.manifest_url,
            started_at=started_at,
            poll_url=f"/ingest/status/{run_id}",
        )

    @app.get("/ingest/status/{run_id}")
    async def get_ingest_status(run_id: str) -> dict[str, Any]:
        """Query real-time metrics and status for an ingestion run."""
        run = await _ledger.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Ingestion run '{run_id}' not found.")

        completed_docs = run.processed_documents + run.skipped_documents + run.failed_documents
        is_completed = completed_docs >= run.total_documents
        status = "COMPLETED" if is_completed and run.total_documents > 0 else "PROCESSING"

        return {
            "run_id": run.run_id,
            "status": status,
            "manifest_url": run.manifest_name,
            "metrics": {
                "total_documents": run.total_documents,
                "processed_documents": run.processed_documents,
                "skipped_documents": run.skipped_documents,
                "failed_documents": run.failed_documents,
                "duration_seconds": run.duration_seconds,
            },
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        }

    return app
