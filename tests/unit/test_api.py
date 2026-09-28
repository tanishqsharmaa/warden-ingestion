from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient
import pytest

from warden_ingestion.api import create_app
from warden_ingestion.models import IngestionRunRecord


@pytest.mark.asyncio
async def test_health_probe():
    mock_ledger = AsyncMock()
    mock_ledger._get_connection = MagicMock()

    app = create_app(ledger=mock_ledger)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert data["service"] == "warden-ingestion"
        assert data["database_connected"] is True
        assert "presidio_workers_alive" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_trigger_ingest_run():
    mock_ledger = AsyncMock()
    mock_controller = AsyncMock()
    mock_controller.run_manifest.return_value = {
        "run_id": "run-test-123",
        "processed_documents": 5,
        "skipped_documents": 0,
        "failed_documents": 0,
    }

    app = create_app(ledger=mock_ledger, controller=mock_controller)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "manifest_url": "file:///data/manifests/test.json",
            "force_reindex": False,
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
            "target_roles_default": ["Employee"],
        }
        resp = await client.post("/ingest/run", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "PROCESSING"
        assert "run_id" in data
        assert "poll_url" in data
        assert data["poll_url"].endswith(data["run_id"])


@pytest.mark.asyncio
async def test_poll_ingest_status():
    mock_ledger = AsyncMock()
    mock_ledger.get_run.return_value = IngestionRunRecord(
        run_id="run-test-123",
        manifest_name="test.json",
        total_documents=10,
        processed_documents=9,
        skipped_documents=1,
        failed_documents=0,
        duration_seconds=15.4,
        started_at="2026-09-29T10:00:00Z",
        completed_at="2026-09-29T10:00:15Z",
    )

    app = create_app(ledger=mock_ledger)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ingest/status/run-test-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "run-test-123"
        assert data["status"] == "COMPLETED"
        assert data["metrics"]["total_documents"] == 10
        assert data["metrics"]["processed_documents"] == 9
        assert data["metrics"]["skipped_documents"] == 1
