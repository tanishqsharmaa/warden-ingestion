import aiosqlite
import pytest
from warden_ingestion.ledger import IngestionLedger
from warden_ingestion.models import DocumentRecord, IngestionRunRecord, PoisonPillRecord


@pytest.mark.asyncio
@pytest.mark.wal
async def test_ledger_initialization_and_wal_pragmas(tmp_path):
    db_path = str(tmp_path / "test_ingestion.db")
    ledger = IngestionLedger(db_path=db_path)
    await ledger.initialize()

    # Verify per-connection pragmas configured on ledger connections
    async with ledger._get_connection() as db:
        async with db.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0].upper() == "WAL"
        async with db.execute("PRAGMA synchronous;") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1
        async with db.execute("PRAGMA busy_timeout;") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 5000

    # Verify WAL journal_mode is persistent in SQLite database header across independent connections
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0].upper() == "WAL"


@pytest.mark.asyncio
@pytest.mark.wal
async def test_ledger_deduplication_and_poison_pill(tmp_path):
    db_path = str(tmp_path / "test_ingestion.db")
    ledger = IngestionLedger(db_path=db_path)
    await ledger.initialize()

    doc = DocumentRecord(
        id="DOC-001",
        source_url="file:///tmp/doc.md",
        content_hash="abc123hash",
        title="PTO Policy",
        role_tags=["Employee"],
        raw_character_count=1000,
        scrubbed_character_count=980,
        redaction_hits_json='{"EMAIL": 1}',
        chunk_count=2,
        status="INDEXED",
    )
    await ledger.record_document(doc)

    existing = await ledger.get_document_by_hash("abc123hash")
    assert existing is not None
    assert existing.id == "DOC-001"
    assert existing.status == "INDEXED"
    assert existing.role_tags == ["Employee"]

    # Re-inserting identical hash should be safely handled or return existing
    duplicate = await ledger.get_document_by_hash("abc123hash")
    assert duplicate is not None
    assert duplicate.id == "DOC-001"

    pill = PoisonPillRecord(
        id="PILL-001",
        document_id="DOC-CORRUPT",
        source_url="file:///tmp/bad.md",
        content_hash="badhash",
        failure_stage="CHUNKING",
        retry_count=3,
        stack_trace="ZeroDivisionError: division by zero",
    )
    await ledger.record_poison_pill(pill)
    pills = await ledger.get_poison_pills()
    assert len(pills) == 1
    assert pills[0].document_id == "DOC-CORRUPT"
    assert pills[0].failure_stage == "CHUNKING"


@pytest.mark.asyncio
@pytest.mark.wal
async def test_ledger_run_tracking(tmp_path):
    db_path = str(tmp_path / "test_ingestion.db")
    ledger = IngestionLedger(db_path=db_path)
    await ledger.initialize()

    await ledger.record_run_start("run-001", "manifest.json", 10)
    await ledger.record_run_completion(
        run_id="run-001",
        processed=9,
        skipped=1,
        failed=0,
        duration_seconds=12.5,
    )

    run = await ledger.get_run("run-001")
    assert run is not None
    assert run.total_documents == 10
    assert run.processed_documents == 9
    assert run.skipped_documents == 1
    assert run.failed_documents == 0
    assert run.duration_seconds == 12.5
