import pytest
from warden_ingestion.config import Settings

def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
    monkeypatch.delenv("RETRIEVAL_GRPC_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    settings = Settings()
    assert settings.SQLITE_DB_PATH == "/data/ingestion.db"
    assert settings.RETRIEVAL_GRPC_URL == "warden-retrieval:50051"
    assert settings.REDIS_URL == "redis://warden-cache-redis:6379"
    assert settings.EMBEDDING_BATCH_SIZE == 64
    assert settings.MAX_QUEUE_BUFFER == 256
    assert settings.PRESIDIO_PROCESS_WORKERS == 4
    assert settings.CHUNK_SIZE_TOKENS == 512
    assert settings.CHUNK_OVERLAP_TOKENS == 50
    assert settings.PRESIDIO_CONFIDENCE_THRESHOLD == 0.75
