"""Configuration and environment variables management for warden-ingestion."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for document ingestion service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    WARDEN_ENV: str = "development"
    SQLITE_DB_PATH: str = "/data/ingestion.db"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-base-en-v1.5"
    ONNX_MODEL_PATH: str = "models/bge-base-en-v1.5-int8/model.onnx"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 64
    PRESIDIO_PROCESS_WORKERS: int = 4
    SPACY_MODEL_NAME: str = "en_core_web_sm"
    MAX_QUEUE_BUFFER: int = 256
    RETRIEVAL_GRPC_URL: str = "warden-retrieval:50051"
    REDIS_URL: str = "redis://warden-cache-redis:6379"
    PRESIDIO_CONFIDENCE_THRESHOLD: float = 0.75
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    LOG_LEVEL: str = "INFO"


settings = Settings()
