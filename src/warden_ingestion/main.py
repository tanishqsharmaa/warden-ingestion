"""Application entrypoint for running warden-ingestion under Uvicorn."""

import logging

from warden_ingestion.api import create_app
from warden_ingestion.chunker import TableAwareChunker
from warden_ingestion.config import settings
from warden_ingestion.controller import IngestionController
from warden_ingestion.embedder import OnnxEmbedder
from warden_ingestion.events import CacheInvalidationPublisher
from warden_ingestion.grpc_client import RetrievalGrpcClient
from warden_ingestion.ledger import IngestionLedger
from warden_ingestion.pii import PresidioScrubberPool
from warden_ingestion.queue import BoundedChunkQueue

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("warden.ingestion")

ledger = IngestionLedger(db_path=settings.SQLITE_DB_PATH)
scrubber = PresidioScrubberPool(
    workers=settings.PRESIDIO_PROCESS_WORKERS,
    threshold=settings.PRESIDIO_CONFIDENCE_THRESHOLD,
)
chunker = TableAwareChunker(
    chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
)
embedder = OnnxEmbedder(
    batch_size=settings.EMBEDDING_BATCH_SIZE,
)
grpc_client = RetrievalGrpcClient(
    target_url=settings.RETRIEVAL_GRPC_URL,
)
publisher = CacheInvalidationPublisher(
    redis_url=settings.REDIS_URL,
)
queue = BoundedChunkQueue(
    maxsize=settings.MAX_QUEUE_BUFFER,
    low_watermark=settings.MAX_QUEUE_BUFFER // 2,
)

controller = IngestionController(
    ledger=ledger,
    scrubber=scrubber,
    chunker=chunker,
    embedder=embedder,
    grpc_client=grpc_client,
    publisher=publisher,
    queue=queue,
)

app = create_app(ledger=ledger, controller=controller)
