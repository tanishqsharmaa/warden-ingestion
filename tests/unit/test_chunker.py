import asyncio
import pytest
from warden_ingestion.chunker import TableAwareChunker
from warden_ingestion.models import ChunkPayload
from warden_ingestion.queue import BoundedChunkQueue


def test_markdown_table_preserved_intact():
    chunker = TableAwareChunker(chunk_size_tokens=512, chunk_overlap_tokens=50)
    markdown_with_table = (
        "# Leave Policy\n\n"
        "Here is the leave entitlement schedule:\n\n"
        "| Service Years | PTO Days | Roll-Over |\n"
        "|---|---|---|\n"
        "| 1 Year | 15 Days | 5 Days |\n"
        "| 2-5 Years | 20 Days | 7 Days |\n"
        "| 5+ Years | 25 Days | 10 Days |\n\n"
        "Employees must request leave 2 weeks in advance."
    )
    chunks = chunker.split_text("DOC-001", markdown_with_table, ["Employee"], "file:///doc.md", True)
    assert len(chunks) >= 1
    # Check that the entire table structure is present in a chunk without mid-row cut
    table_chunk = next(c for c in chunks if "| Service Years |" in c.content)
    assert "| 5+ Years | 25 Days | 10 Days |" in table_chunk.content
    assert table_chunk.doc_id == "DOC-001"
    assert table_chunk.role_tags == ["Employee"]
    assert table_chunk.redacted is True


def test_recursive_chunking_multi_page_text():
    chunker = TableAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)
    paragraphs = [f"Paragraph {i}: " + ("word " * 50) for i in range(10)]
    long_text = "\n\n".join(paragraphs)

    chunks = chunker.split_text("DOC-LONG", long_text, ["Employee", "Manager"], "file:///long.md", False)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.doc_id == "DOC-LONG"
        assert c.token_count > 0


@pytest.mark.asyncio
async def test_bounded_queue_backpressure_pause_and_drain():
    queue = BoundedChunkQueue(maxsize=10, low_watermark=5)

    # Push 10 items (fills to high watermark)
    for i in range(10):
        await queue.push(ChunkPayload("DOC", i, f"chunk {i}", ["Employee"], "url", 50, True))

    assert queue.qsize() == 10

    # Pushing 11th item should block until drained
    push_task = asyncio.create_task(
        queue.push(ChunkPayload("DOC", 10, "chunk 10", ["Employee"], "url", 50, True))
    )
    await asyncio.sleep(0.05)
    assert not push_task.done()  # Should be paused by backpressure

    # Drain 6 items (drops below low_watermark 5)
    batch = await queue.pop_batch(max_batch=6)
    assert len(batch) == 6
    await asyncio.sleep(0.05)
    assert push_task.done()  # Unblocked
    await push_task
    assert queue.qsize() == 5
