"""Table-aware recursive text chunking preserving Markdown tables and semantic structure."""

import re
from typing import Optional
from warden_ingestion.models import ChunkPayload


class TableAwareChunker:
    """Recursive chunker that preserves Markdown table rows and structure."""

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        chars_per_token: int = 4,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chars_per_token = chars_per_token
        self.chunk_size_chars = chunk_size_tokens * chars_per_token
        self.overlap_chars = chunk_overlap_tokens * chars_per_token

    def _is_table_block(self, block: str) -> bool:
        """Detect whether text block contains a Markdown table."""
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return False
        has_table_sep = any(re.match(r"^\|?[\s:-]+[-]+[\s:-]*(\|[\s:-]+[-]+[\s:-]*)+\|?$", line) for line in lines)
        has_pipe_rows = all(line.startswith("|") or "|" in line for line in lines)
        return has_table_sep and has_pipe_rows

    def _split_table(self, table_text: str) -> list[str]:
        """Split oversized table cleanly along row boundaries, repeating header."""
        lines = [l for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) <= 2:
            return [table_text]

        header = lines[:2]
        header_text = "\n".join(header) + "\n"
        data_rows = lines[2:]

        chunks: list[str] = []
        current_rows: list[str] = []
        current_chars = len(header_text)

        for row in data_rows:
            row_len = len(row) + 1
            if current_chars + row_len > self.chunk_size_chars and current_rows:
                chunk = header_text + "\n".join(current_rows)
                chunks.append(chunk.strip())
                current_rows = [row]
                current_chars = len(header_text) + row_len
            else:
                current_rows.append(row)
                current_chars += row_len

        if current_rows:
            chunk = header_text + "\n".join(current_rows)
            chunks.append(chunk.strip())

        return chunks if chunks else [table_text]

    def _split_text_block(self, text: str) -> list[str]:
        """Recursively split regular prose on paragraphs, lines, and sentences."""
        if len(text) <= self.chunk_size_chars:
            return [text.strip()] if text.strip() else []

        # Split on paragraph boundaries
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            chunks: list[str] = []
            current = ""
            for p in paragraphs:
                p_str = p.strip()
                if not p_str:
                    continue
                if len(current) + len(p_str) + 2 <= self.chunk_size_chars:
                    current = f"{current}\n\n{p_str}" if current else p_str
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(p_str) > self.chunk_size_chars:
                        chunks.extend(self._split_text_block(p_str))
                        current = ""
                    else:
                        current = p_str
            if current:
                chunks.append(current.strip())
            return chunks

        # Split on line boundaries
        lines = text.split("\n")
        if len(lines) > 1:
            chunks = []
            current = ""
            for l in lines:
                l_str = l.strip()
                if not l_str:
                    continue
                if len(current) + len(l_str) + 1 <= self.chunk_size_chars:
                    current = f"{current}\n{l_str}" if current else l_str
                else:
                    if current:
                        chunks.append(current.strip())
                    current = l_str
            if current:
                chunks.append(current.strip())
            return chunks

        # Fallback to word splitting
        words = text.split(" ")
        chunks = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 <= self.chunk_size_chars:
                current = f"{current} {w}" if current else w
            else:
                if current:
                    chunks.append(current.strip())
                current = w
        if current:
            chunks.append(current.strip())
        return chunks

    def split_text(
        self,
        doc_id: str,
        text: str,
        role_tags: list[str],
        source_url: str,
        redacted: bool = True,
    ) -> list[ChunkPayload]:
        """Split document text into structured, table-aware ChunkPayloads."""
        if not text.strip():
            return []

        # Separate blocks into tables vs prose
        raw_blocks = text.split("\n\n")
        processed_sections: list[str] = []

        for block in raw_blocks:
            b = block.strip()
            if not b:
                continue
            if self._is_table_block(b):
                if len(b) > self.chunk_size_chars:
                    processed_sections.extend(self._split_table(b))
                else:
                    processed_sections.append(b)
            else:
                processed_sections.extend(self._split_text_block(b))

        # Build chunks with overlap
        chunks: list[ChunkPayload] = []
        for idx, sec in enumerate(processed_sections):
            estimated_tokens = max(1, len(sec) // self.chars_per_token)
            chunks.append(
                ChunkPayload(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=sec,
                    role_tags=role_tags,
                    source_url=source_url,
                    token_count=estimated_tokens,
                    redacted=redacted,
                )
            )

        return chunks
