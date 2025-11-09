"""Type stubs for the document chunking module.

Provides type definitions for chunking documents into overlapping token-based chunks
while preserving semantic boundaries.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

@dataclass
class ChunkMetadata:
    """Metadata for a chunk."""

    chunk_index: int
    start_token_pos: int
    end_token_pos: int
    sentence_count: int
    overlap_tokens: int

@dataclass
class Chunk:
    """Represents a chunk of text with token metadata."""

    text: str
    tokens: list[int]
    token_count: int
    start_pos: int
    end_pos: int
    metadata: ChunkMetadata

class ChunkerConfig(BaseModel):
    """Configuration for the chunking algorithm."""

    chunk_size: int
    overlap_tokens: int
    preserve_boundaries: bool
    min_chunk_size: int

class Chunker:
    """Splits text into overlapping chunks while preserving semantic boundaries."""

    config: ChunkerConfig

    def __init__(self, config: ChunkerConfig | None = None) -> None: ...
    def chunk_text(self, text: str, token_ids: list[int]) -> list[Chunk]: ...
    def _identify_sentences(self, text: str) -> list[tuple[int, int]]: ...
    def _find_sentence_boundaries(
        self, start_token: int, end_token: int, sentence_ranges: list[tuple[int, int]]
    ) -> tuple[int, int]: ...
