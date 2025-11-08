"""Embedding generation and database persistence module.

Provides functionality for generating embeddings from document chunks and
persisting them to PostgreSQL with pgvector for semantic search.

Also provides sentence-transformers model loading with intelligent caching
for the all-mpnet-base-v2 embedding model.
"""

from src.embedding.database import ChunkInserter, InsertionStats
from src.embedding.model_loader import (
    ModelLoadError,
    ModelLoader,
    ModelValidationError,
)

__all__ = [
    "ChunkInserter",
    "InsertionStats",
    "ModelLoader",
    "ModelLoadError",
    "ModelValidationError",
]
