"""Embedding generation and database persistence module.

Provides functionality for generating embeddings from document chunks and
persisting them to PostgreSQL with pgvector for semantic search.

Also provides sentence-transformers model loading with intelligent caching
for the all-mpnet-base-v2 embedding model.
"""

from src.embedding.database import ChunkInserter, InsertionStats

__all__ = [
    "ChunkInserter",
    "InsertionStats",
]

# Lazy import model_loader to avoid torch dependency when not needed
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import for model loader components."""
    if name in ("ModelLoader", "ModelLoadError", "ModelValidationError"):
        from src.embedding.model_loader import (
            ModelLoadError,
            ModelLoader,
            ModelValidationError,
        )

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
