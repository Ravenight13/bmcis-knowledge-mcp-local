"""Document parsing and extraction utilities.

Provides functionality for reading and parsing various document formats
including markdown with metadata extraction, frontmatter parsing, and
structured content extraction. Includes tokenization support for token counting
and encoding text using OpenAI's tiktoken library.

Batch processing pipeline orchestrates all components for end-to-end document
ingestion into the knowledge base.
"""

from src.document_parsing.batch_processor import (
    Batch,
    BatchConfig,
    BatchProgress,
    BatchProcessor,
    BatchResult,
    Chunker,
    ContextHeaderGenerator,
    ErrorRecoveryAction,
    calculate_batch_size,
    create_batches,
)
from src.document_parsing.markdown_reader import (
    DocumentMetadata,
    MarkdownReader,
    ParseError,
)
from src.document_parsing.models import (
    BatchProcessingStats,
    ProcessedChunk,
)
from src.document_parsing.tokenizer import (
    Tokenizer,
    TokenizerConfig,
)

__all__ = [
    # Batch processing
    "Batch",
    "BatchConfig",
    "BatchProgress",
    "BatchProcessor",
    "BatchProcessingStats",
    "BatchResult",
    "ErrorRecoveryAction",
    "calculate_batch_size",
    "create_batches",
    # Components
    "Chunker",
    "ContextHeaderGenerator",
    "MarkdownReader",
    "ParseError",
    "Tokenizer",
    "TokenizerConfig",
    # Models
    "DocumentMetadata",
    "ProcessedChunk",
]
