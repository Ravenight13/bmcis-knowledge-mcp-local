"""Document parsing and extraction utilities.

Provides functionality for reading and parsing various document formats
including markdown with metadata extraction, frontmatter parsing, and
structured content extraction. Includes tokenization support for token counting
and encoding text using OpenAI's tiktoken library.

Batch processing pipeline orchestrates all components for end-to-end document
ingestion into the knowledge base.
"""

from src.document_parsing.batch_processor import (
    BatchConfig,
    BatchProcessor,
    Chunker,
    ContextHeaderGenerator,
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
    "BatchConfig",
    "BatchProcessor",
    "BatchProcessingStats",
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
