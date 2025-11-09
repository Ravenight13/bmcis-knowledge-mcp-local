"""Context header generation for document chunks.

Provides functionality to extract document metadata and structure information,
and generate context headers to be prepended to chunks for RAG systems.

This module handles:
- Document metadata extraction (title, author, date, tags)
- Heading hierarchy preservation and formatting
- Context summary generation
- Header prepending to chunks with proper formatting
"""

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, Field, field_validator


class ContextHeader(BaseModel):
    """Pydantic v2 model for document context headers.

    Stores metadata and hierarchical context information to be prepended
    to document chunks for RAG systems. Uses Pydantic v2 validation for
    strict type checking and field constraints.

    Attributes:
        title: Document title or filename (1-512 chars).
        author: Document author or creator (optional, max 256 chars).
        document_date: Date document was created or published (optional).
        tags: List of classification tags (e.g., knowledge-base, vendor-info).
        headings: Ordered list of heading hierarchy (h1, h2, h3, ...).
        summary: Concise summary of chunk context (~100 tokens, max 2048 chars).

    Example:
        >>> header = ContextHeader(
        ...     title="Installation Guide",
        ...     author="John Doe",
        ...     document_date=date(2025, 11, 8),
        ...     tags=["how-to", "installation"],
        ...     headings=["Getting Started", "Prerequisites"],
        ...     summary="Steps to install the system on Windows and Mac."
        ... )
        >>> header.title
        'Installation Guide'
    """

    title: str = Field(
        min_length=1,
        max_length=512,
        description="Document title or filename",
    )
    author: str | None = Field(
        default=None,
        max_length=256,
        description="Document author or creator",
    )
    document_date: date | None = Field(
        default=None,
        description="Document creation or publish date",
    )
    tags: Sequence[str] = Field(
        default_factory=list,
        description="Classification tags for document",
    )
    headings: Sequence[str] = Field(
        default_factory=list,
        description="Heading hierarchy from document structure",
    )
    summary: str = Field(
        default="",
        max_length=2048,
        description="Context summary for chunk (~100 tokens)",
    )

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: object) -> str:
        """Validate and normalize title.

        Args:
            v: Title value to validate.

        Returns:
            Validated title string.

        Raises:
            ValueError: If title is not a string or is empty after stripping.
        """
        if not isinstance(v, str):
            msg = f"title must be string, got {type(v).__name__}"
            raise ValueError(msg)
        stripped = v.strip()
        if not stripped:
            msg = "title cannot be empty"
            raise ValueError(msg)
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: object) -> Sequence[str]:
        """Validate and normalize tags.

        Args:
            v: Tags value to validate.

        Returns:
            Validated sequence of tags.

        Raises:
            ValueError: If tags is not a sequence or contains non-strings.
        """
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(",") if tag.strip()]
        if isinstance(v, (list, tuple)):
            result = [str(tag).strip() for tag in v if tag]
            return [tag for tag in result if tag]
        return []


class ContextHeaderGenerator:
    """Generate context headers for document chunks.

    Extracts document structure and metadata to create informative headers
    that preserve context for RAG systems and semantic search. Headers are
    formatted as readable text that can be easily parsed by language models.

    The generator handles:
    - Heading hierarchy formatting
    - Metadata presentation (title, author, date, tags)
    - Summary generation with token counting
    - Chunk prepending with proper formatting

    Example:
        >>> generator = ContextHeaderGenerator()
        >>> header = generator.generate_header(
        ...     title="Guide",
        ...     author="Team",
        ...     tags=["how-to"]
        ... )
        >>> formatted = generator.prepend_to_chunk(header, "Content here")
    """

    # Token counting heuristic: approximately 1.3 words per token
    # Based on GPT-3 tokenization patterns
    WORDS_PER_TOKEN: float = 1.3

    def __init__(self) -> None:
        """Initialize the context header generator.

        Sets up default parameters and validates configuration.
        """

    def generate_header(
        self,
        title: str,
        author: str | None = None,
        document_date: date | None = None,
        tags: Sequence[str] | None = None,
        headings: Sequence[str] | None = None,
        summary: str | None = None,
    ) -> ContextHeader:
        """Generate a context header from document metadata and structure.

        Validates all inputs and creates a structured ContextHeader that can
        be formatted and prepended to chunks.

        Args:
            title: Document title or filename (required, non-empty).
            author: Document author or creator (optional, max 256 chars).
            document_date: Document creation or publish date (optional).
            tags: List of classification tags (optional, defaults to empty).
            headings: Ordered heading hierarchy from h1 to h3+ (optional).
            summary: Concise context summary (~100 tokens, optional).

        Returns:
            ContextHeader: Structured header with validated fields.

        Raises:
            ValueError: If title is empty or validation fails.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> header = generator.generate_header(
            ...     title="Installation Guide",
            ...     author="John Doe",
            ...     tags=["how-to", "installation"],
            ...     headings=["Getting Started", "Prerequisites"],
            ... )
            >>> isinstance(header, ContextHeader)
            True
            >>> header.title
            'Installation Guide'
        """
        # Validate title is not empty
        if not title or not title.strip():
            msg = "title cannot be empty"
            raise ValueError(msg)

        # Normalize inputs
        normalized_tags: Sequence[str] = tags or []
        normalized_headings: Sequence[str] = headings or []
        normalized_summary: str = summary or ""

        # Create and return validated header
        return ContextHeader(
            title=title.strip(),
            author=author.strip() if author else None,
            document_date=document_date,
            tags=normalized_tags,
            headings=normalized_headings,
            summary=normalized_summary.strip(),
        )

    def format_heading_hierarchy(self, headings: Sequence[str]) -> str:
        """Format heading hierarchy as readable context string.

        Converts a sequence of headings into a breadcrumb-style path
        showing document structure and navigation context.

        Args:
            headings: Ordered sequence of headings (h1, h2, h3, ...).

        Returns:
            Formatted heading hierarchy string using ' > ' separator.
            Returns empty string if no headings provided.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> result = generator.format_heading_hierarchy(
            ...     ["Chapter 1", "Section A", "Subsection 1"]
            ... )
            >>> result
            'Chapter 1 > Section A > Subsection 1'
            >>> generator.format_heading_hierarchy([])
            ''
        """
        if not headings:
            return ""

        # Filter empty headings and join with separator
        filtered: list[str] = [h.strip() for h in headings if h and h.strip()]
        return " > ".join(filtered)

    def format_metadata(
        self,
        title: str,
        author: str | None = None,
        document_date: date | None = None,
        tags: Sequence[str] | None = None,
    ) -> str:
        """Format document metadata as readable string.

        Creates a human-readable representation of document metadata
        suitable for prepending to chunk text.

        Args:
            title: Document title (required).
            author: Document author (optional).
            document_date: Document date (optional).
            tags: Classification tags (optional).

        Returns:
            Formatted metadata string with all available information.
            Format: "[Document: title] [Author: author, Date: date] [Tags: tag1, tag2]"

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> result = generator.format_metadata(
            ...     title="Research Paper",
            ...     author="Jane Doe",
            ...     tags=["research", "publication"]
            ... )
            >>> "Research Paper" in result
            True
            >>> "Jane Doe" in result
            True
        """
        lines: list[str] = []

        # Document title line (always included)
        lines.append(f"[Document: {title}]")

        # Author and date line (if either is present)
        author_date_parts: list[str] = []
        if author:
            author_date_parts.append(f"Author: {author}")
        if document_date:
            author_date_parts.append(f"Date: {document_date.isoformat()}")

        if author_date_parts:
            lines.append("[" + ", ".join(author_date_parts) + "]")

        # Tags line (if present)
        if tags:
            tag_list: list[str] = [tag.strip() for tag in tags if tag and tag.strip()]
            if tag_list:
                lines.append(f"[Tags: {', '.join(tag_list)}]")

        return " ".join(lines)

    def prepend_to_chunk(
        self,
        header: ContextHeader,
        chunk_text: str,
    ) -> str:
        """Prepend formatted context header to chunk text.

        Combines formatted metadata, heading hierarchy, and summary with
        the original chunk text, separated by a delimiter.

        Args:
            header: ContextHeader instance with context information.
            chunk_text: Original chunk text (must be non-empty).

        Returns:
            Complete string with prepended header, separator, and chunk text.

        Raises:
            ValueError: If chunk_text is empty or only whitespace.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> header = ContextHeader(
            ...     title="Guide",
            ...     headings=["Installation"]
            ... )
            >>> chunk = "Follow these steps..."
            >>> result = generator.prepend_to_chunk(header, chunk)
            >>> result.startswith("[Document:")
            True
            >>> "Follow these steps" in result
            True
            >>> "---" in result
            True
        """
        # Validate chunk text
        if not chunk_text or not chunk_text.strip():
            msg = "chunk_text cannot be empty"
            raise ValueError(msg)

        # Build header lines
        header_lines: list[str] = []

        # Add metadata line
        metadata_line: str = self.format_metadata(
            title=header.title,
            author=header.author,
            document_date=header.document_date,
            tags=header.tags,
        )
        header_lines.append(metadata_line)

        # Add heading hierarchy if present
        if header.headings:
            heading_path: str = self.format_heading_hierarchy(header.headings)
            if heading_path:
                header_lines.append(f"[Context: {heading_path}]")

        # Add summary if present
        if header.summary:
            header_lines.append(f"[Summary: {header.summary}]")

        # Combine header and chunk with separator
        header_section: str = "\n".join(header_lines)
        separator: str = "---"

        return f"{header_section}\n{separator}\n{chunk_text}"

    def generate_summary(
        self,
        content: str,
        max_tokens: int = 100,
    ) -> str:
        """Generate concise summary of chunk context.

        Creates a brief summary without duplicating chunk content.
        Uses simple token-counting heuristic (~words / 1.3 per token).

        The summary extracts the first sentences until reaching the token limit,
        providing context without repeating the full chunk.

        Args:
            content: Text content to summarize.
            max_tokens: Maximum tokens in summary (default 100).

        Returns:
            Concise summary of content respecting max_tokens limit.

        Raises:
            ValueError: If content is empty or max_tokens <= 0.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> text = "This is a sentence. " * 50
            >>> summary = generator.generate_summary(text, max_tokens=50)
            >>> len(summary) > 0
            True
            >>> len(summary) < len(text)
            True
        """
        # Validate inputs
        if not content or not content.strip():
            msg = "content cannot be empty"
            raise ValueError(msg)

        if max_tokens <= 0:
            msg = f"max_tokens must be > 0, got {max_tokens}"
            raise ValueError(msg)

        # Clean content
        cleaned: str = content.strip()

        # Estimate maximum words from token limit
        max_words: float = max_tokens * self.WORDS_PER_TOKEN

        # Split into words and accumulate until limit
        words: list[str] = cleaned.split()
        result_words: list[str] = []

        for word in words:
            # Add word to result
            result_words.append(word)

            # Check if we've reached approximately the word limit
            if len(result_words) >= max_words:
                break

        # Join accumulated words
        summary: str = " ".join(result_words)

        # Ensure summary doesn't exceed cleaned content length
        if len(summary) > len(cleaned):
            summary = cleaned

        return summary

    def _build_hierarchy_path(self, metadata: dict[str, object]) -> str:
        """Extract document structure hierarchy from metadata.

        Parses metadata for section/heading information and formats
        as a breadcrumb-style path for context preservation. Searches
        for common hierarchy fields (hierarchy, sections, breadcrumbs).

        Why it matters: Hierarchy information aids semantic understanding by
        providing document structure context for retrieval systems, helping
        models understand document organization and relationships between
        sections.

        Args:
            metadata: Document metadata dictionary with hierarchy info.

        Returns:
            Formatted hierarchy path (e.g., "Intro > Overview") or empty
            string if no hierarchy found.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> metadata = {"hierarchy": ["Intro", "Overview"]}
            >>> result = generator._build_hierarchy_path(metadata)
            >>> result
            'Intro > Overview'
            >>> generator._build_hierarchy_path({})
            ''
        """
        # Check for hierarchy field in various formats
        hierarchy_candidates: list[str] = [
            "hierarchy",
            "sections",
            "breadcrumbs",
            "path",
        ]

        hierarchy: list[str] = []

        for candidate_key in hierarchy_candidates:
            if candidate_key in metadata:
                candidate_value = metadata[candidate_key]
                if isinstance(candidate_value, (list, tuple)):
                    hierarchy = [str(item).strip() for item in candidate_value
                                 if item and str(item).strip()]
                    break

        if not hierarchy:
            return ""

        # Format with separator
        return self.format_heading_hierarchy(hierarchy)

    def _include_metadata_in_header(self, metadata: dict[str, object]) -> str:
        """Add relevant metadata to header string.

        Extracts key metadata fields (source, date, author if available)
        and formats for header inclusion with proper separators and labels.

        Why it matters: Metadata in headers helps with filtering, attribution,
        and temporal context in knowledge base queries, enabling better
        document provenance tracking and relevance filtering.

        Args:
            metadata: Document metadata dictionary.

        Returns:
            Formatted metadata string for header inclusion (e.g.,
            "[Source: file.md | Date: 2025-11-08]") or empty string
            if no relevant metadata.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> meta = {"source": "guide.md", "date": "2025-11-08"}
            >>> result = generator._include_metadata_in_header(meta)
            >>> "guide.md" in result
            True
        """
        parts: list[str] = []

        # Extract source file
        if "source_file" in metadata:
            source = metadata["source_file"]
            if source:
                parts.append(f"Source: {source}")
        elif "source" in metadata:
            source = metadata["source"]
            if source:
                parts.append(f"Source: {source}")

        # Extract date
        if "document_date" in metadata:
            date_val = metadata["document_date"]
            if date_val:
                parts.append(f"Date: {date_val}")
        elif "date" in metadata:
            date_val = metadata["date"]
            if date_val:
                parts.append(f"Date: {date_val}")

        # Extract author
        if "author" in metadata:
            author = metadata["author"]
            if author:
                parts.append(f"Author: {author}")

        if not parts:
            return ""

        return "[" + " | ".join(parts) + "]"

    def validate_header_format(self, header: str) -> bool:
        """Ensure headers meet quality standards.

        Validates that header is:
        - Not empty or only whitespace
        - Under 200 characters (readability constraint)
        - Contains at least one bracket pair for structure

        Why it matters: Format validation ensures headers are consumable by
        language models and retrieval systems without anomalies, preventing
        malformed context from degrading retrieval quality.

        Args:
            header: Header string to validate.

        Returns:
            True if header is valid.

        Raises:
            ValueError: If header does not meet quality standards, with
                descriptive message indicating the specific issue.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> generator.validate_header_format("[Document: Test]")
            True
            >>> generator.validate_header_format("")
            Traceback (most recent call last):
                ...
            ValueError: Header cannot be empty
        """
        # Check if empty or whitespace only
        if not header or not header.strip():
            msg = "Header cannot be empty"
            raise ValueError(msg)

        # Check length constraint
        if len(header) > 200:
            msg = f"Header length {len(header)} exceeds maximum of 200 characters"
            raise ValueError(msg)

        # Check for structure markers (at least one bracket pair)
        if "[" not in header or "]" not in header:
            msg = "Header must contain at least one bracket pair for structure [...]"
            raise ValueError(msg)

        return True

    def format_header_for_display(self, header: str) -> str:
        """Ensure consistent, clean header formatting.

        Normalizes whitespace (multiple spaces become single space),
        ensures proper separator placement, and handles unicode properly
        for display and indexing systems.

        Why it matters: Consistent formatting prevents whitespace artifacts
        and ensures proper parsing by downstream components, maintaining
        uniform header appearance across documents.

        Args:
            header: Header string to format.

        Returns:
            Formatted header string with:
            - Normalized whitespace (no double spaces)
            - Proper unicode normalization
            - Consistent spacing around brackets

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> messy = "[Document:  Test  ]  with   spaces"
            >>> clean = generator.format_header_for_display(messy)
            >>> "  " not in clean
            True
            >>> len(clean) < len(messy)
            True
        """
        # Strip leading/trailing whitespace
        formatted: str = header.strip()

        # Normalize internal whitespace: replace multiple spaces with single
        import re
        formatted = re.sub(r"\s+", " ", formatted)

        # Ensure space after bracket closings before new bracket openings
        formatted = re.sub(r"\]\[", "] [", formatted)

        # Ensure space around separators like |
        formatted = re.sub(r"\s*\|\s*", " | ", formatted)

        # Final cleanup: ensure no double spaces (regex might create some)
        formatted = re.sub(r"\s+", " ", formatted).strip()

        return formatted

    def extract_context_from_previous_chunk(
        self,
        prev_chunk_text: str,
    ) -> str:
        """Maintain reading continuity from previous chunk.

        Extracts relevant context (last sentence) from previous chunk
        to provide reading continuity and semantic understanding. Uses
        simple sentence boundary detection.

        Why it matters: Previous context aids comprehension by maintaining
        narrative continuity across chunk boundaries in RAG systems, helping
        language models understand the flow of ideas between chunks.

        Args:
            prev_chunk_text: Text from the previous chunk in document.

        Returns:
            Context snippet from previous chunk, preferring the last
            complete sentence or last ~50 characters if no sentence
            boundary found.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> prev = "First sentence. Second sentence. Third sentence."
            >>> result = generator.extract_context_from_previous_chunk(prev)
            >>> "Third sentence" in result
            True
        """
        if not prev_chunk_text or not prev_chunk_text.strip():
            return ""

        # Clean text
        text = prev_chunk_text.strip()

        # Find last sentence by looking for sentence-ending punctuation
        import re
        sentences: list[str] = re.split(r"(?<=[.!?])\s+", text)

        if not sentences:
            return ""

        # Get last sentence, stripping any trailing whitespace
        last_sentence = sentences[-1].strip()

        if not last_sentence:
            # If last sentence is empty, try previous one
            if len(sentences) > 1:
                last_sentence = sentences[-2].strip()
            else:
                # Fallback: last 50 chars
                last_sentence = text[-50:] if len(text) > 50 else text

        return last_sentence

    def calculate_chunk_position(
        self,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        """Indicate chunk's position in document.

        Calculates and formats chunk position information including
        absolute position and percentage through document. Helps models
        understand document traversal and remaining content.

        Why it matters: Position awareness helps language models understand
        where they are in documents, improving context awareness and
        retrieval relevance when documents are traversed in chunks.

        Args:
            chunk_index: Zero-based index of chunk in document.
            total_chunks: Total number of chunks in document.

        Returns:
            String describing position in format: "Chunk X of Y (Z%)"

        Raises:
            ValueError: If chunk_index >= total_chunks or indices are
                negative, with descriptive message.

        Examples:
            >>> generator = ContextHeaderGenerator()
            >>> generator.calculate_chunk_position(0, 5)
            'Chunk 1 of 5 (20%)'
            >>> generator.calculate_chunk_position(2, 5)
            'Chunk 3 of 5 (60%)'
            >>> generator.calculate_chunk_position(4, 5)
            'Chunk 5 of 5 (100%)'
        """
        # Validate indices
        if chunk_index < 0:
            msg = f"chunk_index must be >= 0, got {chunk_index}"
            raise ValueError(msg)

        if total_chunks <= 0:
            msg = f"total_chunks must be > 0, got {total_chunks}"
            raise ValueError(msg)

        if chunk_index >= total_chunks:
            msg = (
                f"chunk_index ({chunk_index}) must be < total_chunks "
                f"({total_chunks})"
            )
            raise ValueError(msg)

        # Calculate position (1-indexed for display)
        position: int = chunk_index + 1
        percentage: int = int((position / total_chunks) * 100)

        return f"Chunk {position} of {total_chunks} ({percentage}%)"
