"""Markdown document reader with metadata extraction and content parsing.

Provides functionality to read markdown files, extract frontmatter metadata
(YAML or JSON), parse document structure (headings, links), and extract
plain text content. Supports type-safe metadata models with Pydantic v2.

Module Features:
    - YAML and JSON frontmatter parsing
    - Automatic metadata extraction
    - Heading hierarchy preservation
    - Link and reference extraction
    - Graceful error handling for malformed documents
    - Comprehensive logging of parsing operations

Example:
    >>> from src.document_parsing import MarkdownReader
    >>> reader = MarkdownReader()
    >>> doc = reader.read_file(Path("documents/example.md"))
    >>> print(doc.metadata.title)
    >>> print(doc.body)
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.core.logging import StructuredLogger

# Module logger for document parsing operations
logger: logging.Logger = StructuredLogger.get_logger(__name__)


class ParseError(Exception):
    """Raised when document parsing fails.

    Attributes:
        message: Description of what failed during parsing.
        file_path: Path to the file that failed to parse (if applicable).
        context: Additional context about the parsing error.
    """

    def __init__(
        self, message: str, file_path: Path | None = None, context: str | None = None
    ) -> None:
        """Initialize ParseError with optional context.

        Args:
            message: Error message describing the parsing failure.
            file_path: Path to the file that failed to parse.
            context: Additional context about the error.
        """
        self.message = message
        self.file_path = file_path
        self.context = context
        full_message = f"{message}"
        if file_path:
            full_message += f" (file: {file_path})"
        if context:
            full_message += f" - {context}"
        super().__init__(full_message)


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from document frontmatter.

    Handles metadata from YAML or JSON frontmatter with flexible field
    handling. Unknown fields are preserved in a generic metadata dict.
    Implements validation for required and optional fields.

    Attributes:
        title: Document title (required).
        author: Document author (optional).
        date: Publication/creation date as ISO string or datetime (optional).
        tags: List of document tags for categorization (optional).
        description: Short document description (optional).
        custom_fields: Dictionary of additional metadata fields (optional).
    """

    title: str = Field(
        ...,
        description="Document title",
        min_length=1,
    )
    author: str | None = Field(
        default=None,
        description="Document author",
    )
    date: str | None = Field(
        default=None,
        description="Document date (ISO format)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Document tags for categorization",
    )
    description: str | None = Field(
        default=None,
        description="Short document description",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata fields",
    )

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Example Document",
                    "author": "John Doe",
                    "date": "2024-01-15",
                    "tags": ["example", "markdown"],
                    "description": "An example document",
                }
            ]
        },
    }

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: Any) -> list[str]:
        """Normalize tags to list of strings.

        Args:
            v: Tags value (string, list, or None).

        Returns:
            List of tag strings.
        """
        if v is None:
            return []
        if isinstance(v, str):
            # Split comma-separated string into tags
            return [tag.strip() for tag in v.split(",") if tag.strip()]
        if isinstance(v, list):
            return [str(tag).strip() for tag in v if tag]
        return []

    @field_validator("date", mode="before")
    @classmethod
    def normalize_date(cls, v: Any) -> str | None:
        """Normalize date to ISO string format.

        Args:
            v: Date value (string, datetime, or None).

        Returns:
            ISO format date string or None.
        """
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str):
            return v.strip() if v.strip() else None
        return str(v)


@dataclass
class Link:
    """Represents a link in the document.

    Attributes:
        text: Link text/label as displayed.
        url: Link URL/target.
        line_number: Line number where link appears in document.
    """

    text: str
    url: str
    line_number: int


@dataclass
class Heading:
    """Represents a heading in the document.

    Attributes:
        level: Heading level (1-6, where 1 is h1 and 6 is h6).
        text: Heading text content.
        line_number: Line number where heading appears in document.
    """

    level: int
    text: str
    line_number: int


@dataclass
class ParsedDocument:
    """Result of parsing a markdown document.

    Attributes:
        metadata: Structured metadata extracted from frontmatter.
        body: Document body text (markdown stripped to plain text).
        headings: List of headings found in document.
        links: List of links found in document.
        raw_content: Original file content (for reference).
    """

    metadata: DocumentMetadata
    body: str
    headings: list[Heading]
    links: list[Link]
    raw_content: str


class MarkdownReader:
    """Read and parse markdown documents with metadata extraction.

    Supports YAML and JSON frontmatter parsing, heading hierarchy extraction,
    link preservation, and graceful error handling for malformed documents.

    Example:
        >>> from pathlib import Path
        >>> from src.document_parsing import MarkdownReader
        >>> reader = MarkdownReader()
        >>> doc = reader.read_file(Path("docs/example.md"))
        >>> print(f"Title: {doc.metadata.title}")
        >>> print(f"Body length: {len(doc.body)} chars")
        >>> for heading in doc.headings:
        ...     print(f"  {'#' * heading.level} {heading.text}")
    """

    # Regex patterns for markdown parsing
    FRONTMATTER_YAML_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE
    )
    FRONTMATTER_JSON_PATTERN = re.compile(
        r"^~~~json\s*\n(.*?)\n~~~\s*\n", re.DOTALL | re.MULTILINE
    )
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*#*)?$", re.MULTILINE)
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    CODE_FENCE_PATTERN = re.compile(r"^(?:```|~~~).*?^(?:```|~~~)", re.MULTILINE | re.DOTALL)

    def __init__(self) -> None:
        """Initialize MarkdownReader with default configuration."""
        logger.debug("MarkdownReader initialized")

    def read_file(self, file_path: Path) -> ParsedDocument:
        """Read and parse a markdown file.

        Reads the file, extracts frontmatter (YAML or JSON), parses document
        structure (headings, links), and extracts plain text body.

        Args:
            file_path: Path to the markdown file to read.

        Returns:
            ParsedDocument: Parsed document with metadata, body, and structure.

        Raises:
            ParseError: If file cannot be read or parsing fails.
            FileNotFoundError: If file does not exist.

        Example:
            >>> from pathlib import Path
            >>> reader = MarkdownReader()
            >>> doc = reader.read_file(Path("documents/example.md"))
            >>> print(doc.metadata.title)
        """
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8")
            logger.debug(
                "Read markdown file: %s (size: %d bytes)", file_path, len(content)
            )

            # Parse document components
            metadata = self._extract_metadata(content)
            body = self._extract_body(content)
            headings = self._extract_headings(content)
            links = self._extract_links(content)

            logger.info(
                "Parsed markdown document: %s (headings: %d, links: %d)",
                file_path,
                len(headings),
                len(links),
            )

            return ParsedDocument(
                metadata=metadata,
                body=body,
                headings=headings,
                links=links,
                raw_content=content,
            )

        except ParseError:
            # Re-raise ParseError with file context
            raise
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to parse markdown file: %s", file_path, exc_info=True)
            raise ParseError(
                f"Failed to parse markdown file: {e}", file_path=file_path
            ) from e

    def _extract_metadata(self, content: str) -> DocumentMetadata:
        """Extract metadata from document frontmatter.

        Attempts to parse YAML frontmatter (--- ... ---) followed by
        JSON frontmatter (~~~json ... ~~~). If no frontmatter is found,
        returns minimal metadata with empty title.

        Args:
            content: Raw document content.

        Returns:
            DocumentMetadata: Extracted metadata from frontmatter.

        Raises:
            ParseError: If frontmatter parsing fails.
        """
        # Try YAML frontmatter first
        yaml_match = self.FRONTMATTER_YAML_PATTERN.match(content)
        if yaml_match:
            try:
                return self._parse_yaml_metadata(yaml_match.group(1))
            except Exception as e:
                logger.warning(
                    "Failed to parse YAML frontmatter, trying fallback: %s", e
                )
                # Continue to try JSON or fallback

        # Try JSON frontmatter
        json_match = self.FRONTMATTER_JSON_PATTERN.match(content)
        if json_match:
            try:
                return self._parse_json_metadata(json_match.group(1))
            except Exception as e:
                logger.warning(
                    "Failed to parse JSON frontmatter, using fallback: %s", e
                )

        # No valid frontmatter found, extract title from first heading
        logger.debug("No valid frontmatter found, extracting title from content")
        return self._extract_default_metadata(content)

    def _parse_yaml_metadata(self, yaml_content: str) -> DocumentMetadata:
        """Parse YAML frontmatter into DocumentMetadata.

        Uses a simple YAML-like parser to extract key-value pairs without
        external dependencies. Handles basic YAML syntax:
        - Simple key: value pairs
        - Lists: key: [item1, item2]
        - Multiline strings with |

        Args:
            yaml_content: YAML content from frontmatter.

        Returns:
            DocumentMetadata: Parsed metadata.

        Raises:
            ParseError: If YAML parsing fails.
        """
        metadata_dict: dict[str, Any] = {}

        for line in yaml_content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Parse value
            if value.lower() in ("true", "false"):
                metadata_dict[key] = value.lower() == "true"
            elif value.startswith("[") and value.endswith("]"):
                # Simple list parsing
                items = value[1:-1].split(",")
                metadata_dict[key] = [item.strip().strip('"\'') for item in items]
            elif value.startswith('"') and value.endswith('"'):
                metadata_dict[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                metadata_dict[key] = value[1:-1]
            else:
                metadata_dict[key] = value

        logger.debug("Parsed YAML metadata with %d fields", len(metadata_dict))

        try:
            return DocumentMetadata(**metadata_dict)
        except Exception as e:
            logger.error("YAML metadata validation failed: %s", e)
            raise ParseError(f"YAML metadata validation failed: {e}", context=str(metadata_dict)) from e

    def _parse_json_metadata(self, json_content: str) -> DocumentMetadata:
        """Parse JSON frontmatter into DocumentMetadata.

        Args:
            json_content: JSON content from frontmatter.

        Returns:
            DocumentMetadata: Parsed metadata.

        Raises:
            ParseError: If JSON parsing fails.
        """
        try:
            metadata_dict = json.loads(json_content)
            logger.debug("Parsed JSON metadata with %d fields", len(metadata_dict))
            return DocumentMetadata(**metadata_dict)
        except json.JSONDecodeError as e:
            logger.error("JSON metadata parsing failed: %s", e)
            raise ParseError(f"JSON metadata parsing failed: {e}", context=json_content) from e
        except Exception as e:
            logger.error("JSON metadata validation failed: %s", e)
            raise ParseError(f"JSON metadata validation failed: {e}", context=json_content) from e

    def _extract_default_metadata(self, content: str) -> DocumentMetadata:
        """Extract minimal metadata from content when no frontmatter present.

        Uses first heading as title if available, otherwise "Untitled".

        Args:
            content: Raw document content.

        Returns:
            DocumentMetadata: Minimal metadata with extracted title.
        """
        # Try to get title from first heading
        heading_match = self.HEADING_PATTERN.search(content)
        if heading_match:
            title = heading_match.group(2).strip()
            logger.debug("Extracted title from first heading: %s", title)
            return DocumentMetadata(title=title)

        # Fallback to "Untitled"
        logger.debug("No heading found, using default title: Untitled")
        return DocumentMetadata(title="Untitled")

    def _extract_body(self, content: str) -> str:
        """Extract plain text body from markdown content.

        Removes frontmatter and converts markdown to plain text by:
        - Removing code blocks
        - Removing HTML tags
        - Converting markdown formatting (bold, italic, links)
        - Preserving headings and paragraphs

        Args:
            content: Raw markdown content.

        Returns:
            Plain text body content.
        """
        # Remove frontmatter
        body = self.FRONTMATTER_YAML_PATTERN.sub("", content)
        body = self.FRONTMATTER_JSON_PATTERN.sub("", body)

        # Remove code blocks
        body = self.CODE_FENCE_PATTERN.sub("", body)

        # Remove inline code (backticks)
        body = re.sub(r"`([^`]+)`", r"\1", body)

        # Convert links [text](url) to just text
        body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)

        # Remove bold/italic markers (order matters: *** before ** and *)
        # Triple asterisk/underscore (bold italic)
        body = re.sub(r"\*{3}(.+?)\*{3}", r"\1", body)
        body = re.sub(r"_{3}(.+?)_{3}", r"\1", body)
        # Double asterisk/underscore (bold)
        body = re.sub(r"\*{2}(.+?)\*{2}", r"\1", body)
        body = re.sub(r"_{2}(.+?)_{2}", r"\1", body)
        # Single asterisk/underscore (italic)
        body = re.sub(r"\*(.+?)\*", r"\1", body)
        body = re.sub(r"_(.+?)_", r"\1", body)

        # Remove HTML tags
        body = re.sub(r"<[^>]+>", "", body)

        # Remove horizontal rules
        body = re.sub(r"^[-*_]{3,}$", "", body, flags=re.MULTILINE)

        # Clean up excessive whitespace
        body = re.sub(r"\n\s*\n\s*\n", "\n\n", body)  # Max 2 newlines
        body = re.sub(r"[ \t]+", " ", body)  # Single spaces
        body = body.strip()

        logger.debug("Extracted body text: %d chars", len(body))
        return body

    def _extract_headings(self, content: str) -> list[Heading]:
        """Extract heading hierarchy from markdown content.

        Excludes headings from code blocks.

        Args:
            content: Raw markdown content.

        Returns:
            List of Heading objects in order of appearance.
        """
        # Remove code blocks to avoid extracting headings from within them
        content_without_code = self.CODE_FENCE_PATTERN.sub("", content)

        headings: list[Heading] = []
        for line_num, line in enumerate(content_without_code.split("\n"), 1):
            match = self.HEADING_PATTERN.match(line)
            if match:
                level = len(match.group(1))  # Count # symbols
                text = match.group(2).strip()
                headings.append(Heading(level=level, text=text, line_number=line_num))

        logger.debug("Extracted %d headings from document", len(headings))
        return headings

    def _extract_links(self, content: str) -> list[Link]:
        """Extract all links from markdown content.

        Args:
            content: Raw markdown content.

        Returns:
            List of Link objects in order of appearance.
        """
        links: list[Link] = []
        for line_num, line in enumerate(content.split("\n"), 1):
            # Skip code blocks
            if line.strip().startswith("```"):
                continue

            for match in self.LINK_PATTERN.finditer(line):
                text = match.group(1)
                url = match.group(2)
                links.append(Link(text=text, url=url, line_number=line_num))

        logger.debug("Extracted %d links from document", len(links))
        return links
