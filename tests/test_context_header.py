"""Comprehensive test suite for context header generation module.

Tests cover:
- ContextHeader Pydantic model validation
- ContextHeaderGenerator method functionality
- Header formatting and generation
- Chunk prepending with various inputs
- Edge cases and error handling
"""

from datetime import date
from typing import Sequence

import pytest

from src.document_parsing.context_header import (
    ContextHeader,
    ContextHeaderGenerator,
)


class TestContextHeaderModel:
    """Test suite for ContextHeader Pydantic v2 model."""

    def test_minimal_header_creation(self) -> None:
        """Test creating header with only required title field."""
        header = ContextHeader(title="Test Document")
        assert header.title == "Test Document"
        assert header.author is None
        assert header.document_date is None
        assert header.tags == []
        assert header.headings == []
        assert header.summary == ""

    def test_full_header_creation(self) -> None:
        """Test creating header with all fields populated."""
        test_date = date(2025, 11, 8)
        header = ContextHeader(
            title="Complete Document",
            author="John Doe",
            document_date=test_date,
            tags=["vendor-info", "how-to"],
            headings=["Chapter 1", "Section A"],
            summary="This is a test summary.",
        )
        assert header.title == "Complete Document"
        assert header.author == "John Doe"
        assert header.document_date == test_date
        assert "vendor-info" in header.tags
        assert "how-to" in header.tags
        assert "Chapter 1" in header.headings
        assert "Section A" in header.headings
        assert header.summary == "This is a test summary."

    def test_title_required(self) -> None:
        """Test that title field is required."""
        with pytest.raises(ValueError):
            ContextHeader(title="")

    def test_title_validation_whitespace(self) -> None:
        """Test that title is stripped of whitespace."""
        header = ContextHeader(title="  Document Title  ")
        assert header.title == "Document Title"

    def test_author_optional(self) -> None:
        """Test that author field is optional."""
        header = ContextHeader(title="Doc", author=None)
        assert header.author is None

    def test_author_max_length(self) -> None:
        """Test author field maximum length constraint."""
        # Max length is 256
        valid_author = "a" * 256
        header = ContextHeader(title="Doc", author=valid_author)
        assert header.author == valid_author

        # Exceeds max length
        invalid_author = "a" * 257
        with pytest.raises(ValueError):
            ContextHeader(title="Doc", author=invalid_author)

    def test_title_max_length(self) -> None:
        """Test title field maximum length constraint."""
        # Max length is 512
        valid_title = "a" * 512
        header = ContextHeader(title=valid_title)
        assert header.title == valid_title

        # Exceeds max length
        invalid_title = "a" * 513
        with pytest.raises(ValueError):
            ContextHeader(title=invalid_title)

    def test_tags_as_list(self) -> None:
        """Test tags field accepts list input."""
        header = ContextHeader(
            title="Doc",
            tags=["tag1", "tag2", "tag3"],
        )
        assert "tag1" in header.tags
        assert "tag2" in header.tags
        assert len(header.tags) == 3

    def test_tags_as_tuple(self) -> None:
        """Test tags field accepts tuple input."""
        header = ContextHeader(
            title="Doc",
            tags=("tag1", "tag2"),
        )
        assert "tag1" in header.tags
        assert len(header.tags) == 2

    def test_tags_default_empty(self) -> None:
        """Test tags defaults to empty sequence."""
        header = ContextHeader(title="Doc")
        assert len(header.tags) == 0

    def test_headings_hierarchy(self) -> None:
        """Test headings field preserves hierarchy order."""
        headings = ["Chapter 1", "Section A", "Subsection 1"]
        header = ContextHeader(title="Doc", headings=headings)
        assert list(header.headings) == headings

    def test_summary_field(self) -> None:
        """Test summary field with typical content."""
        summary_text = "This is a summary of the section."
        header = ContextHeader(title="Doc", summary=summary_text)
        assert header.summary == summary_text

    def test_summary_max_length(self) -> None:
        """Test summary field maximum length constraint."""
        # Max length is 2048
        valid_summary = "a" * 2048
        header = ContextHeader(title="Doc", summary=valid_summary)
        assert header.summary == valid_summary

        # Exceeds max length
        invalid_summary = "a" * 2049
        with pytest.raises(ValueError):
            ContextHeader(title="Doc", summary=invalid_summary)

    def test_model_dump(self) -> None:
        """Test Pydantic model_dump method."""
        header = ContextHeader(
            title="Test",
            author="Author",
            tags=["tag1"],
        )
        dumped = header.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["title"] == "Test"
        assert dumped["author"] == "Author"

    def test_model_validate(self) -> None:
        """Test Pydantic model_validate method."""
        data = {
            "title": "Test",
            "author": "Author",
            "tags": ["tag1"],
        }
        header = ContextHeader.model_validate(data)
        assert header.title == "Test"
        assert header.author == "Author"


class TestContextHeaderGenerator:
    """Test suite for ContextHeaderGenerator class."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_generator_initialization(self, generator: ContextHeaderGenerator) -> None:
        """Test generator initialization."""
        assert isinstance(generator, ContextHeaderGenerator)
        assert hasattr(generator, "WORDS_PER_TOKEN")
        assert generator.WORDS_PER_TOKEN == 1.3

    def test_generate_header_minimal(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test generating header with minimal inputs."""
        header = generator.generate_header(title="Document")
        assert isinstance(header, ContextHeader)
        assert header.title == "Document"
        assert header.author is None
        assert header.document_date is None

    def test_generate_header_full(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test generating header with all parameters."""
        test_date = date(2025, 11, 8)
        header = generator.generate_header(
            title="Complete Doc",
            author="Jane Doe",
            document_date=test_date,
            tags=["vendor", "how-to"],
            headings=["Intro", "Details"],
            summary="A summary.",
        )
        assert header.title == "Complete Doc"
        assert header.author == "Jane Doe"
        assert header.document_date == test_date
        assert "vendor" in header.tags
        assert "how-to" in header.tags
        assert "Intro" in header.headings
        assert header.summary == "A summary."

    def test_generate_header_empty_title_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            generator.generate_header(title="")

        with pytest.raises(ValueError, match="title cannot be empty"):
            generator.generate_header(title="   ")

    def test_generate_header_title_whitespace_stripped(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that title whitespace is stripped."""
        header = generator.generate_header(title="  Trimmed Title  ")
        assert header.title == "Trimmed Title"

    def test_generate_header_author_whitespace_stripped(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that author whitespace is stripped."""
        header = generator.generate_header(
            title="Doc",
            author="  John Doe  ",
        )
        assert header.author == "John Doe"

    def test_format_heading_hierarchy_single(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting single heading."""
        result = generator.format_heading_hierarchy(["Chapter 1"])
        assert result == "Chapter 1"

    def test_format_heading_hierarchy_multiple(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting multiple headings."""
        headings = ["Part I", "Chapter 2", "Section B", "Subsection 1"]
        result = generator.format_heading_hierarchy(headings)
        assert result == "Part I > Chapter 2 > Section B > Subsection 1"

    def test_format_heading_hierarchy_empty(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting empty heading list."""
        result = generator.format_heading_hierarchy([])
        assert result == ""

    def test_format_heading_hierarchy_with_whitespace(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that heading whitespace is stripped."""
        headings = ["  Chapter 1  ", "  Section A  "]
        result = generator.format_heading_hierarchy(headings)
        assert result == "Chapter 1 > Section A"

    def test_format_heading_hierarchy_filters_empty(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty headings are filtered out."""
        headings = ["Chapter 1", "", "  ", "Section A"]
        result = generator.format_heading_hierarchy(headings)
        assert result == "Chapter 1 > Section A"

    def test_format_metadata_title_only(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with title only."""
        result = generator.format_metadata(title="Document")
        assert "[Document: Document]" in result

    def test_format_metadata_with_author(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with author."""
        result = generator.format_metadata(
            title="Document",
            author="John Doe",
        )
        assert "[Document: Document]" in result
        assert "John Doe" in result

    def test_format_metadata_with_date(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with date."""
        test_date = date(2025, 11, 8)
        result = generator.format_metadata(
            title="Document",
            document_date=test_date,
        )
        assert "[Document: Document]" in result
        assert "2025-11-08" in result

    def test_format_metadata_with_tags(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with tags."""
        result = generator.format_metadata(
            title="Document",
            tags=["vendor", "how-to"],
        )
        assert "[Document: Document]" in result
        assert "vendor" in result
        assert "how-to" in result

    def test_format_metadata_all_fields(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with all fields."""
        test_date = date(2025, 11, 8)
        result = generator.format_metadata(
            title="Guide",
            author="Jane Doe",
            document_date=test_date,
            tags=["installation", "windows"],
        )
        assert "[Document: Guide]" in result
        assert "Jane Doe" in result
        assert "2025-11-08" in result
        assert "installation" in result
        assert "windows" in result

    def test_format_metadata_empty_tags(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with empty tags list."""
        result = generator.format_metadata(
            title="Document",
            tags=[],
        )
        assert "[Document: Document]" in result
        # Should not have tags section
        assert "[Tags:" not in result

    def test_format_metadata_tags_with_whitespace(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test formatting metadata with tags containing whitespace."""
        result = generator.format_metadata(
            title="Document",
            tags=["  tag1  ", "  tag2  "],
        )
        assert "tag1" in result
        assert "tag2" in result

    def test_prepend_to_chunk_basic(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test prepending header to chunk."""
        header = ContextHeader(title="Guide")
        chunk = "This is chunk content."
        result = generator.prepend_to_chunk(header, chunk)

        assert "[Document: Guide]" in result
        assert "---" in result
        assert "This is chunk content." in result

    def test_prepend_to_chunk_with_hierarchy(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test prepending header with heading hierarchy."""
        header = ContextHeader(
            title="Manual",
            headings=["Installation", "Windows Setup"],
        )
        chunk = "Follow these steps."
        result = generator.prepend_to_chunk(header, chunk)

        assert "[Document: Manual]" in result
        assert "[Context: Installation > Windows Setup]" in result
        assert "Follow these steps." in result
        assert "---" in result

    def test_prepend_to_chunk_with_summary(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test prepending header with summary."""
        header = ContextHeader(
            title="Guide",
            summary="Steps to complete installation.",
        )
        chunk = "Detailed installation instructions here."
        result = generator.prepend_to_chunk(header, chunk)

        assert "[Summary: Steps to complete installation.]" in result
        assert "Detailed installation instructions here." in result

    def test_prepend_to_chunk_full(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test prepending header with all fields."""
        test_date = date(2025, 11, 8)
        header = ContextHeader(
            title="Installation Guide",
            author="Jane Doe",
            document_date=test_date,
            tags=["installation", "windows"],
            headings=["Getting Started", "System Requirements"],
            summary="Requirements and initial setup information.",
        )
        chunk = "Windows 10 or later required. 4GB RAM minimum."
        result = generator.prepend_to_chunk(header, chunk)

        assert "[Document: Installation Guide]" in result
        assert "Jane Doe" in result
        assert "2025-11-08" in result
        assert "installation" in result
        assert "[Context: Getting Started > System Requirements]" in result
        assert "[Summary: Requirements and initial setup information.]" in result
        assert "Windows 10 or later required" in result

    def test_prepend_to_chunk_empty_chunk_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty chunk raises ValueError."""
        header = ContextHeader(title="Guide")
        with pytest.raises(ValueError, match="chunk_text cannot be empty"):
            generator.prepend_to_chunk(header, "")

        with pytest.raises(ValueError, match="chunk_text cannot be empty"):
            generator.prepend_to_chunk(header, "   ")

    def test_prepend_to_chunk_preserves_content(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that prepending preserves exact chunk content."""
        header = ContextHeader(title="Doc")
        chunk = "Line 1\nLine 2\nLine 3"
        result = generator.prepend_to_chunk(header, chunk)

        # Verify exact chunk appears in result
        assert "Line 1\nLine 2\nLine 3" in result

    def test_generate_summary_basic(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test generating summary from content."""
        content = "This is a test sentence. " * 50
        summary = generator.generate_summary(content, max_tokens=50)

        assert len(summary) > 0
        assert len(summary) <= len(content)

    def test_generate_summary_respects_token_limit(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that summary respects max_tokens parameter."""
        # Create content with known word count
        content = "word " * 200  # 200 words
        summary = generator.generate_summary(content, max_tokens=50)

        # With 1.3 words per token, 50 tokens = ~65 words
        # Summary should be much shorter than original
        assert len(summary.split()) < 150

    def test_generate_summary_empty_content_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            generator.generate_summary("", max_tokens=50)

        with pytest.raises(ValueError, match="content cannot be empty"):
            generator.generate_summary("   ", max_tokens=50)

    def test_generate_summary_invalid_max_tokens_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that invalid max_tokens raises ValueError."""
        content = "Some content."
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            generator.generate_summary(content, max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            generator.generate_summary(content, max_tokens=-5)

    def test_generate_summary_short_content(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test generating summary from short content."""
        content = "Brief content."
        summary = generator.generate_summary(content, max_tokens=100)

        # Should return content or similar length
        assert summary == content or len(summary) <= len(content)

    def test_generate_summary_long_token_limit(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test summary with high token limit."""
        content = "This is a test. " * 50
        summary = generator.generate_summary(content, max_tokens=500)

        # Should be equal to or close to original
        assert len(summary) >= len(content) * 0.8


class TestContextHeaderIntegration:
    """Integration tests for complete header generation workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from generation to prepending."""
        generator = ContextHeaderGenerator()

        # Generate header
        header = generator.generate_header(
            title="System Configuration",
            author="Admin Team",
            document_date=date(2025, 11, 8),
            tags=["configuration", "production"],
            headings=["Network Setup", "Security"],
            summary="Configure network interfaces and security settings.",
        )

        # Prepare chunk
        chunk_text = (
            "Set up IP addresses for all network interfaces. "
            "Configure firewall rules for production environment."
        )

        # Prepend header to chunk
        result = generator.prepend_to_chunk(header, chunk_text)

        # Validate result structure
        assert "[Document: System Configuration]" in result
        assert "Admin Team" in result
        assert "2025-11-08" in result
        assert "configuration" in result
        assert "[Context: Network Setup > Security]" in result
        assert "[Summary: Configure network interfaces" in result
        assert "---" in result
        assert chunk_text in result

    def test_header_formatting_consistency(self) -> None:
        """Test that header formatting is consistent across calls."""
        generator = ContextHeaderGenerator()

        header = ContextHeader(
            title="Document",
            author="Author",
            tags=["tag1", "tag2"],
        )

        chunk = "Content"

        # Generate multiple times
        result1 = generator.prepend_to_chunk(header, chunk)
        result2 = generator.prepend_to_chunk(header, chunk)

        # Should be identical
        assert result1 == result2

    def test_special_characters_in_metadata(self) -> None:
        """Test handling of special characters in metadata fields."""
        generator = ContextHeaderGenerator()

        header = ContextHeader(
            title="Document: Guide & Manual",
            author="Jane Doe (Lead)",
            tags=["tag-1", "tag_2"],
            headings=["Section 1.1", "Sub-section A"],
        )

        chunk = "Content with & special < characters >"
        result = generator.prepend_to_chunk(header, chunk)

        # Should contain all special characters
        assert "Document: Guide & Manual" in result
        assert "Jane Doe (Lead)" in result
        assert "tag-1" in result
        assert "Content with & special < characters >" in result

    def test_long_heading_hierarchy(self) -> None:
        """Test handling of deep heading hierarchies."""
        generator = ContextHeaderGenerator()

        # Create deep hierarchy (h1 through h6)
        headings = [
            "Part I",
            "Chapter 1",
            "Section A",
            "Subsection 1",
            "Paragraph a",
            "Sub-paragraph i",
        ]

        header = ContextHeader(
            title="Complex Document",
            headings=headings,
        )

        chunk = "Deeply nested content."
        result = generator.prepend_to_chunk(header, chunk)

        # Verify hierarchy is preserved
        for heading in headings:
            assert heading in result

        # Verify separator format
        assert " > " in result


class TestContextHeaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_in_title(self) -> None:
        """Test handling of Unicode characters in title."""
        header = ContextHeader(title="ドキュメント (Document)")
        assert "ドキュメント" in header.title

    def test_unicode_in_author(self) -> None:
        """Test handling of Unicode characters in author."""
        header = ContextHeader(
            title="Doc",
            author="José García",
        )
        assert "José" in header.author

    def test_multiline_summary(self) -> None:
        """Test handling of multiline summary text."""
        summary = "Line 1\nLine 2\nLine 3"
        header = ContextHeader(title="Doc", summary=summary)
        assert header.summary == summary

    def test_very_long_tags(self) -> None:
        """Test handling of many tags."""
        tags = [f"tag-{i}" for i in range(100)]
        header = ContextHeader(title="Doc", tags=tags)
        assert len(header.tags) == 100

    def test_empty_heading_strings_filtered(self) -> None:
        """Test that empty heading strings are handled."""
        generator = ContextHeaderGenerator()
        headings: Sequence[str] = ["Chapter", "", "Section"]
        result = generator.format_heading_hierarchy(headings)

        # Empty string should not appear in result
        assert result == "Chapter > Section"
        assert " > " in result
        assert "  " not in result  # No double spaces

    def test_chunk_with_markdown_formatting(self) -> None:
        """Test prepending header to chunk with Markdown formatting."""
        generator = ContextHeaderGenerator()
        header = ContextHeader(title="Guide")

        chunk = "# Main Heading\n\n**Bold text** and *italic text*.\n\n```code block```"
        result = generator.prepend_to_chunk(header, chunk)

        # Markdown should be preserved
        assert "# Main Heading" in result
        assert "**Bold text**" in result
        assert "```code block```" in result

    def test_chunk_with_code_blocks(self) -> None:
        """Test prepending header to chunk containing code."""
        generator = ContextHeaderGenerator()
        header = ContextHeader(title="Code Example")

        chunk = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = generator.prepend_to_chunk(header, chunk)

        assert 'def hello_world():' in result
        assert 'print("Hello, World!")' in result

    def test_summary_generation_with_special_chars(self) -> None:
        """Test summary generation with special characters."""
        generator = ContextHeaderGenerator()

        content = "Special chars: & < > \" ' \\ / | @ # $ % ^ ( ) [ ] { } " * 20
        summary = generator.generate_summary(content, max_tokens=50)

        # Should contain some special characters from content
        assert len(summary) > 0
        assert len(summary) <= len(content)


class TestContextHeaderPerformance:
    """Performance-related tests."""

    def test_large_header_generation(self) -> None:
        """Test generating header with large amounts of data."""
        generator = ContextHeaderGenerator()

        # Large tags list
        tags = [f"tag-{i}" for i in range(1000)]

        # Large headings list
        headings = [f"Heading {i}" for i in range(100)]

        header = generator.generate_header(
            title="Large Document",
            tags=tags,
            headings=headings,
        )

        assert len(header.tags) == 1000
        assert len(header.headings) == 100

    def test_large_chunk_prepending(self) -> None:
        """Test prepending header to large chunk."""
        generator = ContextHeaderGenerator()
        header = ContextHeader(title="Large Document")

        # Create large chunk (10,000 words)
        chunk = "word " * 10000
        result = generator.prepend_to_chunk(header, chunk)

        # Should contain header and chunk
        assert "[Document: Large Document]" in result
        assert chunk in result

    def test_summary_generation_large_content(self) -> None:
        """Test summary generation on large content."""
        generator = ContextHeaderGenerator()

        # Large content (10,000 words)
        content = "This is a test word. " * 10000

        summary = generator.generate_summary(content, max_tokens=100)

        # Should be much shorter than content
        assert len(summary) < len(content) / 10
        assert len(summary) > 0


class TestBuildHierarchyPath:
    """Test suite for _build_hierarchy_path method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_build_hierarchy_path_from_hierarchy_field(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction from 'hierarchy' field."""
        metadata = {"hierarchy": ["Chapter 1", "Section A"]}
        result = generator._build_hierarchy_path(metadata)
        assert result == "Chapter 1 > Section A"

    def test_build_hierarchy_path_from_sections_field(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction from 'sections' field."""
        metadata = {"sections": ["Part I", "Chapter 2"]}
        result = generator._build_hierarchy_path(metadata)
        assert result == "Part I > Chapter 2"

    def test_build_hierarchy_path_from_breadcrumbs_field(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction from 'breadcrumbs' field."""
        metadata = {"breadcrumbs": ["Home", "Products", "Details"]}
        result = generator._build_hierarchy_path(metadata)
        assert result == "Home > Products > Details"

    def test_build_hierarchy_path_empty_metadata(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction with empty metadata."""
        result = generator._build_hierarchy_path({})
        assert result == ""

    def test_build_hierarchy_path_no_hierarchy_fields(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction when no hierarchy fields present."""
        metadata = {"title": "Document", "author": "John"}
        result = generator._build_hierarchy_path(metadata)
        assert result == ""

    def test_build_hierarchy_path_with_whitespace(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that hierarchy items are stripped of whitespace."""
        metadata = {"hierarchy": ["  Chapter 1  ", "  Section A  "]}
        result = generator._build_hierarchy_path(metadata)
        assert result == "Chapter 1 > Section A"

    def test_build_hierarchy_path_filters_empty_items(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty hierarchy items are filtered."""
        metadata = {"hierarchy": ["Chapter 1", "", "Section A", None]}
        result = generator._build_hierarchy_path(metadata)
        # Should only contain non-empty items
        assert "Chapter 1" in result
        assert "Section A" in result
        assert "  " not in result  # No double spaces

    def test_build_hierarchy_path_tuple_input(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test hierarchy extraction from tuple."""
        metadata = {"hierarchy": ("Part I", "Chapter 1")}
        result = generator._build_hierarchy_path(metadata)
        assert result == "Part I > Chapter 1"


class TestIncludeMetadataInHeader:
    """Test suite for _include_metadata_in_header method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_include_metadata_source_file(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test metadata inclusion with source_file field."""
        metadata = {"source_file": "guide.md"}
        result = generator._include_metadata_in_header(metadata)
        assert "Source: guide.md" in result
        assert "[" in result and "]" in result

    def test_include_metadata_source(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test metadata inclusion with source field (alternative)."""
        metadata = {"source": "document.txt"}
        result = generator._include_metadata_in_header(metadata)
        assert "Source: document.txt" in result

    def test_include_metadata_date(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test metadata inclusion with date field."""
        metadata = {"document_date": "2025-11-08"}
        result = generator._include_metadata_in_header(metadata)
        assert "Date: 2025-11-08" in result

    def test_include_metadata_author(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test metadata inclusion with author field."""
        metadata = {"author": "Jane Doe"}
        result = generator._include_metadata_in_header(metadata)
        assert "Author: Jane Doe" in result

    def test_include_metadata_multiple_fields(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test metadata inclusion with multiple fields."""
        metadata = {
            "source_file": "guide.md",
            "document_date": "2025-11-08",
            "author": "John Smith",
        }
        result = generator._include_metadata_in_header(metadata)
        assert "Source: guide.md" in result
        assert "Date: 2025-11-08" in result
        assert "Author: John Smith" in result
        assert " | " in result  # Separator between fields

    def test_include_metadata_empty_metadata(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test with empty metadata dictionary."""
        result = generator._include_metadata_in_header({})
        assert result == ""

    def test_include_metadata_none_values_ignored(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that None values are ignored."""
        metadata = {
            "source_file": "guide.md",
            "author": None,
            "document_date": None,
        }
        result = generator._include_metadata_in_header(metadata)
        assert "Source: guide.md" in result
        assert "Author:" not in result
        assert "Date:" not in result


class TestValidateHeaderFormat:
    """Test suite for validate_header_format method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_validate_header_format_valid(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test validation of valid header."""
        assert generator.validate_header_format("[Document: Test]") is True

    def test_validate_header_format_empty_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that empty header raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generator.validate_header_format("")

        with pytest.raises(ValueError, match="cannot be empty"):
            generator.validate_header_format("   ")

    def test_validate_header_format_too_long_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that header exceeding 200 chars raises ValueError."""
        long_header = "[Document: " + "a" * 200 + "]"
        with pytest.raises(ValueError, match="exceeds maximum"):
            generator.validate_header_format(long_header)

    def test_validate_header_format_no_brackets_raises(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that header without brackets raises ValueError."""
        with pytest.raises(ValueError, match="bracket pair"):
            generator.validate_header_format("Document: Test")

    def test_validate_header_format_missing_closing_bracket(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test header with only opening bracket."""
        with pytest.raises(ValueError, match="bracket pair"):
            generator.validate_header_format("[Document: Test")

    def test_validate_header_format_multiple_bracket_pairs(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test valid header with multiple bracket pairs."""
        assert (
            generator.validate_header_format("[Document: Test] [Author: Jane]")
            is True
        )

    def test_validate_header_format_exactly_200_chars(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test header with exactly 200 characters."""
        # Create header that's exactly 200 chars (including brackets and content)
        header = "[" + "a" * 198 + "]"
        assert len(header) == 200
        assert generator.validate_header_format(header) is True


class TestFormatHeaderForDisplay:
    """Test suite for format_header_for_display method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_format_header_normalizes_whitespace(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that multiple spaces are normalized to single space."""
        messy = "[Document:  Test]  [Extra:  Value]"
        clean = generator.format_header_for_display(messy)
        assert "  " not in clean  # No double spaces
        # Verify both parts are present and separated correctly
        assert "[Document: Test]" in clean
        assert "[Extra: Value]" in clean
        assert "] [" in clean  # Proper spacing between brackets

    def test_format_header_strips_edges(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that leading/trailing whitespace is removed."""
        messy = "   [Document: Test]   "
        clean = generator.format_header_for_display(messy)
        assert clean == "[Document: Test]"

    def test_format_header_fixes_bracket_spacing(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that bracket spacing is corrected."""
        messy = "[Document: Test][Author: Jane]"
        clean = generator.format_header_for_display(messy)
        assert "] [" in clean
        assert "][" not in clean

    def test_format_header_normalizes_pipe_separator(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that pipe separators have consistent spacing."""
        messy = "[A|B]  [C  |  D]"
        clean = generator.format_header_for_display(messy)
        # Should have consistent spacing around pipes
        assert clean.count(" | ") >= 1

    def test_format_header_preserves_content(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that formatting preserves essential content."""
        header = "[Document: Test] [Author: Jane]"
        clean = generator.format_header_for_display(header)
        assert "Document" in clean
        assert "Test" in clean
        assert "Author" in clean
        assert "Jane" in clean

    def test_format_header_unicode_preserved(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that unicode characters are preserved."""
        header = "[Document: ドキュメント] [Author: José]"
        clean = generator.format_header_for_display(header)
        assert "ドキュメント" in clean
        assert "José" in clean


class TestExtractContextFromPreviousChunk:
    """Test suite for extract_context_from_previous_chunk method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_extract_last_sentence(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test extraction of last sentence from chunk."""
        prev = "First sentence. Second sentence. Third sentence."
        result = generator.extract_context_from_previous_chunk(prev)
        assert "Third sentence" in result

    def test_extract_context_single_sentence(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test extraction with single sentence."""
        prev = "Only sentence here."
        result = generator.extract_context_from_previous_chunk(prev)
        assert "Only sentence" in result

    def test_extract_context_empty_chunk(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test extraction with empty previous chunk."""
        result = generator.extract_context_from_previous_chunk("")
        assert result == ""

        result = generator.extract_context_from_previous_chunk("   ")
        assert result == ""

    def test_extract_context_no_sentence_boundary(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test extraction when no sentence boundaries found."""
        prev = "Text without proper punctuation"
        result = generator.extract_context_from_previous_chunk(prev)
        # Should return something (last part or whole text)
        assert len(result) > 0
        assert "Text" in result

    def test_extract_context_question_mark_boundary(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test sentence boundary detection with question mark."""
        prev = "First? Second! Third."
        result = generator.extract_context_from_previous_chunk(prev)
        assert "Third" in result

    def test_extract_context_with_whitespace_handling(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that extracted context has whitespace stripped."""
        prev = "Sentence one.   Sentence two.   "
        result = generator.extract_context_from_previous_chunk(prev)
        # Should not have trailing whitespace
        assert result == result.strip()


class TestCalculateChunkPosition:
    """Test suite for calculate_chunk_position method."""

    @pytest.fixture
    def generator(self) -> ContextHeaderGenerator:
        """Create generator instance for tests."""
        return ContextHeaderGenerator()

    def test_calculate_chunk_position_first(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test position calculation for first chunk."""
        result = generator.calculate_chunk_position(0, 5)
        assert result == "Chunk 1 of 5 (20%)"

    def test_calculate_chunk_position_middle(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test position calculation for middle chunk."""
        result = generator.calculate_chunk_position(2, 5)
        assert result == "Chunk 3 of 5 (60%)"

    def test_calculate_chunk_position_last(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test position calculation for last chunk."""
        result = generator.calculate_chunk_position(4, 5)
        assert result == "Chunk 5 of 5 (100%)"

    def test_calculate_chunk_position_single_chunk(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test position calculation when document has single chunk."""
        result = generator.calculate_chunk_position(0, 1)
        assert result == "Chunk 1 of 1 (100%)"

    def test_calculate_chunk_position_large_document(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test position calculation for large document."""
        result = generator.calculate_chunk_position(50, 100)
        assert "Chunk 51 of 100" in result
        assert "(51%)" in result

    def test_calculate_chunk_position_invalid_index_negative(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that negative chunk index raises ValueError."""
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            generator.calculate_chunk_position(-1, 5)

    def test_calculate_chunk_position_invalid_total_zero(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that zero total_chunks raises ValueError."""
        with pytest.raises(ValueError, match="total_chunks must be > 0"):
            generator.calculate_chunk_position(0, 0)

    def test_calculate_chunk_position_invalid_total_negative(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that negative total_chunks raises ValueError."""
        with pytest.raises(ValueError, match="total_chunks must be > 0"):
            generator.calculate_chunk_position(0, -5)

    def test_calculate_chunk_position_index_out_of_bounds(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that chunk_index >= total_chunks raises ValueError."""
        with pytest.raises(ValueError, match="must be < total_chunks"):
            generator.calculate_chunk_position(5, 5)

        with pytest.raises(ValueError, match="must be < total_chunks"):
            generator.calculate_chunk_position(10, 5)

    def test_calculate_chunk_position_percentage_rounding(
        self, generator: ContextHeaderGenerator
    ) -> None:
        """Test that percentage is calculated correctly."""
        # Chunk index 0 = chunk 1: 1/3 = 33.33% should round down to 33%
        result = generator.calculate_chunk_position(0, 3)
        assert "(33%)" in result

        # Chunk index 1 = chunk 2: 2/3 = 66.66% should round down to 66%
        result = generator.calculate_chunk_position(1, 3)
        assert "(66%)" in result
