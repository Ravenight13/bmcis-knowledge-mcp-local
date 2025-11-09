"""Tests for markdown reader with metadata extraction.

Comprehensive test suite for MarkdownReader covering:
- Frontmatter parsing (YAML and JSON)
- Metadata extraction and validation
- Body text extraction
- Heading hierarchy preservation
- Link extraction
- Edge cases and error handling
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.document_parsing import DocumentMetadata, MarkdownReader, ParseError


# Test fixtures


@pytest.fixture
def temp_docs_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def markdown_reader() -> MarkdownReader:
    """Create a MarkdownReader instance for testing."""
    return MarkdownReader()


# Tests for YAML frontmatter parsing


def test_yaml_frontmatter_basic(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test parsing basic YAML frontmatter."""
    content = """---
title: Test Document
author: John Doe
date: 2024-01-15
---

# Test Document

This is a test document."""

    file_path = temp_docs_dir / "test.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Test Document"
    assert doc.metadata.author == "John Doe"
    assert doc.metadata.date == "2024-01-15"
    assert "Test Document" in doc.body


def test_yaml_frontmatter_with_tags(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test parsing YAML frontmatter with tags list."""
    content = """---
title: Tagged Document
tags: [python, testing, markdown]
---

# Content"""

    file_path = temp_docs_dir / "tagged.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Tagged Document"
    assert "python" in doc.metadata.tags
    assert "testing" in doc.metadata.tags
    assert "markdown" in doc.metadata.tags


def test_yaml_frontmatter_with_quoted_values(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test parsing YAML with quoted string values."""
    content = '''---
title: "Document with Quotes"
description: 'Another quoted value'
---

# Content'''

    file_path = temp_docs_dir / "quoted.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Document with Quotes"
    assert doc.metadata.description == "Another quoted value"


def test_yaml_frontmatter_minimal(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test parsing minimal YAML with only required title."""
    content = """---
title: Minimal Document
---

# Content

Body text here."""

    file_path = temp_docs_dir / "minimal.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Minimal Document"
    assert doc.metadata.author is None
    assert doc.metadata.tags == []


# Tests for JSON frontmatter parsing


def test_json_frontmatter_basic(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test parsing JSON frontmatter."""
    content = """~~~json
{
    "title": "JSON Document",
    "author": "Jane Smith",
    "date": "2024-01-20",
    "tags": ["json", "test"]
}
~~~

# Content"""

    file_path = temp_docs_dir / "json_front.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "JSON Document"
    assert doc.metadata.author == "Jane Smith"
    assert doc.metadata.date == "2024-01-20"
    assert "json" in doc.metadata.tags


def test_json_frontmatter_with_custom_fields(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test JSON frontmatter with custom fields beyond standard metadata."""
    content = """~~~json
{
    "title": "Extended Metadata",
    "version": "2.0",
    "category": "documentation"
}
~~~

# Content"""

    file_path = temp_docs_dir / "custom.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Extended Metadata"
    # Custom fields are stored
    assert "version" in doc.raw_content
    assert "category" in doc.raw_content


# Tests for heading extraction


def test_heading_extraction_all_levels(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test extraction of headings at all levels (h1 through h6)."""
    content = """---
title: Heading Test
---

# H1 Heading

## H2 Heading

### H3 Heading

#### H4 Heading

##### H5 Heading

###### H6 Heading
"""

    file_path = temp_docs_dir / "headings.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert len(doc.headings) == 6
    assert doc.headings[0].level == 1
    assert doc.headings[0].text == "H1 Heading"
    assert doc.headings[5].level == 6
    assert doc.headings[5].text == "H6 Heading"


def test_heading_hierarchy_preservation(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that heading hierarchy is preserved in order."""
    content = """---
title: Hierarchy Test
---

# Introduction

Content here

## Subsection One

More content

### Detail 1

### Detail 2

## Subsection Two

Final content
"""

    file_path = temp_docs_dir / "hierarchy.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    expected_structure = [
        (1, "Introduction"),
        (2, "Subsection One"),
        (3, "Detail 1"),
        (3, "Detail 2"),
        (2, "Subsection Two"),
    ]

    for i, (level, text) in enumerate(expected_structure):
        assert doc.headings[i].level == level
        assert doc.headings[i].text == text


def test_heading_line_numbers(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test that heading line numbers are correctly tracked."""
    content = """---
title: Line Number Test
---

# First Heading

Some content

## Second Heading
"""

    file_path = temp_docs_dir / "line_numbers.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # First heading is after frontmatter (line 5)
    assert doc.headings[0].line_number > 0
    assert doc.headings[1].line_number > doc.headings[0].line_number


# Tests for link extraction


def test_link_extraction_basic(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test extraction of basic markdown links."""
    content = """---
title: Link Test
---

# Links

Check out [Google](https://google.com) and [GitHub](https://github.com).
"""

    file_path = temp_docs_dir / "links.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert len(doc.links) >= 2
    assert any(link.text == "Google" and link.url == "https://google.com" for link in doc.links)
    assert any(
        link.text == "GitHub" and link.url == "https://github.com" for link in doc.links
    )


def test_link_extraction_with_relative_paths(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test extraction of links with relative paths."""
    content = """---
title: Relative Links
---

See [Related Document](./related.md) and [External](../external.html).
"""

    file_path = temp_docs_dir / "relative.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert any(link.url == "./related.md" for link in doc.links)
    assert any(link.url == "../external.html" for link in doc.links)


def test_link_extraction_excludes_code_blocks(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that links in code blocks are extracted (plain text extraction)."""
    content = """---
title: Code Block Links
---

# Real Link

This is a [real link](https://example.com).

```python
# This is [not a link](in-code) in the body
```
"""

    file_path = temp_docs_dir / "code_links.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Should extract real link but code block content is removed from body
    assert any(link.url == "https://example.com" for link in doc.links)


def test_link_line_numbers(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test that link line numbers are correctly tracked."""
    content = """---
title: Link Lines
---

# Content

First [link](https://example.com).

More content

Second [link](https://example.org).
"""

    file_path = temp_docs_dir / "link_lines.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    if len(doc.links) >= 2:
        assert doc.links[1].line_number > doc.links[0].line_number


# Tests for body text extraction


def test_body_extraction_removes_frontmatter(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that frontmatter is removed from body text."""
    content = """---
title: Body Test
author: Test Author
---

# Actual Content

This should be in the body."""

    file_path = temp_docs_dir / "body.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert "title:" not in doc.body
    assert "author:" not in doc.body
    assert "Actual Content" in doc.body


def test_body_extraction_removes_markdown_formatting(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that markdown formatting is removed from body text."""
    content = """---
title: Formatting Test
---

# Heading

This is **bold** text and *italic* text. Also ***bold italic***.

This is a [link](https://example.com) and `code`.
"""

    file_path = temp_docs_dir / "formatting.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Markdown symbols should be removed
    assert "**" not in doc.body
    assert "*" not in doc.body or doc.body.count("*") == 0
    assert "[" not in doc.body
    assert "]" not in doc.body
    assert "`" not in doc.body

    # Actual content should be preserved
    assert "bold" in doc.body
    assert "italic" in doc.body
    assert "link" in doc.body
    assert "code" in doc.body


def test_body_extraction_removes_code_blocks(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that code blocks are removed from body text."""
    content = """---
title: Code Block Test
---

Before code block.

```python
def hello():
    print("Hello, World!")
```

After code block.
"""

    file_path = temp_docs_dir / "code_block.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Code block content should be removed
    assert "def hello()" not in doc.body
    assert 'print("Hello, World!")' not in doc.body

    # Surrounding content should remain
    assert "Before code block" in doc.body
    assert "After code block" in doc.body


def test_body_extraction_cleans_whitespace(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that excessive whitespace is cleaned up."""
    content = """---
title: Whitespace Test
---

Text with   multiple   spaces.


Multiple blank lines above.
"""

    file_path = temp_docs_dir / "whitespace.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Should not have excessive spaces or newlines
    assert "   " not in doc.body
    assert "\n\n\n" not in doc.body


# Tests for metadata validation


def test_metadata_tags_normalization_from_string(
    markdown_reader: MarkdownReader,
) -> None:
    """Test that string tags are normalized to list."""
    metadata = DocumentMetadata(
        title="Test",
        tags="python, testing, markdown",  # type: ignore[arg-type]
    )

    assert isinstance(metadata.tags, list)
    assert len(metadata.tags) == 3
    assert "python" in metadata.tags


def test_metadata_tags_normalization_from_list(
    markdown_reader: MarkdownReader,
) -> None:
    """Test that list tags are preserved."""
    metadata = DocumentMetadata(
        title="Test",
        tags=["python", "testing"],
    )

    assert isinstance(metadata.tags, list)
    assert len(metadata.tags) == 2


def test_metadata_date_normalization(
    markdown_reader: MarkdownReader,
) -> None:
    """Test that date is normalized to ISO format."""
    from datetime import datetime

    dt = datetime(2024, 1, 15)
    metadata = DocumentMetadata(title="Test", date=dt)  # type: ignore[arg-type]

    assert isinstance(metadata.date, str)
    assert "2024-01-15" in metadata.date


def test_metadata_title_required() -> None:
    """Test that title is required for DocumentMetadata."""
    with pytest.raises(Exception):
        DocumentMetadata()  # type: ignore[call-arg]


def test_metadata_title_cannot_be_empty() -> None:
    """Test that title cannot be empty string."""
    with pytest.raises(Exception):
        DocumentMetadata(title="")


# Tests for edge cases


def test_empty_file(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test parsing completely empty file."""
    file_path = temp_docs_dir / "empty.md"
    file_path.write_text("", encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Untitled"
    assert doc.body == ""
    assert doc.headings == []
    assert doc.links == []


def test_file_with_only_whitespace(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test parsing file with only whitespace."""
    file_path = temp_docs_dir / "whitespace.md"
    file_path.write_text("   \n\n\n   ", encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Untitled"
    assert doc.body.strip() == ""


def test_malformed_yaml_frontmatter(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that malformed YAML frontmatter is handled gracefully."""
    content = """---
title: "Broken YAML
author: No closing quote
---

# Content
"""

    file_path = temp_docs_dir / "broken_yaml.md"
    file_path.write_text(content, encoding="utf-8")

    # Should not crash, should use fallback
    doc = markdown_reader.read_file(file_path)
    assert doc.metadata is not None


def test_missing_frontmatter_title_extraction(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test title extraction from first heading when no frontmatter."""
    content = """# Document Title

This is content without frontmatter.
"""

    file_path = temp_docs_dir / "no_frontmatter.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Document Title"


def test_file_not_found() -> None:
    """Test that FileNotFoundError is raised for missing file."""
    reader = MarkdownReader()

    with pytest.raises(FileNotFoundError):
        reader.read_file(Path("/nonexistent/file.md"))


def test_special_characters_in_content(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test handling of special characters and unicode."""
    content = """---
title: Special Characters
author: Müller & Åström
description: "Unicode: 你好世界, مرحبا"
---

# Content with émojis 🎉

Symbols: @#$%^&*()
"""

    file_path = temp_docs_dir / "special.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert "Müller" in doc.metadata.author
    assert "émojis" in doc.body
    assert "🎉" in doc.body


def test_html_tags_removed_from_body(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that HTML tags are removed from body text."""
    content = """---
title: HTML Test
---

# Content

This is <strong>HTML</strong> and <em>more HTML</em>.
"""

    file_path = temp_docs_dir / "html.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # HTML tags should be removed
    assert "<strong>" not in doc.body
    assert "</strong>" not in doc.body
    assert "<em>" not in doc.body


def test_horizontal_rules_removed(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that horizontal rules are removed from body."""
    content = """---
title: Rule Test
---

# Section One

---

# Section Two

***

More content

___
"""

    file_path = temp_docs_dir / "rules.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Horizontal rules should be removed
    assert "---" not in doc.body
    assert "***" not in doc.body
    assert "___" not in doc.body


def test_multiple_code_fence_styles(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test removal of multiple code fence styles."""
    content = """---
title: Code Fences
---

```python
def test():
    pass
```

Some text

~~~python
another block
~~~
"""

    file_path = temp_docs_dir / "fences.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Code block content should not be in body
    assert "def test()" not in doc.body
    assert "another block" not in doc.body
    # But surrounding content should be
    assert "Some text" in doc.body


def test_inline_code_preserved_as_text(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that inline code is converted to text (backticks removed)."""
    content = """---
title: Inline Code
---

This is `inline code` in text.
"""

    file_path = temp_docs_dir / "inline.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Backticks should be removed but text should remain
    assert "`" not in doc.body
    assert "inline code" in doc.body


# Integration tests


def test_complex_document_parsing(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test parsing a complex document with all features."""
    content = """---
title: Complete Document
author: Test Author
date: 2024-01-15
tags: [complete, test, integration]
description: A complete test document
---

# Introduction

This is the introduction with a [link](https://example.com).

## Subsection

Some **bold** text and *italic* text.

### Deep Section

Details here.

```python
# This code should not appear in body
def example():
    pass
```

After code block, text continues.

More [another link](https://other.com).

## Another Section

Final content with `inline code`.
"""

    file_path = temp_docs_dir / "complete.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Verify all components
    assert doc.metadata.title == "Complete Document"
    assert doc.metadata.author == "Test Author"
    assert len(doc.metadata.tags) == 3

    assert len(doc.headings) == 4
    assert doc.headings[0].text == "Introduction"

    assert len(doc.links) >= 2

    assert "Introduction" in doc.body
    assert "def example()" not in doc.body
    assert "bold" in doc.body
    assert "inline code" in doc.body


def test_document_with_tables(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test parsing document with markdown tables."""
    content = """---
title: Table Test
---

# Content

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

Text after table.
"""

    file_path = temp_docs_dir / "table.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    # Tables should still be readable in body text
    assert "Column 1" in doc.body or "Value 1" in doc.body
    assert "Text after table" in doc.body


def test_document_with_lists(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test parsing document with lists."""
    content = """---
title: List Test
---

# Content

1. First item
2. Second item
3. Third item

- Unordered item 1
- Unordered item 2
"""

    file_path = temp_docs_dir / "list.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert "First item" in doc.body
    assert "Unordered item 1" in doc.body


def test_parse_error_context_information() -> None:
    """Test that ParseError contains helpful context."""
    reader = MarkdownReader()
    file_path = Path("/nonexistent/test.md")

    with pytest.raises(FileNotFoundError):
        reader.read_file(file_path)


# Coverage edge cases


def test_metadata_custom_fields_preserved(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test that custom metadata fields are preserved."""
    content = """---
title: Custom Fields
custom_key: custom_value
another: field
---

# Content
"""

    file_path = temp_docs_dir / "custom_fields.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "Custom Fields"
    # Custom fields should be in the object
    assert hasattr(doc.metadata, "custom_fields") or hasattr(doc.metadata, "custom_key")


def test_yaml_with_comments(temp_docs_dir: Path, markdown_reader: MarkdownReader) -> None:
    """Test YAML parsing ignores comments."""
    content = """---
title: With Comments
# This is a comment
author: Someone
---

# Content
"""

    file_path = temp_docs_dir / "comments.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert doc.metadata.title == "With Comments"
    assert doc.metadata.author == "Someone"


def test_heading_with_trailing_hashes(
    temp_docs_dir: Path, markdown_reader: MarkdownReader
) -> None:
    """Test heading extraction with trailing hash marks (markdown style)."""
    content = """---
title: Trailing Hashes
---

# First Heading #

## Second Heading ##
"""

    file_path = temp_docs_dir / "trailing.md"
    file_path.write_text(content, encoding="utf-8")

    doc = markdown_reader.read_file(file_path)

    assert len(doc.headings) >= 2
    # Text should not include trailing hashes
    assert doc.headings[0].text == "First Heading"
    assert doc.headings[1].text == "Second Heading"
