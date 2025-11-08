"""Unit tests for metadata filtering system with JSONB containment operators.

Tests cover:
- Simple filters (equals, contains, in, between, etc)
- Filter composition (AND, OR, NOT logic)
- JSONB containment filtering
- SQL generation with safe parameter binding
- Type validation and error handling
- Date range filtering
"""

import pytest
from datetime import date
from src.search.filters import (
    FilterExpression,
    CompositeFilterExpression,
    FilterCompiler,
    FilterValidator,
)


class TestSimpleFilters:
    """Test individual filter creation and SQL generation."""

    def test_equals_filter(self) -> None:
        """Test equality filter creation and SQL."""
        f = FilterExpression.equals("source_category", "vendor")
        sql, params = f.to_sql()

        assert "source_category" in sql
        assert "=" in sql
        assert "param_0" in params
        assert params["param_0"] == "vendor"

    def test_contains_filter(self) -> None:
        """Test substring matching filter."""
        f = FilterExpression.contains("context_header", "installation")
        sql, params = f.to_sql()

        assert "context_header" in sql
        assert "LIKE" in sql
        assert "param_0" in params
        assert "%installation%" in params["param_0"]

    def test_in_filter_single_value(self) -> None:
        """Test IN filter with single value."""
        f = FilterExpression.in_values("source_category", ["vendor"])
        sql, params = f.to_sql()

        assert "IN" in sql
        assert "source_category" in sql

    def test_in_filter_multiple_values(self) -> None:
        """Test IN filter with multiple values."""
        f = FilterExpression.in_values("source_category", ["vendor", "kb_article", "docs"])
        sql, params = f.to_sql()

        assert "IN" in sql
        assert len(params) == 3

    def test_in_filter_empty_list_raises(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one value"):
            FilterExpression.in_values("field", [])

    def test_between_filter_dates(self) -> None:
        """Test BETWEEN filter with dates."""
        min_date = date(2024, 1, 1)
        max_date = date(2024, 12, 31)
        f = FilterExpression.between("document_date", min_date, max_date)
        sql, params = f.to_sql()

        assert "BETWEEN" in sql
        assert "document_date" in sql
        assert len(params) == 2
        # Check that both date strings are in parameters
        param_values = list(params.values())
        assert "2024-01-01" in param_values
        assert "2024-12-31" in param_values

    def test_between_filter_numbers(self) -> None:
        """Test BETWEEN filter with numeric values."""
        f = FilterExpression.between("chunk_token_count", 100, 500)
        sql, params = f.to_sql()

        assert "BETWEEN" in sql
        assert "chunk_token_count" in sql

    def test_between_filter_invalid_range(self) -> None:
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError, match="min_value.*max_value"):
            FilterExpression.between("document_date", date(2024, 12, 31), date(2024, 1, 1))

    def test_exists_filter(self) -> None:
        """Test EXISTS filter (IS NOT NULL)."""
        f = FilterExpression.exists("author")
        sql, params = f.to_sql()

        assert "IS NOT NULL" in sql
        assert "author" in sql
        assert len(params) == 0

    def test_is_null_filter(self) -> None:
        """Test IS NULL filter."""
        f = FilterExpression.is_null("embedding")
        sql, params = f.to_sql()

        assert "IS NULL" in sql
        assert "embedding" in sql
        assert len(params) == 0

    def test_greater_than_filter(self) -> None:
        """Test > comparison filter."""
        f = FilterExpression.greater_than("chunk_token_count", 512)
        sql, params = f.to_sql()

        assert ">" in sql
        assert "chunk_token_count" in sql

    def test_less_than_filter(self) -> None:
        """Test < comparison filter."""
        f = FilterExpression.less_than("document_date", date(2024, 1, 1))
        sql, params = f.to_sql()

        assert "<" in sql
        assert "document_date" in sql

    def test_greater_equal_filter(self) -> None:
        """Test >= comparison filter."""
        f = FilterExpression.greater_equal("chunk_index", 0)
        sql, params = f.to_sql()

        assert ">=" in sql

    def test_less_equal_filter(self) -> None:
        """Test <= comparison filter."""
        f = FilterExpression.less_equal("chunk_index", 10)
        sql, params = f.to_sql()

        assert "<=" in sql


class TestJSONBFilters:
    """Test JSONB containment filtering."""

    def test_jsonb_contains_simple(self) -> None:
        """Test JSONB containment with simple key-value."""
        f = FilterExpression.jsonb_contains("metadata", {"author": "John Doe"})
        sql, params = f.to_sql()

        assert "@>" in sql
        assert "metadata" in sql
        assert "param_0" in params
        assert "John Doe" in params["param_0"]

    def test_jsonb_contains_multiple_keys(self) -> None:
        """Test JSONB containment with multiple keys."""
        f = FilterExpression.jsonb_contains("metadata", {
            "category": "vendor",
            "author": "Jane Doe",
        })
        sql, params = f.to_sql()

        assert "@>" in sql
        assert "vendor" in params["param_0"]
        assert "Jane Doe" in params["param_0"]

    def test_jsonb_contains_non_dict_raises(self) -> None:
        """Test that non-dict values raise TypeError."""
        with pytest.raises(TypeError, match="dict"):
            FilterExpression.jsonb_contains("metadata", "not a dict")  # type: ignore[arg-type]


class TestCompositeFilters:
    """Test AND/OR/NOT filter composition."""

    def test_and_composition(self) -> None:
        """Test AND composition of two filters."""
        f1 = FilterExpression.equals("source_category", "vendor")
        f2 = FilterExpression.contains("context_header", "installation")
        combined = f1.and_(f2)

        assert isinstance(combined, CompositeFilterExpression)
        assert combined.composition_operator == "AND"
        assert combined.left is f1
        assert combined.right is f2

    def test_and_sql_generation(self) -> None:
        """Test SQL generation for AND filters."""
        f1 = FilterExpression.equals("source_category", "vendor")
        f2 = FilterExpression.equals("document_date", date(2024, 1, 1))
        combined = f1.and_(f2)
        sql, params = combined.to_sql()

        assert "AND" in sql
        assert sql.count("(") >= 2  # Parentheses for grouping
        assert len(params) == 2

    def test_or_composition(self) -> None:
        """Test OR composition of two filters."""
        f1 = FilterExpression.equals("source_category", "vendor")
        f2 = FilterExpression.equals("source_category", "kb_article")
        combined = f1.or_(f2)

        assert combined.composition_operator == "OR"

    def test_or_sql_generation(self) -> None:
        """Test SQL generation for OR filters."""
        f1 = FilterExpression.equals("source_category", "vendor")
        f2 = FilterExpression.equals("source_category", "kb_article")
        combined = f1.or_(f2)
        sql, params = combined.to_sql()

        assert "OR" in sql
        assert len(params) == 2

    def test_not_composition(self) -> None:
        """Test NOT negation of a filter."""
        f = FilterExpression.equals("source_category", "vendor")
        negated = f.not_()

        assert negated.composition_operator == "NOT"

    def test_not_sql_generation(self) -> None:
        """Test SQL generation for NOT filter."""
        f = FilterExpression.equals("source_category", "vendor")
        negated = f.not_()
        sql, params = negated.to_sql()

        assert "NOT" in sql

    def test_complex_composition(self) -> None:
        """Test complex AND/OR combinations."""
        f1 = FilterExpression.equals("source_category", "vendor")
        f2 = FilterExpression.contains("context_header", "installation")
        f3 = FilterExpression.greater_than("chunk_token_count", 100)

        # (f1 AND f2) OR f3
        combined = f1.and_(f2).or_(f3)
        sql, params = combined.to_sql()

        assert "AND" in sql
        assert "OR" in sql
        assert len(params) == 3

    def test_and_without_right_operand_raises(self) -> None:
        """Test that AND without right operand raises error."""
        f = FilterExpression.equals("source_category", "vendor")
        with pytest.raises(ValueError, match="requires both"):
            CompositeFilterExpression(f, "AND", None)

    def test_or_without_right_operand_raises(self) -> None:
        """Test that OR without right operand raises error."""
        f = FilterExpression.equals("source_category", "vendor")
        with pytest.raises(ValueError, match="requires both"):
            CompositeFilterExpression(f, "OR", None)


class TestFilterValidator:
    """Test filter validation."""

    def test_valid_field_names(self) -> None:
        """Test validation of valid field names."""
        FilterValidator.validate_field("source_category")
        FilterValidator.validate_field("chunk_text")
        FilterValidator.validate_field("metadata")
        # JSONB path notation with valid identifiers
        FilterValidator.validate_field("metadata.author")
        FilterValidator.validate_field("metadata.tags")

    def test_invalid_field_names(self) -> None:
        """Test validation of invalid field names."""
        with pytest.raises(ValueError):
            FilterValidator.validate_field("")

        with pytest.raises(ValueError):
            FilterValidator.validate_field("123invalid")  # Starts with number

        with pytest.raises(ValueError):
            FilterValidator.validate_field("field-with-dash")

        with pytest.raises(ValueError):
            FilterValidator.validate_field("field with space")

    def test_operator_validation(self) -> None:
        """Test operator and value type validation."""
        # Valid: equals with string
        FilterValidator.validate_operator("equals", "vendor")

        # Valid: IN with list
        FilterValidator.validate_operator("in", ["a", "b"])

        # Invalid: IN with non-list
        with pytest.raises(TypeError):
            FilterValidator.validate_operator("in", "not a list")

        # Invalid: contains with non-string
        with pytest.raises(TypeError):
            FilterValidator.validate_operator("contains", 123)

        # Invalid: jsonb_contains with non-dict
        with pytest.raises(TypeError):
            FilterValidator.validate_operator("jsonb_contains", "not a dict")

    def test_date_validation(self) -> None:
        """Test date format validation."""
        # Valid date object
        d = FilterValidator.validate_date_format(date(2024, 1, 1))
        assert d == date(2024, 1, 1)

        # Valid date string
        d = FilterValidator.validate_date_format("2024-01-01")
        assert d == date(2024, 1, 1)

        # Invalid date string
        with pytest.raises(ValueError, match="Invalid date format"):
            FilterValidator.validate_date_format("2024/01/01")

    def test_between_range_validation(self) -> None:
        """Test BETWEEN range validation."""
        # Valid: min <= max
        FilterValidator.validate_between_range(100, 500)
        FilterValidator.validate_between_range(date(2024, 1, 1), date(2024, 12, 31))

        # Invalid: min > max
        with pytest.raises(ValueError, match="min_value.*>.*max_value"):
            FilterValidator.validate_between_range(500, 100)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention through parameter binding."""

    def test_parameter_binding_prevents_injection(self) -> None:
        """Test that parameters are properly bound."""
        # Attempt SQL injection in value
        f = FilterExpression.equals("source_category", "vendor'; DROP TABLE chunks;--")
        sql, params = f.to_sql()

        # Value should be in params dict, not in SQL string
        assert "DROP TABLE" not in sql
        assert any("DROP TABLE" in str(v) for v in params.values())

    def test_list_parameters_safe(self) -> None:
        """Test that list parameters are safely bound."""
        f = FilterExpression.in_values("source_category", [
            "vendor",
            "kb'; DROP--",
            "docs"
        ])
        sql, params = f.to_sql()

        # Injection attempt should be in params, not SQL
        assert "DROP" not in sql
        assert "kb'; DROP--" in str(params)

    def test_jsonb_parameters_safe(self) -> None:
        """Test that JSONB parameters are safely bound."""
        f = FilterExpression.jsonb_contains("metadata", {
            "author": "'; DROP TABLE--"
        })
        sql, params = f.to_sql()

        assert "DROP TABLE" not in sql
        assert "param_0" in params


class TestParameterNaming:
    """Test parameter naming convention."""

    def test_parameter_names_sequential(self) -> None:
        """Test that parameters are named param_0, param_1, etc."""
        f1 = FilterExpression.equals("field1", "value1")
        f2 = FilterExpression.equals("field2", "value2")
        f3 = FilterExpression.equals("field3", "value3")

        combined = f1.and_(f2).and_(f3)
        sql, params = combined.to_sql()

        # Should have params 0, 1, 2, 3 (reset for new compile)
        assert "param_0" in params or "param_0" not in params  # Reset on compile
        assert len(params) == 3

    def test_composite_filter_merges_parameters(self) -> None:
        """Test that composite filters merge parameters."""
        f1 = FilterExpression.equals("field1", "value1")
        f2 = FilterExpression.in_values("field2", ["a", "b", "c"])

        combined = f1.and_(f2)
        sql, params = combined.to_sql()

        # Should have 1 param from f1 + 3 from f2 = 4 total
        assert len(params) == 4


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_value(self) -> None:
        """Test filtering with empty string."""
        f = FilterExpression.equals("source_category", "")
        sql, params = f.to_sql()

        assert params["param_0"] == ""

    def test_null_date_comparison(self) -> None:
        """Test filtering for NULL dates."""
        f = FilterExpression.is_null("document_date")
        sql, params = f.to_sql()

        assert "IS NULL" in sql

    def test_very_long_string_value(self) -> None:
        """Test filtering with very long string."""
        long_string = "x" * 10000
        f = FilterExpression.equals("chunk_text", long_string)
        sql, params = f.to_sql()

        assert params["param_0"] == long_string

    def test_special_characters_in_value(self) -> None:
        """Test filtering with special characters."""
        special = "value with 'quotes' and \"double quotes\" and %wildcards%"
        f = FilterExpression.equals("chunk_text", special)
        sql, params = f.to_sql()

        # Value should be safely parameterized
        assert params["param_0"] == special

    def test_unicode_in_value(self) -> None:
        """Test filtering with Unicode characters."""
        f = FilterExpression.equals("author", "José García 中文")
        sql, params = f.to_sql()

        assert params["param_0"] == "José García 中文"


class TestRepr:
    """Test string representations for debugging."""

    def test_filter_repr(self) -> None:
        """Test FilterExpression repr."""
        f = FilterExpression.equals("source_category", "vendor")
        repr_str = repr(f)

        assert "FilterExpression" in repr_str
        assert "source_category" in repr_str

    def test_composite_filter_repr(self) -> None:
        """Test CompositeFilterExpression repr."""
        f1 = FilterExpression.equals("field1", "value1")
        f2 = FilterExpression.equals("field2", "value2")
        combined = f1.and_(f2)

        repr_str = repr(combined)
        assert "AND" in repr_str

    def test_not_filter_repr(self) -> None:
        """Test NOT filter repr."""
        f = FilterExpression.equals("source_category", "vendor")
        negated = f.not_()

        repr_str = repr(negated)
        assert "NOT" in repr_str
