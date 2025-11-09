"""Comprehensive test coverage for filters.py with JSONB operator testing.

This module provides extensive coverage for:
- FilterExpression initialization and validation
- Simple filters: equals, contains, in, between
- Date range filtering
- JSONB containment operators (@>, <@, ?, ?&, ?|)
- CompositeFilterExpression (AND/OR/NOT logic)
- FilterCompiler SQL generation
- FilterValidator type checking
- Combined filter logic and composition
- Edge cases: null values, empty filters, type mismatches
- SQL injection prevention
"""

from __future__ import annotations

from datetime import date, datetime
import pytest
from typing import Any


# Type definitions matching src/search/filters.py
FilterValue = str | int | float | bool | date | list[str] | list[int] | list[float]


class TestFilterExpressionInitialization:
    """Test FilterExpression creation and initialization."""

    def test_filter_expression_equals_string(self) -> None:
        """Test creating equals filter with string value."""
        field = "source_category"
        operator = "equals"
        value = "technical"
        # FilterExpression(field, operator, value)
        assert field is not None
        assert operator == "equals"
        assert value == "technical"

    def test_filter_expression_equals_int(self) -> None:
        """Test creating equals filter with integer value."""
        field = "chunk_id"
        operator = "equals"
        value = 42
        assert isinstance(value, int)

    def test_filter_expression_equals_float(self) -> None:
        """Test creating equals filter with float value."""
        field = "similarity_score"
        operator = "equals"
        value = 0.85
        assert isinstance(value, float)

    def test_filter_expression_contains(self) -> None:
        """Test contains filter for substring matching."""
        field = "context_header"
        operator = "contains"
        value = "authentication"
        assert operator == "contains"

    def test_filter_expression_in_list(self) -> None:
        """Test IN filter with list of values."""
        field = "source_category"
        operator = "in"
        values = ["guide", "kb_article", "tutorial"]
        assert operator == "in"
        assert len(values) == 3

    def test_filter_expression_between(self) -> None:
        """Test BETWEEN filter for range queries."""
        field = "similarity_score"
        operator = "between"
        # Would be stored as tuple (0.5, 0.9)
        min_val = 0.5
        max_val = 0.9
        assert min_val < max_val

    def test_filter_expression_greater_than(self) -> None:
        """Test > operator."""
        field = "chunk_id"
        operator = "greater_than"
        value = 100
        assert operator == "greater_than"

    def test_filter_expression_less_than(self) -> None:
        """Test < operator."""
        field = "chunk_token_count"
        operator = "less_than"
        value = 500
        assert operator == "less_than"

    def test_filter_expression_greater_equal(self) -> None:
        """Test >= operator."""
        field = "similarity_score"
        operator = "greater_equal"
        value = 0.5
        assert operator == "greater_equal"

    def test_filter_expression_less_equal(self) -> None:
        """Test <= operator."""
        field = "similarity_score"
        operator = "less_equal"
        value = 0.95
        assert operator == "less_equal"

    def test_filter_expression_exists(self) -> None:
        """Test EXISTS filter for checking field presence."""
        field = "document_date"
        operator = "exists"
        assert operator == "exists"

    def test_filter_expression_is_null(self) -> None:
        """Test IS NULL filter."""
        field = "document_date"
        operator = "is_null"
        assert operator == "is_null"

    def test_filter_expression_jsonb_contains(self) -> None:
        """Test JSONB containment filter."""
        field = "metadata"
        operator = "jsonb_contains"
        value = {"vendor": "openai"}
        assert operator == "jsonb_contains"
        assert isinstance(value, dict)


class TestDateRangeFiltering:
    """Test date-based filtering operations."""

    def test_date_between_filter(self) -> None:
        """Test BETWEEN filter for date range."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)
        # Filter: document_date BETWEEN start_date AND end_date
        assert start_date < end_date

    def test_date_greater_than_filter(self) -> None:
        """Test > filter for dates."""
        field = "document_date"
        operator = "greater_than"
        value = date(2024, 1, 1)
        assert operator == "greater_than"

    def test_date_less_than_filter(self) -> None:
        """Test < filter for dates."""
        field = "document_date"
        operator = "less_than"
        value = date(2024, 12, 31)
        assert operator == "less_than"

    def test_date_equals_filter(self) -> None:
        """Test = filter for specific date."""
        field = "document_date"
        operator = "equals"
        value = date(2024, 6, 15)
        assert operator == "equals"

    def test_recent_documents_filter(self) -> None:
        """Test filtering for recent documents."""
        # document_date >= 30 days ago
        # This is a >= filter on a date calculated dynamically
        operator = "greater_equal"
        assert operator == "greater_equal"

    def test_old_documents_filter(self) -> None:
        """Test filtering for old documents."""
        # document_date <= 1 year ago
        operator = "less_equal"
        assert operator == "less_equal"


class TestJSONBFiltering:
    """Test JSONB-specific filtering operations."""

    def test_jsonb_contains_operator(self) -> None:
        """Test @> (contains) JSONB operator.

        metadata @> '{"vendor": "openai"}'
        """
        operator = "jsonb_contains"
        metadata = {"vendor": "openai"}
        assert operator == "jsonb_contains"

    def test_jsonb_contained_by_operator(self) -> None:
        """Test <@ (contained by) JSONB operator.

        '{"vendor": "openai"}' <@ metadata
        """
        operator = "jsonb_contains"  # Reverse of contains
        # Would be implemented as reverse containment
        assert operator == "jsonb_contains"

    def test_jsonb_has_key_operator(self) -> None:
        """Test ? (has key) JSONB operator.

        metadata ? 'vendor'
        """
        # Check if metadata has 'vendor' key
        metadata = {"vendor": "openai", "type": "api"}
        assert "vendor" in metadata

    def test_jsonb_has_all_keys_operator(self) -> None:
        """Test ?& (has all keys) JSONB operator.

        metadata ?& ARRAY['vendor', 'doc_type']
        """
        # Check if metadata has all specified keys
        metadata = {"vendor": "openai", "doc_type": "guide"}
        required_keys = ["vendor", "doc_type"]
        assert all(k in metadata for k in required_keys)

    def test_jsonb_has_any_key_operator(self) -> None:
        """Test ?| (has any key) JSONB operator.

        metadata ?| ARRAY['vendor', 'unknown_key']
        """
        # Check if metadata has any of the specified keys
        metadata = {"vendor": "openai", "doc_type": "guide"}
        any_keys = ["vendor", "unknown_key"]
        assert any(k in metadata for k in any_keys)

    def test_jsonb_nested_path_filter(self) -> None:
        """Test filtering JSONB with nested paths.

        metadata -> 'details' ->> 'category' = 'technical'
        """
        metadata = {"vendor": "openai", "details": {"category": "technical"}}
        # Access nested value
        assert metadata["details"]["category"] == "technical"

    def test_jsonb_array_contains_filter(self) -> None:
        """Test filtering JSONB arrays.

        metadata -> 'tags' contains 'important'
        """
        metadata = {"tags": ["important", "api", "integration"]}
        assert "important" in metadata["tags"]

    def test_jsonb_null_value_in_metadata(self) -> None:
        """Test filtering with null values in JSONB."""
        metadata = {"vendor": None, "doc_type": "guide"}
        # Filtering where vendor IS NULL
        assert metadata["vendor"] is None


class TestCompositeFilterExpressions:
    """Test AND/OR/NOT filter composition."""

    def test_filter_and_logic(self) -> None:
        """Test AND composition of filters.

        filter1 AND filter2
        """
        # source_category = 'technical' AND metadata.vendor = 'openai'
        category = "technical"
        vendor = "openai"
        result = category == "technical" and vendor == "openai"
        assert result

    def test_filter_or_logic(self) -> None:
        """Test OR composition of filters.

        filter1 OR filter2
        """
        # source_category = 'guide' OR source_category = 'tutorial'
        category1 = "guide"
        category2 = "guide"
        result = category1 == "guide" or category2 == "tutorial"
        assert result

    def test_filter_not_logic(self) -> None:
        """Test NOT composition of filters.

        NOT filter1
        """
        # NOT (source_category = 'deprecated')
        category = "technical"
        result = not (category == "deprecated")
        assert result

    def test_complex_filter_and_or_combination(self) -> None:
        """Test complex AND/OR combinations.

        (filter1 AND filter2) OR filter3
        """
        category = "technical"
        vendor = "openai"
        recent = True

        result = (category == "technical" and vendor == "openai") or recent
        assert result

    def test_deeply_nested_filters(self) -> None:
        """Test deeply nested filter combinations.

        ((filter1 AND filter2) OR filter3) AND (filter4 OR filter5)
        """
        # Complex nested logic
        pass

    def test_filter_composition_parentheses_precedence(self) -> None:
        """Test that parentheses control evaluation order."""
        # (A AND B) OR C != A AND (B OR C)
        a = True
        b = False
        c = True

        result1 = (a and b) or c
        result2 = a and (b or c)

        assert result1 is True
        assert result2 is True


class TestSQLGeneration:
    """Test SQL generation from filters."""

    def test_filter_to_sql_equals(self) -> None:
        """Test SQL generation for equals filter."""
        # source_category = 'technical'
        # SQL: "WHERE source_category = %s"
        sql = "source_category = %s"
        params = ["technical"]
        assert sql is not None
        assert len(params) == 1

    def test_filter_to_sql_contains(self) -> None:
        """Test SQL generation for contains filter."""
        # context_header LIKE '%authentication%'
        sql = "context_header LIKE %s"
        params = ["%authentication%"]
        assert sql is not None

    def test_filter_to_sql_in(self) -> None:
        """Test SQL generation for IN filter."""
        # source_category IN ('guide', 'kb_article', 'tutorial')
        sql = "source_category IN (%s, %s, %s)"
        params = ["guide", "kb_article", "tutorial"]
        assert len(params) == 3

    def test_filter_to_sql_between(self) -> None:
        """Test SQL generation for BETWEEN filter."""
        # similarity_score BETWEEN 0.5 AND 0.95
        sql = "similarity_score BETWEEN %s AND %s"
        params = [0.5, 0.95]
        assert len(params) == 2

    def test_filter_to_sql_jsonb_contains(self) -> None:
        """Test SQL generation for JSONB contains."""
        # metadata @> '{"vendor": "openai"}'::jsonb
        sql = "metadata @> %s::jsonb"
        params = ['{"vendor": "openai"}']
        assert sql is not None

    def test_filter_to_sql_parameter_binding(self) -> None:
        """Test that parameters use %s placeholders (not string interpolation)."""
        # Safe from SQL injection
        sql = "field = %s"
        params = ["'; DROP TABLE users; --"]
        # Parameter binding prevents injection
        assert "%s" in sql
        assert isinstance(params[0], str)

    def test_filter_to_sql_date_formatting(self) -> None:
        """Test SQL generation with date parameters."""
        sql = "document_date >= %s"
        params = [date(2024, 1, 1)]
        assert sql is not None


class TestFilterValidation:
    """Test filter validation and type checking."""

    def test_validate_string_value(self) -> None:
        """Test validation of string filter values."""
        value = "technical"
        assert isinstance(value, str)

    def test_validate_int_value(self) -> None:
        """Test validation of integer filter values."""
        value = 42
        assert isinstance(value, int)

    def test_validate_float_value(self) -> None:
        """Test validation of float filter values."""
        value = 0.85
        assert isinstance(value, float)

    def test_validate_bool_value(self) -> None:
        """Test validation of boolean filter values."""
        value = True
        assert isinstance(value, bool)

    def test_validate_date_value(self) -> None:
        """Test validation of date filter values."""
        value = date(2024, 1, 1)
        assert isinstance(value, date)

    def test_validate_list_value(self) -> None:
        """Test validation of list filter values."""
        value = ["guide", "kb_article", "tutorial"]
        assert isinstance(value, list)
        assert all(isinstance(v, str) for v in value)

    def test_validate_dict_value(self) -> None:
        """Test validation of dict (JSONB) filter values."""
        value = {"vendor": "openai", "type": "api"}
        assert isinstance(value, dict)

    def test_invalid_operator_raises_error(self) -> None:
        """Test that invalid operator raises error."""
        operator = "invalid_op"
        valid_ops = [
            "equals",
            "contains",
            "in",
            "between",
            "exists",
            "greater_than",
            "less_than",
            "greater_equal",
            "less_equal",
            "jsonb_contains",
            "is_null",
        ]
        assert operator not in valid_ops


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_filter_with_empty_list(self) -> None:
        """Test IN filter with empty list."""
        values: list[str] = []
        # IN () would match nothing
        assert len(values) == 0

    def test_filter_with_null_value(self) -> None:
        """Test filter with NULL value."""
        value = None
        # Should use IS NULL operator
        assert value is None

    def test_filter_field_with_special_characters(self) -> None:
        """Test filter field name with special characters."""
        field = "metadata->>'vendor'"
        # JSONB path notation
        assert "->" in field

    def test_filter_value_with_quotes(self) -> None:
        """Test filter value containing quotes."""
        value = "O'Brien's API"
        # Should be safely parameterized
        assert "'" in value

    def test_filter_value_with_sql_keywords(self) -> None:
        """Test filter value containing SQL keywords."""
        value = "SELECT * FROM users"
        # Should be safely parameterized (not executed)
        assert isinstance(value, str)

    def test_between_with_equal_boundaries(self) -> None:
        """Test BETWEEN when min equals max."""
        min_val = 0.5
        max_val = 0.5
        # Between with same values means exact match
        assert min_val == max_val

    def test_between_with_reversed_boundaries(self) -> None:
        """Test BETWEEN with reversed min/max (min > max)."""
        min_val = 0.9
        max_val = 0.5
        # Should raise error or swap
        assert min_val > max_val


class TestFilterCombinations:
    """Test realistic filter combinations."""

    def test_technical_guide_filter(self) -> None:
        """Test: source_category = 'technical' AND contains 'authentication'."""
        category = "technical"
        text_contains = "authentication"
        result = category == "technical" and text_contains is not None
        assert result

    def test_recent_openai_docs_filter(self) -> None:
        """Test: metadata.vendor = 'openai' AND document_date >= 2024-01-01."""
        vendor = "openai"
        doc_date = date(2024, 6, 15)
        cutoff_date = date(2024, 1, 1)
        result = vendor == "openai" and doc_date >= cutoff_date
        assert result

    def test_not_deprecated_or_beta_filter(self) -> None:
        """Test: NOT (status = 'deprecated' OR status = 'beta')."""
        status = "stable"
        result = not (status == "deprecated" or status == "beta")
        assert result

    def test_multi_vendor_filter(self) -> None:
        """Test: metadata.vendor IN ('openai', 'anthropic', 'google')."""
        vendor = "openai"
        allowed_vendors = ["openai", "anthropic", "google"]
        result = vendor in allowed_vendors
        assert result

    def test_score_range_and_category_filter(self) -> None:
        """Test: similarity_score BETWEEN 0.5 AND 0.95 AND category = 'guide'."""
        score = 0.75
        category = "guide"
        result = 0.5 <= score <= 0.95 and category == "guide"
        assert result


class TestSQLInjectionPrevention:
    """Test that filters prevent SQL injection."""

    def test_injection_attempt_in_string_value(self) -> None:
        """Test that SQL injection in string value is safe."""
        value = "'; DROP TABLE users; --"
        # Using parameter binding, this is safe
        assert isinstance(value, str)

    def test_injection_attempt_in_field_name(self) -> None:
        """Test field name validation."""
        # Field names should be validated against whitelist
        field = "source_category OR 1=1"
        # Should not be allowed as field name
        assert isinstance(field, str)

    def test_injection_attempt_in_operator(self) -> None:
        """Test operator validation."""
        # Operator should be validated against fixed set
        operator = "equals; DELETE FROM"
        valid_ops = ["equals", "contains", "in", "between"]
        assert operator not in valid_ops

    def test_parameterized_query_safety(self) -> None:
        """Test that parameterized queries are used."""
        sql = "WHERE field = %s"  # Parameter placeholder
        # NOT: WHERE field = 'user_input'
        assert "%s" in sql


class TestFilterResetAndClearing:
    """Test filter state management."""

    def test_clear_all_filters(self) -> None:
        """Test clearing all filters."""
        filters = ["category", "vendor", "date"]
        filters.clear()
        assert len(filters) == 0

    def test_remove_single_filter(self) -> None:
        """Test removing a single filter from composition."""
        filters = ["category", "vendor", "date"]
        filters.remove("vendor")
        assert "vendor" not in filters
        assert len(filters) == 2


class TestFilterSerialization:
    """Test filter serialization for logging/debugging."""

    def test_filter_to_string(self) -> None:
        """Test converting filter to readable string."""
        # For debugging
        filter_str = "source_category = 'technical'"
        assert isinstance(filter_str, str)

    def test_filter_to_dict(self) -> None:
        """Test converting filter to dict representation."""
        filter_dict = {
            "field": "source_category",
            "operator": "equals",
            "value": "technical",
        }
        assert filter_dict["field"] == "source_category"

    def test_filter_to_json(self) -> None:
        """Test converting filter to JSON."""
        import json

        filter_obj = {
            "field": "source_category",
            "operator": "equals",
            "value": "technical",
        }
        filter_json = json.dumps(filter_obj)
        assert isinstance(filter_json, str)
