"""Type stubs for metadata filtering system with JSONB containment operators.

Provides type-safe metadata filtering for search queries using JSONB
containment operators with support for complex filter composition.

Type definitions for:
- FilterValue: Union of supported filter value types
- FilterExpression: Base filter expression class
- SimpleFilter: Single-field filters
- CompositeFilter: AND/OR/NOT compositions
- FilterCompiler: SQL generation from filter expressions
"""

from datetime import date
from typing import Any, Literal, TypeVar

# Type variable for different field types
T = TypeVar("T", str, int, float, bool, date, list[str], list[int], list[float])

# Supported filter value types
FilterValue = str | int | float | bool | date | list[str] | list[int] | list[float]

# Supported operators for filtering
FilterOperator = Literal[
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

# JSONB operators for PostgreSQL
JSONBOperator = Literal["@>", "<@", "?", "?&", "?|"]

# Filter composition operators
CompositionOperator = Literal["AND", "OR", "NOT"]

class FilterExpression:
    """Base class for filter expressions with support for composition.

    Represents a single filter condition that can be combined with other
    filters using AND/OR/NOT logic. Provides type-safe filter building
    with validation and SQL compilation.

    Attributes:
        field: The field name being filtered (supports dotted paths for JSONB).
        operator: The comparison operator to use.
        value: The value to compare against.
        is_jsonb: Whether this field is in JSONB metadata.
    """

    field: str
    operator: FilterOperator
    value: FilterValue | dict[str, Any]
    is_jsonb: bool

    def __init__(
        self,
        field: str,
        operator: FilterOperator,
        value: FilterValue | dict[str, Any],
        is_jsonb: bool = False,
    ) -> None:
        """Initialize a filter expression.

        Args:
            field: Field name to filter on.
            operator: Comparison operator.
            value: Value to compare against.
            is_jsonb: Whether field is in JSONB column.
        """
        ...

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """Convert filter to SQL WHERE clause with parameters.

        Returns:
            Tuple of (sql_where_clause, parameters_dict) for safe parameter binding.

        Raises:
            ValueError: If filter expression is invalid or incomplete.
        """
        ...

    def and_(self, other: "FilterExpression") -> "CompositeFilterExpression":
        """Combine with another filter using AND logic.

        Args:
            other: Another FilterExpression to combine.

        Returns:
            New CompositeFilterExpression with AND operator.
        """
        ...

    def or_(self, other: "FilterExpression") -> "CompositeFilterExpression":
        """Combine with another filter using OR logic.

        Args:
            other: Another FilterExpression to combine.

        Returns:
            New CompositeFilterExpression with OR operator.
        """
        ...

    def not_(self) -> "CompositeFilterExpression":
        """Negate this filter using NOT logic.

        Returns:
            New CompositeFilterExpression with NOT operator.
        """
        ...

    @staticmethod
    def equals(field: str, value: FilterValue, is_jsonb: bool = False) -> "FilterExpression":
        """Create equals filter (field = value).

        Args:
            field: Field name.
            value: Value to match exactly.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for equality comparison.
        """
        ...

    @staticmethod
    def contains(
        field: str, substring: str, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create contains filter (field LIKE '%value%').

        Args:
            field: Field name.
            substring: Substring to search for.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for substring matching.
        """
        ...

    @staticmethod
    def in_values(
        field: str, values: list[FilterValue], is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create IN filter (field IN (value1, value2, ...)).

        Args:
            field: Field name.
            values: List of values to match against.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for membership test.
        """
        ...

    @staticmethod
    def between(
        field: str, min_value: FilterValue, max_value: FilterValue, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create BETWEEN filter (field BETWEEN min AND max).

        Args:
            field: Field name.
            min_value: Minimum value (inclusive).
            max_value: Maximum value (inclusive).
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for range filtering.

        Raises:
            ValueError: If min_value > max_value.
        """
        ...

    @staticmethod
    def exists(field: str, is_jsonb: bool = True) -> "FilterExpression":
        """Create EXISTS filter (field IS NOT NULL).

        Args:
            field: Field name.
            is_jsonb: Whether field is in JSONB (default True).

        Returns:
            FilterExpression for null checking.
        """
        ...

    @staticmethod
    def is_null(field: str, is_jsonb: bool = False) -> "FilterExpression":
        """Create IS NULL filter (field IS NULL).

        Args:
            field: Field name.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for null checking.
        """
        ...

    @staticmethod
    def greater_than(
        field: str, value: FilterValue, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create > comparison filter.

        Args:
            field: Field name.
            value: Comparison value.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for greater-than comparison.
        """
        ...

    @staticmethod
    def less_than(
        field: str, value: FilterValue, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create < comparison filter.

        Args:
            field: Field name.
            value: Comparison value.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for less-than comparison.
        """
        ...

    @staticmethod
    def greater_equal(
        field: str, value: FilterValue, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create >= comparison filter.

        Args:
            field: Field name.
            value: Comparison value.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for greater-than-or-equal comparison.
        """
        ...

    @staticmethod
    def less_equal(
        field: str, value: FilterValue, is_jsonb: bool = False
    ) -> "FilterExpression":
        """Create <= comparison filter.

        Args:
            field: Field name.
            value: Comparison value.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for less-than-or-equal comparison.
        """
        ...

    @staticmethod
    def jsonb_contains(field: str, json_value: dict[str, Any]) -> "FilterExpression":
        """Create JSONB containment filter (metadata @> '{...}').

        Checks if JSONB column contains the specified key-value pairs using
        PostgreSQL's @> operator for efficient JSONB matching.

        Args:
            field: JSONB field name (usually 'metadata').
            json_value: Dictionary of key-value pairs to match.

        Returns:
            FilterExpression for JSONB containment.

        Raises:
            TypeError: If json_value is not a dict.
        """
        ...

class CompositeFilterExpression(FilterExpression):
    """Composite filter expression for AND/OR/NOT combinations.

    Represents a combination of multiple filter expressions using logical
    operators (AND, OR, NOT). Supports arbitrary nesting depth and complex
    filter hierarchies.

    Attributes:
        left: Left-hand filter expression (required).
        right: Right-hand filter expression (required for AND/OR, None for NOT).
        operator: Logical operator (AND, OR, NOT).
    """

    left: FilterExpression
    right: FilterExpression | None
    operator: CompositionOperator

    def __init__(
        self,
        left: FilterExpression,
        operator: CompositionOperator,
        right: FilterExpression | None = None,
    ) -> None:
        """Initialize a composite filter expression.

        Args:
            left: Left filter expression.
            operator: Composition operator (AND, OR, NOT).
            right: Right filter expression (required for AND/OR).

        Raises:
            ValueError: If right is None for AND/OR operators.
        """
        ...

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """Recursively compile composite filter to SQL.

        Returns:
            Tuple of (sql_where_clause, merged_parameters).
        """
        ...

class FilterCompiler:
    """Compiler for converting filter expressions to SQL with parameter binding.

    Handles type-safe conversion of FilterExpression objects to parameterized
    SQL WHERE clauses with support for:
    - JSONB containment operators (@>, <@, ?, ?&, ?|)
    - Direct column comparisons
    - Nested metadata filtering
    - Complex compositions (AND/OR/NOT)

    Type-safe parameter binding prevents SQL injection while maintaining
    query efficiency through proper index usage.
    """

    @staticmethod
    def compile(filter_expr: FilterExpression) -> tuple[str, dict[str, Any]]:
        """Compile filter expression to parameterized SQL WHERE clause.

        Recursively compiles filter expressions (including composites) to
        SQL with proper parameter binding for all values.

        Args:
            filter_expr: FilterExpression to compile.

        Returns:
            Tuple of (sql_where_clause, parameters_dict).

        Raises:
            ValueError: If filter expression is invalid.
            TypeError: If unsupported types encountered.
        """
        ...

    @staticmethod
    def compile_simple(
        field: str, operator: FilterOperator, value: FilterValue | dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Compile simple single-field filter to SQL.

        Args:
            field: Field name.
            operator: Filter operator.
            value: Filter value.

        Returns:
            Tuple of (sql_condition, parameters).
        """
        ...

    @staticmethod
    def compile_composite(
        left: FilterExpression,
        operator: CompositionOperator,
        right: FilterExpression | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Compile composite filter (AND/OR/NOT) to SQL.

        Args:
            left: Left filter expression.
            operator: AND, OR, or NOT.
            right: Right filter expression (None for NOT).

        Returns:
            Tuple of (sql_condition, merged_parameters).
        """
        ...

class FilterValidator:
    """Validates filter expressions before SQL compilation.

    Performs type checking, field validation, and operator compatibility
    checks to ensure filters are safe and correct before execution.
    """

    @staticmethod
    def validate_field(field: str) -> None:
        """Validate field name (prevent SQL injection).

        Args:
            field: Field name to validate.

        Raises:
            ValueError: If field name contains invalid characters.
        """
        ...

    @staticmethod
    def validate_operator(
        operator: FilterOperator, value: FilterValue | dict[str, Any]
    ) -> None:
        """Validate operator and value type compatibility.

        Args:
            operator: Filter operator.
            value: Filter value.

        Raises:
            TypeError: If value type incompatible with operator.
            ValueError: If operator is unsupported.
        """
        ...

    @staticmethod
    def validate_date_format(date_value: str | date) -> date:
        """Validate and parse date values.

        Args:
            date_value: Date string or date object.

        Returns:
            Validated date object.

        Raises:
            ValueError: If date format invalid.
        """
        ...

    @staticmethod
    def validate_between_range(min_val: FilterValue, max_val: FilterValue) -> None:
        """Validate min <= max for BETWEEN filters.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.

        Raises:
            ValueError: If min > max.
        """
        ...
