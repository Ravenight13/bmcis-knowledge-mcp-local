"""Metadata filtering system with JSONB containment operators.

Provides type-safe metadata filtering for search queries using JSONB
containment operators with support for complex filter composition and
SQL injection prevention through parameter binding.

Architecture:
- FilterExpression: Base class for single filters
- CompositeFilterExpression: AND/OR/NOT compositions
- FilterCompiler: Converts filters to parameterized SQL
- FilterValidator: Type checking and validation

Usage:
    # Simple filters
    filter1 = FilterExpression.equals("source_category", "vendor")
    filter2 = FilterExpression.contains("context_header", "installation")

    # Composite filters
    combined = filter1.and_(filter2)
    range_filter = FilterExpression.between("document_date", date(2024, 1, 1), date(2024, 12, 31))

    # JSONB filtering
    metadata_filter = FilterExpression.jsonb_contains("metadata", {"author": "John Doe"})

    # To SQL
    sql, params = combined.to_sql()
"""

import json
import logging
import re
from datetime import date, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Type definitions
FilterValue = str | int | float | bool | date | list[str] | list[int] | list[float]
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
CompositionOperator = Literal["AND", "OR", "NOT"]
JSONBOperator = Literal["@>", "<@", "?", "?&", "?|"]


class FilterExpression:
    """Base class for filter expressions with type-safe filter building.

    Represents a single filter condition that can be combined with other
    filters using AND/OR/NOT logic. Provides static factory methods for
    common filter types (equals, contains, in, between, etc).
    """

    __slots__ = ("field", "operator", "value", "is_jsonb")

    def __init__(
        self,
        field: str,
        operator: FilterOperator,
        value: FilterValue | dict[str, Any] | None = None,
        is_jsonb: bool = False,
    ) -> None:
        """Initialize a filter expression.

        Args:
            field: Field name to filter on (supports dotted paths for JSONB).
            operator: Comparison operator.
            value: Value to compare against.
            is_jsonb: Whether field is in JSONB metadata column.

        Raises:
            ValueError: If field name contains invalid characters.
        """
        FilterValidator.validate_field(field)
        FilterValidator.validate_operator(operator, value)

        self.field: str = field
        self.operator: FilterOperator = operator
        self.value: FilterValue | dict[str, Any] | None = value
        self.is_jsonb: bool = is_jsonb

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """Convert filter to SQL WHERE clause with parameters.

        Returns:
            Tuple of (sql_where_clause, parameters_dict).
            Parameters are keyed as 'param_N' for safe binding.

        Raises:
            ValueError: If filter expression is invalid.
        """
        return FilterCompiler.compile(self)

    def and_(self, other: "FilterExpression") -> "CompositeFilterExpression":
        """Combine with another filter using AND logic.

        Args:
            other: Another FilterExpression to combine.

        Returns:
            New CompositeFilterExpression with AND operator.
        """
        return CompositeFilterExpression(self, "AND", other)

    def or_(self, other: "FilterExpression") -> "CompositeFilterExpression":
        """Combine with another filter using OR logic.

        Args:
            other: Another FilterExpression to combine.

        Returns:
            New CompositeFilterExpression with OR operator.
        """
        return CompositeFilterExpression(self, "OR", other)

    def not_(self) -> "CompositeFilterExpression":
        """Negate this filter using NOT logic.

        Returns:
            New CompositeFilterExpression with NOT operator.
        """
        return CompositeFilterExpression(self, "NOT", None)

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
        return FilterExpression(field, "equals", value, is_jsonb)

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
        return FilterExpression(field, "contains", substring, is_jsonb)

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

        Raises:
            ValueError: If values list is empty.
        """
        if not values:
            raise ValueError("in_values requires at least one value")
        return FilterExpression(field, "in", values, is_jsonb)  # type: ignore[arg-type]

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
        FilterValidator.validate_between_range(min_value, max_value)
        # Store as tuple (min, max)
        return FilterExpression(field, "between", (min_value, max_value), is_jsonb)  # type: ignore[arg-type]

    @staticmethod
    def exists(field: str, is_jsonb: bool = True) -> "FilterExpression":
        """Create EXISTS filter (field IS NOT NULL).

        Args:
            field: Field name.
            is_jsonb: Whether field is in JSONB (default True).

        Returns:
            FilterExpression for null checking.
        """
        return FilterExpression(field, "exists", None, is_jsonb)

    @staticmethod
    def is_null(field: str, is_jsonb: bool = False) -> "FilterExpression":
        """Create IS NULL filter (field IS NULL).

        Args:
            field: Field name.
            is_jsonb: Whether field is in JSONB.

        Returns:
            FilterExpression for null checking.
        """
        return FilterExpression(field, "is_null", None, is_jsonb)

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
        return FilterExpression(field, "greater_than", value, is_jsonb)

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
        return FilterExpression(field, "less_than", value, is_jsonb)

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
        return FilterExpression(field, "greater_equal", value, is_jsonb)

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
        return FilterExpression(field, "less_equal", value, is_jsonb)

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
        if not isinstance(json_value, dict):
            raise TypeError(f"jsonb_contains requires dict, got {type(json_value)}")
        return FilterExpression(field, "jsonb_contains", json_value, is_jsonb=True)

    def __repr__(self) -> str:
        """Return string representation of filter for debugging."""
        return f"FilterExpression({self.field} {self.operator} {self.value!r})"


class CompositeFilterExpression(FilterExpression):
    """Composite filter expression for AND/OR/NOT combinations.

    Represents a combination of multiple filter expressions using logical
    operators. Supports arbitrary nesting depth for complex hierarchies.
    """

    __slots__ = ("left", "right", "composition_operator")

    def __init__(
        self,
        left: FilterExpression,
        composition_operator: CompositionOperator,
        right: FilterExpression | None = None,
    ) -> None:
        """Initialize a composite filter expression.

        Args:
            left: Left filter expression.
            composition_operator: Composition operator (AND, OR, NOT).
            right: Right filter expression (required for AND/OR).

        Raises:
            ValueError: If right is None for AND/OR operators.
        """
        if composition_operator in ("AND", "OR") and right is None:
            raise ValueError(f"{composition_operator} requires both left and right operands")

        self.left: FilterExpression = left
        self.right: FilterExpression | None = right
        self.composition_operator: CompositionOperator = composition_operator

        # Set base class attributes for compatibility
        self.field = f"({left.field} {composition_operator} {right.field if right else ''})"
        # Composite filters reassign operator from parent class
        self.operator: Any = "equals"
        self.value = None
        self.is_jsonb = False

    def to_sql(self) -> tuple[str, dict[str, Any]]:
        """Recursively compile composite filter to SQL.

        Returns:
            Tuple of (sql_where_clause, merged_parameters).
        """
        return FilterCompiler.compile(self)

    def __repr__(self) -> str:
        """Return string representation of composite filter."""
        if self.composition_operator == "NOT":
            return f"NOT({self.left!r})"
        return f"({self.left!r} {self.composition_operator} {self.right!r})"


class FilterCompiler:
    """Compiler for converting filter expressions to SQL with parameter binding.

    Handles type-safe conversion of FilterExpression objects to parameterized
    SQL WHERE clauses with support for:
    - JSONB containment operators (@>, <@, ?, ?&, ?|)
    - Direct column comparisons
    - Nested metadata filtering
    - Complex compositions (AND/OR/NOT)
    """

    _param_counter: int = 0

    @classmethod
    def compile(cls, filter_expr: FilterExpression) -> tuple[str, dict[str, Any]]:
        """Compile filter expression to parameterized SQL WHERE clause.

        Recursively compiles filter expressions (including composites) to
        SQL with proper parameter binding for all values.

        Args:
            filter_expr: FilterExpression to compile.

        Returns:
            Tuple of (sql_where_clause, parameters_dict).

        Raises:
            ValueError: If filter expression is invalid.
        """
        cls._param_counter = 0
        return cls._compile_internal(filter_expr)

    @classmethod
    def _compile_internal(cls, filter_expr: FilterExpression) -> tuple[str, dict[str, Any]]:
        """Internal recursive compilation method."""
        if isinstance(filter_expr, CompositeFilterExpression):
            return cls._compile_composite(filter_expr)
        return cls._compile_simple(filter_expr)

    @classmethod
    def _compile_simple(cls, filter_expr: FilterExpression) -> tuple[str, dict[str, Any]]:
        """Compile simple single-field filter to SQL."""
        field = filter_expr.field
        operator = filter_expr.operator
        value = filter_expr.value

        param_name = f"param_{cls._param_counter}"
        cls._param_counter += 1

        params: dict[str, Any] = {}

        # Handle NULL checks (no parameter needed)
        if operator == "is_null":
            return f"{field} IS NULL", {}
        if operator == "exists":
            return f"{field} IS NOT NULL", {}

        # JSONB containment operator
        if operator == "jsonb_contains":
            if not isinstance(value, dict):
                raise ValueError(f"jsonb_contains requires dict value")
            params[param_name] = json.dumps(value)
            return f"{field} @> %({param_name})s::jsonb", params

        # BETWEEN operator
        if operator == "between":
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("between requires tuple of (min, max)")
            min_val, max_val = value
            param_min = f"param_{cls._param_counter}"
            cls._param_counter += 1
            param_max = f"param_{cls._param_counter}"
            cls._param_counter += 1
            params[param_min] = cls._convert_value(min_val)
            params[param_max] = cls._convert_value(max_val)
            return f"{field} BETWEEN %({param_min})s AND %({param_max})s", params

        # IN operator
        if operator == "in":
            if not isinstance(value, list):
                raise ValueError("in requires list value")
            placeholders = []
            for i, v in enumerate(value):
                param = f"param_{cls._param_counter}"
                cls._param_counter += 1
                params[param] = cls._convert_value(v)
                placeholders.append(f"%({param})s")
            return f"{field} IN ({', '.join(placeholders)})", params

        # LIKE operator (contains)
        if operator == "contains":
            params[param_name] = f"%{value}%"
            return f"{field} LIKE %({param_name})s", params

        # Comparison operators
        op_map: dict[FilterOperator, str] = {
            "equals": "=",
            "greater_than": ">",
            "less_than": "<",
            "greater_equal": ">=",
            "less_equal": "<=",
        }

        if operator in op_map:
            params[param_name] = cls._convert_value(value)
            sql_op = op_map[operator]
            return f"{field} {sql_op} %({param_name})s", params

        raise ValueError(f"Unsupported operator: {operator}")

    @classmethod
    def _compile_composite(cls, composite: CompositeFilterExpression) -> tuple[str, dict[str, Any]]:
        """Compile composite filter (AND/OR/NOT) to SQL."""
        operator = composite.composition_operator

        if operator == "NOT":
            sql, params = cls._compile_internal(composite.left)
            return f"NOT ({sql})", params

        # AND or OR
        if composite.right is None:
            raise ValueError(f"{operator} requires both operands")

        left_sql, left_params = cls._compile_internal(composite.left)
        right_sql, right_params = cls._compile_internal(composite.right)

        # Merge parameters
        merged_params = {**left_params, **right_params}

        return f"({left_sql}) {operator} ({right_sql})", merged_params

    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert Python value to SQL-compatible type.

        Handles type conversion for parameters including dates, lists, etc.
        """
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, list):
            return value
        return value


class FilterValidator:
    """Validates filter expressions before SQL compilation.

    Performs type checking, field validation, and operator compatibility
    checks to ensure filters are safe and correct before execution.
    """

    # Field name pattern: alphanumeric, underscore, dot for JSONB paths
    _FIELD_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")

    # List of valid table columns (PostgreSQL table schema)
    _VALID_COLUMNS = {
        # knowledge_base table columns
        "chunk_text",
        "chunk_hash",
        "context_header",
        "source_file",
        "source_category",
        "document_date",
        "chunk_index",
        "total_chunks",
        "chunk_token_count",
        "metadata",
        "embedding",
        "created_at",
        "updated_at",
    }

    @staticmethod
    def validate_field(field: str) -> None:
        """Validate field name (prevent SQL injection).

        Args:
            field: Field name to validate.

        Raises:
            ValueError: If field name contains invalid characters.
        """
        if not field:
            raise ValueError("Field name cannot be empty")

        if not FilterValidator._FIELD_PATTERN.match(field):
            raise ValueError(f"Invalid field name: {field}")

    @staticmethod
    def validate_operator(operator: FilterOperator, value: Any) -> None:
        """Validate operator and value type compatibility.

        Args:
            operator: Filter operator.
            value: Filter value.

        Raises:
            TypeError: If value type incompatible with operator.
            ValueError: If operator is unsupported.
        """
        valid_operators: set[FilterOperator] = {
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
        }

        if operator not in valid_operators:
            raise ValueError(f"Unsupported operator: {operator}")

        # NULL checks don't require values
        if operator in ("is_null", "exists"):
            return

        # JSONB containment requires dict
        if operator == "jsonb_contains":
            if not isinstance(value, dict):
                raise TypeError(f"jsonb_contains requires dict, got {type(value)}")
            return

        # IN requires list
        if operator == "in":
            if not isinstance(value, list):
                raise TypeError(f"in operator requires list, got {type(value)}")
            if not value:
                raise ValueError("in operator requires non-empty list")
            return

        # BETWEEN requires tuple
        if operator == "between":
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("between operator requires tuple of (min, max)")
            return

        # Contains requires string
        if operator == "contains":
            if not isinstance(value, str):
                raise TypeError(f"contains requires string, got {type(value)}")
            return

    @staticmethod
    def validate_date_format(date_value: str | date) -> date:
        """Validate and parse date values.

        Args:
            date_value: Date string (YYYY-MM-DD) or date object.

        Returns:
            Validated date object.

        Raises:
            ValueError: If date format invalid.
        """
        if isinstance(date_value, date):
            return date_value

        try:
            return date.fromisoformat(date_value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date format: {date_value}. Expected YYYY-MM-DD") from e

    @staticmethod
    def validate_between_range(min_val: FilterValue, max_val: FilterValue) -> None:
        """Validate min <= max for BETWEEN filters.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.

        Raises:
            ValueError: If min > max.
        """
        if isinstance(min_val, date) and isinstance(max_val, date):
            if min_val > max_val:
                raise ValueError(f"min_value ({min_val}) > max_value ({max_val})")
        elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if min_val > max_val:
                raise ValueError(f"min_value ({min_val}) > max_value ({max_val})")
