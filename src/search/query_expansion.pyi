"""Type stubs for query expansion system.

Provides type definitions for expanding queries with synonyms and related
terms to improve search coverage and relevance.
"""

from typing import Final


# Entity expansion dictionary type
ENTITY_EXPANSIONS: Final[dict[str, list[str]]]


class QueryExpander:
    """Expand queries with synonyms and related terms.

    Automatically identifies business entities in queries and expands them
    with related terms to improve search coverage and result relevance.
    """

    def __init__(self) -> None:
        """Initialize query expander with entity expansion mappings."""
        ...

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms.

        Identifies entities in the query and returns an expanded query
        using OR operators to include alternative terms.

        Format: "original OR expansion1 OR expansion2 OR ..."

        Args:
            query: Search query to expand.

        Returns:
            Expanded query string with OR-delimited alternatives.
            Returns original query if no entities matched.
        """
        ...

    def _normalize_term(self, term: str) -> str:
        """Normalize term for matching (lowercase, stripped).

        Args:
            term: Term to normalize.

        Returns:
            Normalized term in lowercase with whitespace stripped.
        """
        ...

    def _find_entity_matches(self, query: str) -> dict[str, list[str]]:
        """Find all entity matches in query and their expansions.

        Performs case-insensitive matching against entity expansion keys.

        Args:
            query: Query to search for entities.

        Returns:
            Dictionary mapping matched entities to their expansion lists.
        """
        ...

    def _deduplicate_terms(self, terms: list[str]) -> list[str]:
        """Remove duplicate terms (case-insensitive).

        Args:
            terms: List of terms that may contain duplicates.

        Returns:
            List with duplicates removed, preserving order.
        """
        ...
