"""Query expansion system for improving search coverage.

Expands queries with synonyms and related terms to improve search coverage
and result relevance. Identifies business entities and provides semantic
alternatives.

Example:
    >>> expander = QueryExpander()
    >>> expanded = expander.expand_query("ProSource commission rates")
    >>> print(expanded)
    ProSource commission rates OR pro-source OR prosource vendor OR commission rate OR commission structure OR payment
"""

from __future__ import annotations

import logging
from typing import Final

from src.core.logging import StructuredLogger

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Entity expansion mappings for business domain
ENTITY_EXPANSIONS: Final[dict[str, list[str]]] = {
    "prosource": [
        "ProSource",
        "pro-source",
        "prosource vendor",
    ],
    "commission": [
        "commission",
        "commission rate",
        "commission structure",
        "payment",
    ],
    "dealer": [
        "dealer",
        "dealer types",
        "dealer classification",
        "customer",
    ],
    "team": [
        "team",
        "sales team",
        "organization",
        "district",
    ],
    "lutron": [
        "Lutron",
        "lutron control",
        "lighting control",
    ],
    "market": [
        "market",
        "market segment",
        "market data",
        "region",
    ],
    "sales": [
        "sales",
        "revenue",
        "growth",
        "performance",
    ],
    "vendor": [
        "vendor",
        "supplier",
        "partner",
        "manufacturer",
    ],
    "price": [
        "price",
        "pricing",
        "cost",
        "value",
    ],
    "product": [
        "product",
        "product line",
        "offering",
        "solution",
    ],
}


class QueryExpander:
    """Expand queries with synonyms and related terms.

    Automatically identifies business entities in queries and expands them
    with related terms to improve search coverage and result relevance.

    The expander uses a predefined entity expansion dictionary mapping
    normalized entity names to lists of related terms.

    Performance target: <10ms per expansion
    """

    def __init__(self) -> None:
        """Initialize query expander with entity expansion mappings."""
        self._expansions = ENTITY_EXPANSIONS
        logger.debug(
            "QueryExpander initialized",
            extra={"entity_count": len(self._expansions)},
        )

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms.

        Identifies entities in the query and returns an expanded query
        using OR operators to include alternative terms. Avoids duplicating
        the original query text.

        Format: "original OR expansion1 OR expansion2 OR ..."

        Args:
            query: Search query to expand.

        Returns:
            Expanded query string with OR-delimited alternatives.
            Returns original query if no entities matched.

        Example:
            >>> expander = QueryExpander()
            >>> result = expander.expand_query("ProSource commission")
            >>> # result contains "ProSource commission OR pro-source OR prosource vendor ..."
        """
        if not query or not query.strip():
            return query

        # Find all entity matches and their expansions
        matches: dict[str, list[str]] = self._find_entity_matches(query)

        if not matches:
            # No entities found, return original query
            logger.debug("No entity matches found in query", extra={"query": query})
            return query

        # Collect all expansion terms
        all_expansion_terms: list[str] = []

        for entity_key, expansions in matches.items():
            # Add all expansions for this entity
            all_expansion_terms.extend(expansions)

        # Remove duplicates (case-insensitive)
        unique_expansions: list[str] = self._deduplicate_terms(
            all_expansion_terms
        )

        # Build expanded query: original OR expansion1 OR expansion2 ...
        expanded_query = f"{query} OR {' OR '.join(unique_expansions)}"

        logger.debug(
            "Query expanded",
            extra={
                "original": query,
                "entities_found": len(matches),
                "expansions_added": len(unique_expansions),
            },
        )

        return expanded_query

    def _normalize_term(self, term: str) -> str:
        """Normalize term for matching (lowercase, stripped).

        Args:
            term: Term to normalize.

        Returns:
            Normalized term in lowercase with whitespace stripped.
        """
        return term.lower().strip()

    def _find_entity_matches(self, query: str) -> dict[str, list[str]]:
        """Find all entity matches in query and their expansions.

        Performs case-insensitive matching against entity expansion keys.
        Matches are found by checking if normalized query contains
        normalized entity keys.

        Args:
            query: Query to search for entities.

        Returns:
            Dictionary mapping matched entity keys to their expansion lists.
            Returns empty dict if no matches found.

        Example:
            >>> expander = QueryExpander()
            >>> matches = expander._find_entity_matches("ProSource commission rates")
            >>> # Returns {"prosource": [...], "commission": [...]}
        """
        query_normalized = self._normalize_term(query)
        matches: dict[str, list[str]] = {}

        # Check each entity key
        for entity_key, expansions in self._expansions.items():
            # Case-insensitive match
            if entity_key in query_normalized:
                matches[entity_key] = expansions

        return matches

    def _deduplicate_terms(self, terms: list[str]) -> list[str]:
        """Remove duplicate terms (case-insensitive).

        Preserves order of first occurrence while removing subsequent
        duplicates based on normalized (lowercase) comparison.

        Args:
            terms: List of terms that may contain duplicates.

        Returns:
            List with duplicates removed, preserving order of first occurrence.

        Example:
            >>> expander = QueryExpander()
            >>> result = expander._deduplicate_terms(
            ...     ["ProSource", "pro-source", "Commission", "commission"]
            ... )
            >>> len(result)
            2
        """
        seen_normalized: set[str] = set()
        unique_terms: list[str] = []

        for term in terms:
            normalized = self._normalize_term(term)
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_terms.append(term)

        return unique_terms
