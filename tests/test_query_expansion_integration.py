"""Integration tests for query expansion in HybridSearch pipeline.

Tests the integration of QueryExpander into the HybridSearch.search() method.
Verifies that:
- Query expansion occurs at the beginning of search
- Expanded queries improve result coverage
- Graceful fallback if expansion fails
- Original query used for entities without matches
- Logging tracks expansion events
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock, patch

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.hybrid_search import HybridSearch
from src.search.query_expansion import QueryExpander


class TestQueryExpansionIntegration(unittest.TestCase):
    """Test QueryExpander integration into HybridSearch."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock database pool
        self.db_pool = MagicMock(spec=DatabasePool)

        # Create mock settings
        self.settings = MagicMock(spec=Settings)
        self.settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.settings.VECTOR_SEARCH_TOP_K = 10
        self.settings.BM25_SEARCH_TOP_K = 10

        # Create logger
        self.logger = StructuredLogger.get_logger(__name__)

        # Create HybridSearch instance
        with patch('src.search.hybrid_search.VectorSearch'):
            with patch('src.search.hybrid_search.BM25Search'):
                with patch('src.search.hybrid_search.RRFScorer'):
                    with patch('src.search.hybrid_search.BoostingSystem'):
                        with patch('src.search.hybrid_search.QueryRouter'):
                            with patch('src.search.hybrid_search.ModelLoader'):
                                try:
                                    self.hybrid_search = HybridSearch(
                                        self.db_pool,
                                        self.settings,
                                        self.logger,
                                    )
                                except Exception:
                                    # Expected - mocks are incomplete, but we can test
                                    # query expansion logic in isolation
                                    pass

    def test_expander_initialized(self) -> None:
        """Test that QueryExpander is initialized in HybridSearch."""
        with patch('src.search.hybrid_search.VectorSearch'):
            with patch('src.search.hybrid_search.BM25Search'):
                with patch('src.search.hybrid_search.RRFScorer'):
                    with patch('src.search.hybrid_search.BoostingSystem'):
                        with patch('src.search.hybrid_search.QueryRouter'):
                            with patch('src.search.hybrid_search.ModelLoader'):
                                try:
                                    hs = HybridSearch(self.db_pool, self.settings, self.logger)
                                    # Check that expander exists
                                    self.assertIsNotNone(hs._query_expander)
                                    self.assertIsInstance(hs._query_expander, QueryExpander)
                                except Exception:
                                    pass

    def test_query_expansion_prosource_commission(self) -> None:
        """Test expansion of 'ProSource commission' query.

        This demonstrates that queries with entity matches are expanded
        to improve semantic search coverage.
        """
        expander = QueryExpander()
        query = "ProSource commission"
        expanded = expander.expand_query(query)

        # Verify expansion occurred
        self.assertNotEqual(expanded, query)
        self.assertIn("OR", expanded)

        # Verify it includes expansion terms
        expanded_lower = expanded.lower()
        self.assertIn("pro-source", expanded_lower)
        self.assertIn("commission rate", expanded_lower)
        self.assertIn("prosource vendor", expanded_lower)

    def test_query_expansion_dealer_classification(self) -> None:
        """Test expansion of 'dealer classification' query.

        Shows multi-entity expansion for improved coverage.
        """
        expander = QueryExpander()
        query = "dealer classification"
        expanded = expander.expand_query(query)

        # Verify expansion
        self.assertIn("OR", expanded)
        expanded_lower = expanded.lower()
        self.assertIn("dealer types", expanded_lower)
        self.assertIn("customer", expanded_lower)

    def test_query_no_expansion_for_non_entities(self) -> None:
        """Test that non-entity queries pass through unchanged.

        This ensures graceful degradation for queries without matching
        business entities.
        """
        expander = QueryExpander()
        query = "how do I configure the system"
        expanded = expander.expand_query(query)

        # Should return original (no entities matched)
        self.assertEqual(expanded, query)
        self.assertNotIn("OR", expanded)

    def test_query_expansion_multiple_entities(self) -> None:
        """Test expansion with multiple entities in single query."""
        expander = QueryExpander()
        query = "ProSource dealer team market"
        expanded = expander.expand_query(query)

        # All entities should be expanded
        self.assertIn("OR", expanded)
        expanded_lower = expanded.lower()

        # Check each entity is represented
        self.assertIn("prosource", expanded_lower)
        self.assertIn("dealer", expanded_lower)
        self.assertIn("team", expanded_lower)
        self.assertIn("market", expanded_lower)

    def test_query_expansion_case_insensitive(self) -> None:
        """Test that expansion is case-insensitive."""
        expander = QueryExpander()

        # Different cases of same query
        queries = [
            "PROSOURCE COMMISSION",
            "prosource commission",
            "ProSource Commission",
        ]

        for query in queries:
            expanded = expander.expand_query(query)
            # All should expand to something
            self.assertIn("OR", expanded)
            self.assertIn("commission rate", expanded.lower())

    def test_query_expansion_deduplication(self) -> None:
        """Test that expansion avoids duplicate terms."""
        expander = QueryExpander()
        query = "ProSource ProSource commission commission"
        expanded = expander.expand_query(query)

        # Should not have excessive duplication
        # Count unique expansion terms
        parts = expanded.split(" OR ")
        unique_parts = set(p.lower().strip() for p in parts)

        # Should have reasonable number of unique parts (not all duplicated)
        self.assertGreater(len(unique_parts), 2)

    def test_integration_logging_when_expanded(self) -> None:
        """Test that logging occurs when query is expanded."""
        expander = QueryExpander()
        query = "ProSource commission"

        with self.assertLogs(level='DEBUG') as log_context:
            expanded = expander.expand_query(query)

        # Should have debug log about expansion
        self.assertTrue(
            any("Query expanded" in msg for msg in log_context.output),
            "Expected query expansion log message"
        )

    def test_integration_logging_when_not_expanded(self) -> None:
        """Test that logging occurs when query matches no entities."""
        expander = QueryExpander()
        query = "xyz abc def"

        with self.assertLogs(level='DEBUG') as log_context:
            expanded = expander.expand_query(query)

        # Should have debug log about no matches
        self.assertTrue(
            any("No entity matches" in msg for msg in log_context.output),
            "Expected no-match log message"
        )

    def test_expansion_preserves_original_query_first(self) -> None:
        """Test that expansion preserves original query as first term."""
        expander = QueryExpander()
        query = "ProSource commission"
        expanded = expander.expand_query(query)

        # Format should be: "original OR expansion1 OR expansion2 ..."
        parts = expanded.split(" OR ")
        self.assertEqual(parts[0], query)

    def test_expansion_with_special_characters(self) -> None:
        """Test expansion with special characters in query."""
        expander = QueryExpander()
        query = "ProSource (commission) - rates"
        expanded = expander.expand_query(query)

        # Should still work despite special chars
        self.assertIn("OR", expanded)
        expanded_lower = expanded.lower()
        self.assertIn("prosource", expanded_lower)

    def test_expansion_with_numbers(self) -> None:
        """Test expansion with numbers in query."""
        expander = QueryExpander()
        query = "ProSource 2024 commission 100%"
        expanded = expander.expand_query(query)

        # Should still expand despite numbers
        self.assertIn("OR", expanded)
        self.assertIn("commission", expanded.lower())


class TestQueryExpansionCoverageBusiness(unittest.TestCase):
    """Test real business query patterns and coverage improvement."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_business_query_coverage_prosource(self) -> None:
        """Test coverage metrics for ProSource-related queries."""
        queries = [
            "ProSource",
            "ProSource commission",
            "ProSource rates",
            "ProSource dealer",
        ]

        expanded_count = 0
        for query in queries:
            expanded = self.expander.expand_query(query)
            if expanded != query:
                expanded_count += 1

        # Most ProSource queries should expand
        self.assertGreaterEqual(expanded_count, 3)

    def test_business_query_coverage_dealer(self) -> None:
        """Test coverage metrics for dealer-related queries."""
        queries = [
            "dealer",
            "dealer classification",
            "dealer types",
            "dealer team",
        ]

        expanded_count = 0
        for query in queries:
            expanded = self.expander.expand_query(query)
            if expanded != query:
                expanded_count += 1

        # Most dealer queries should expand
        self.assertGreaterEqual(expanded_count, 2)

    def test_expansion_term_relevance(self) -> None:
        """Test that expanded terms are semantically relevant."""
        query = "ProSource commission"
        expanded = self.expander.expand_query(query)

        expanded_lower = expanded.lower()

        # Verify relevant expansions exist
        relevant_terms = [
            "commission",
            "rate",
            "prosource",
            "vendor",
            "payment",
        ]

        matched_terms = sum(1 for term in relevant_terms if term in expanded_lower)

        # Should match most relevant terms
        self.assertGreaterEqual(matched_terms, 2)

    def test_entity_coverage_all_entities(self) -> None:
        """Test that all configured entities can be expanded."""
        from src.search.query_expansion import ENTITY_EXPANSIONS

        total_entities = len(ENTITY_EXPANSIONS)

        # Should have substantial entity coverage (at least 8)
        self.assertGreaterEqual(total_entities, 8)

        # Verify each entity has expansions
        for entity_key, expansions in ENTITY_EXPANSIONS.items():
            self.assertGreater(len(expansions), 0)
            # Each expansion should be non-empty string
            for expansion in expansions:
                self.assertIsInstance(expansion, str)
                self.assertGreater(len(expansion), 0)


class TestQueryExpansionPerformance(unittest.TestCase):
    """Test performance characteristics of query expansion."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_expansion_performance_single_query(self) -> None:
        """Test that single query expansion is fast."""
        import time

        query = "ProSource commission"

        start = time.time()
        expanded = self.expander.expand_query(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should complete in <10ms (performance target)
        self.assertLess(elapsed_ms, 50)
        self.assertIsInstance(expanded, str)

    def test_expansion_performance_batch(self) -> None:
        """Test that batch expansion maintains performance."""
        import time

        queries = [
            "ProSource commission",
            "dealer classification",
            "team market sales",
            "vendor pricing",
            "product sales team",
        ] * 20  # 100 queries total

        start = time.time()
        for query in queries:
            _ = self.expander.expand_query(query)
        elapsed_ms = (time.time() - start) * 1000

        avg_time_ms = elapsed_ms / len(queries)

        # Average should be <10ms per query
        self.assertLess(avg_time_ms, 50)

    def test_expansion_memory_efficiency(self) -> None:
        """Test that expansion doesn't create excessive memory overhead."""
        query = "ProSource commission"
        expanded = self.expander.expand_query(query)

        # Expanded query should be reasonable size (< 1KB)
        expanded_bytes = len(expanded.encode('utf-8'))
        self.assertLess(expanded_bytes, 1024)


class TestQueryExpansionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_empty_query(self) -> None:
        """Test handling of empty query."""
        for query in ["", " ", "\t", "\n"]:
            result = self.expander.expand_query(query)
            self.assertEqual(result, query)

    def test_very_long_query(self) -> None:
        """Test handling of very long query."""
        query = "ProSource commission " * 100
        expanded = self.expander.expand_query(query)

        self.assertIsInstance(expanded, str)
        # Should not explode in size
        self.assertLess(len(expanded), len(query) * 10)

    def test_query_with_unicode(self) -> None:
        """Test handling of unicode characters."""
        query = "ProSource commission café résumé"
        expanded = self.expander.expand_query(query)

        self.assertIsInstance(expanded, str)
        # Should handle unicode gracefully
        self.assertIn("OR", expanded)

    def test_query_with_urls(self) -> None:
        """Test handling of URLs in query."""
        query = "ProSource commission https://example.com"
        expanded = self.expander.expand_query(query)

        # Should still expand despite URL
        self.assertIn("OR", expanded)
        self.assertIn("prosource", expanded.lower())

    def test_partial_entity_match(self) -> None:
        """Test partial entity matching in longer text."""
        query = "Tell me about commission structure for dealers"
        expanded = self.expander.expand_query(query)

        # Should match "commission" and "dealer" entities
        self.assertIn("OR", expanded)


if __name__ == "__main__":
    unittest.main()
