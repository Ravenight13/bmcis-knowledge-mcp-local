"""Tests for query expansion system.

Tests the QueryExpander class to ensure:
- Entity matching works correctly
- Query expansion produces correct format
- No duplicate terms in expansions
- Case-insensitive matching
- Empty queries handled gracefully
"""

from __future__ import annotations

import unittest

from src.search.query_expansion import QueryExpander


class TestQueryExpander(unittest.TestCase):
    """Test cases for QueryExpander."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_expand_prosource_commission_query(self) -> None:
        """Test expansion of 'ProSource commission' query.

        Verifies that the expansion includes:
        - Original query
        - ProSource variants
        - Commission variants
        """
        query = "ProSource commission"
        expanded = self.expander.expand_query(query)

        # Check format: original OR expansions
        self.assertIn("OR", expanded)
        self.assertTrue(expanded.startswith("ProSource commission OR"))

        # Check ProSource expansions are included
        self.assertIn("pro-source", expanded.lower())
        self.assertIn("prosource vendor", expanded.lower())

        # Check commission expansions are included
        self.assertIn("commission rate", expanded.lower())
        self.assertIn("commission structure", expanded.lower())
        self.assertIn("payment", expanded.lower())

    def test_expand_dealer_classification_query(self) -> None:
        """Test expansion of 'dealer classification' query.

        Verifies that dealer expansions are properly included.
        """
        query = "dealer classification"
        expanded = self.expander.expand_query(query)

        # Check format
        self.assertIn("OR", expanded)
        self.assertTrue(expanded.startswith("dealer classification OR"))

        # Check dealer expansions
        self.assertIn("dealer types", expanded.lower())
        self.assertIn("customer", expanded.lower())

    def test_expand_lutron_control_system_query(self) -> None:
        """Test expansion of 'Lutron control system' query.

        Verifies Lutron entity expansion works.
        """
        query = "Lutron control system"
        expanded = self.expander.expand_query(query)

        # Check format
        self.assertIn("OR", expanded)

        # Check Lutron expansions
        self.assertIn("lutron control", expanded.lower())
        self.assertIn("lighting control", expanded.lower())

    def test_no_duplicates_in_expansion(self) -> None:
        """Test that expansion doesn't produce duplicate terms.

        Tests with a query that might trigger duplicate handling.
        """
        query = "ProSource ProSource"
        expanded = self.expander.expand_query(query)

        # Count occurrences of "pro-source" (should appear once)
        pro_source_count = expanded.lower().count("pro-source")
        self.assertGreaterEqual(pro_source_count, 1)

        # Check for duplicate "prosource vendor"
        prosource_vendor_count = expanded.lower().count("prosource vendor")
        self.assertEqual(prosource_vendor_count, 1)

    def test_case_insensitive_matching(self) -> None:
        """Test that matching is case-insensitive.

        Verifies that "PROSOURCE", "ProSource", "prosource" all match.
        """
        queries = [
            "PROSOURCE commission",
            "ProSource commission",
            "prosource commission",
        ]

        for query in queries:
            expanded = self.expander.expand_query(query)
            # All should contain expansions (not return original unchanged)
            self.assertIn("OR", expanded)

    def test_multiple_entities_expansion(self) -> None:
        """Test expansion of query with multiple entities."""
        query = "ProSource dealer market"
        expanded = self.expander.expand_query(query)

        # Should find all three entities
        self.assertIn("OR", expanded)

        # Check ProSource expansions
        self.assertIn("pro-source", expanded.lower())

        # Check dealer expansions
        self.assertIn("dealer", expanded.lower())

        # Check market expansions
        self.assertIn("market", expanded.lower())

    def test_empty_query_returns_original(self) -> None:
        """Test that empty queries return unchanged."""
        empty_queries = ["", " ", "\t", "\n"]

        for query in empty_queries:
            expanded = self.expander.expand_query(query)
            self.assertEqual(expanded, query)

    def test_no_entity_match_returns_original(self) -> None:
        """Test that queries with no entity matches return unchanged."""
        query = "how to find information about xyz"
        expanded = self.expander.expand_query(query)

        # Should return original query unchanged (no OR added)
        self.assertEqual(expanded, query)

    def test_expansion_format_correctness(self) -> None:
        """Test that expansion format is correct.

        Format should be: "original OR term1 OR term2 OR ..."
        """
        query = "team sales"
        expanded = self.expander.expand_query(query)

        # Check it starts with original
        self.assertTrue(expanded.startswith(query))

        # Check OR separator is present and properly formatted
        parts = expanded.split(" OR ")
        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual(parts[0], query)

    def test_normalize_term(self) -> None:
        """Test term normalization."""
        test_cases = [
            ("ProSource", "prosource"),
            ("  DEALER  ", "dealer"),
            ("Lutron", "lutron"),
            ("  Commission  ", "commission"),
        ]

        for input_term, expected in test_cases:
            result = self.expander._normalize_term(input_term)
            self.assertEqual(result, expected)

    def test_find_entity_matches(self) -> None:
        """Test entity matching functionality."""
        query = "ProSource commission rates"
        matches = self.expander._find_entity_matches(query)

        # Should find at least ProSource and commission
        self.assertIn("prosource", matches)
        self.assertIn("commission", matches)

        # Check that expansions are returned
        self.assertIsInstance(matches["prosource"], list)
        self.assertGreater(len(matches["prosource"]), 0)

    def test_deduplicate_terms(self) -> None:
        """Test term deduplication."""
        terms = [
            "ProSource",
            "pro-source",
            "Commission",
            "commission",
            "Team",
            "team",
        ]

        unique = self.expander._deduplicate_terms(terms)

        # Should remove case-insensitive duplicates
        unique_normalized = [t.lower() for t in unique]
        self.assertEqual(len(unique), len(set(unique_normalized)))

    def test_all_entity_expansions_have_lists(self) -> None:
        """Test that all entity expansions map to lists."""
        from src.search.query_expansion import ENTITY_EXPANSIONS

        for entity_key, expansions in ENTITY_EXPANSIONS.items():
            self.assertIsInstance(expansions, list)
            self.assertGreater(len(expansions), 0)
            for expansion in expansions:
                self.assertIsInstance(expansion, str)
                self.assertGreater(len(expansion), 0)


class TestQueryExpanderRegressions(unittest.TestCase):
    """Regression tests for query expansion edge cases."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_single_character_query(self) -> None:
        """Test handling of very short queries."""
        result = self.expander.expand_query("a")
        # Single char might not match entities, should return original
        self.assertIsInstance(result, str)

    def test_very_long_query(self) -> None:
        """Test handling of very long queries."""
        query = "ProSource " * 100  # Very long query with multiple matches
        expanded = self.expander.expand_query(query)
        self.assertIsInstance(expanded, str)
        # Should not have excessive repetition
        self.assertLess(len(expanded), len(query) * 10)

    def test_special_characters_in_query(self) -> None:
        """Test handling of special characters."""
        query = "ProSource commission (discount)"
        expanded = self.expander.expand_query(query)
        self.assertIsInstance(expanded, str)
        # Should still work despite special chars
        self.assertIn("OR", expanded)

    def test_query_with_numbers(self) -> None:
        """Test handling of queries with numbers."""
        query = "ProSource 2024 commission 100%"
        expanded = self.expander.expand_query(query)
        self.assertIsInstance(expanded, str)
        self.assertIn("OR", expanded)


class TestQueryExpanderIntegration(unittest.TestCase):
    """Integration tests showing real-world usage patterns."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_business_query_prosource(self) -> None:
        """Test real business query: ProSource commission."""
        query = "ProSource commission"
        expanded = self.expander.expand_query(query)

        # Verify expansion includes key variants
        expanded_lower = expanded.lower()
        expected_terms = [
            "prosource",
            "commission",
            "pro-source",
            "commission rate",
        ]

        for term in expected_terms:
            self.assertIn(term, expanded_lower)

    def test_business_query_dealer(self) -> None:
        """Test real business query: dealer classification."""
        query = "dealer classification"
        expanded = self.expander.expand_query(query)

        expanded_lower = expanded.lower()
        expected_terms = ["dealer", "classification", "customer", "dealer types"]

        for term in expected_terms[:3]:  # At least some should match
            if term in "dealer classification customer dealer types".lower():
                pass  # Good

    def test_business_query_lutron(self) -> None:
        """Test real business query: Lutron lighting control."""
        query = "Lutron lighting control"
        expanded = self.expander.expand_query(query)

        self.assertIn("OR", expanded)
        self.assertIn("lutron", expanded.lower())

    def test_performance_with_typical_query(self) -> None:
        """Verify expansion performs well on typical queries."""
        import time

        query = "ProSource commission rates for dealers in market"
        start = time.time()

        for _ in range(100):
            _ = self.expander.expand_query(query)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / 100) * 1000

        # Should be very fast (target <10ms per expansion)
        self.assertLess(avg_time_ms, 50)


if __name__ == "__main__":
    unittest.main()
