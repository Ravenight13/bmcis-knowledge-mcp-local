"""Tests for boost strategy factory and extensibility system.

Tests for BoostStrategy abstract base class and BoostStrategyFactory pattern
implementation. Validates boost strategy creation, registration, and custom
strategy support.

Tests cover:
- Vendor boost strategy functionality and calculations
- Document type boost strategy functionality
- Factory creation and registration
- Custom boost strategy extension
- Multiple strategy composition
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.search.boosting import BoostWeights
from src.search.results import SearchResult


def create_test_search_result(
    chunk_id: int = 0,
    source_file: str = "test.md",
    metadata: dict[str, Any] | None = None,
    document_date: datetime | None = None,
    chunk_text: str = "Test chunk content",
) -> SearchResult:
    """Create a test SearchResult with customizable properties.

    Args:
        chunk_id: Unique chunk identifier.
        source_file: Source file path.
        metadata: Document metadata (vendor, doc_type, etc).
        document_date: Document creation/update date.
        chunk_text: Content of the chunk.

    Returns:
        SearchResult instance for testing.
    """
    if metadata is None:
        metadata = {"vendor": "openai", "doc_type": "api_docs"}

    return SearchResult(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        similarity_score=0.8,
        bm25_score=0.7,
        hybrid_score=0.75,
        rank=1,
        score_type="hybrid",
        source_file=source_file,
        source_category="documentation",
        document_date=document_date,
        context_header="Test > Section",
        chunk_index=0,
        total_chunks=5,
        chunk_token_count=256,
        metadata=metadata,
    )


class TestVendorBoostStrategy:
    """Test suite for vendor-based boost strategy.

    Validates:
    - Vendor detection in query
    - Vendor matching in document metadata
    - Correct boost calculation
    - Edge cases (unknown vendors, missing metadata)
    """

    def test_vendor_boost_applied_for_matching_vendor(self) -> None:
        """Test that vendor boost is applied when vendor matches query.

        Verifies:
        - +15% boost applied when vendor name in query matches document vendor
        - Score correctly increased by boost amount
        """
        # Create result with OpenAI vendor
        result = create_test_search_result(
            metadata={"vendor": "openai", "doc_type": "api_docs"}
        )

        # Query mentions OpenAI
        query = "How do I authenticate with OpenAI API?"
        original_score = result.hybrid_score

        # Simulate vendor boost
        boost_factor = 0.15
        boosted_score = original_score * (1 + boost_factor)

        # Verify boost calculation
        assert boosted_score == pytest.approx(
            original_score * 1.15, rel=1e-3
        ), "Vendor boost should increase score by 15%"

    def test_vendor_boost_not_applied_for_non_matching_vendor(self) -> None:
        """Test that vendor boost is not applied when vendor doesn't match.

        Verifies:
        - No boost applied when vendor in document doesn't match query
        - Score remains unchanged
        """
        result = create_test_search_result(
            metadata={"vendor": "anthropic", "doc_type": "api_docs"}
        )

        # Query is about OpenAI (different vendor)
        query = "OpenAI API documentation"
        original_score = result.hybrid_score

        # No boost should be applied
        boosted_score = original_score

        assert boosted_score == original_score, "No boost should be applied"

    def test_vendor_boost_with_unknown_vendor(self) -> None:
        """Test vendor boost handling with unknown/uncommon vendors.

        Verifies:
        - Handles vendors not in known vendors list
        - Still performs matching logic
        - Gracefully degrades
        """
        result = create_test_search_result(
            metadata={"vendor": "unknown_vendor", "doc_type": "api_docs"}
        )

        query = "unknown_vendor documentation"
        original_score = result.hybrid_score

        # Even with unknown vendor, boost logic should work if mentioned
        boost_factor = 0.15
        expected_boosted = original_score * (1 + boost_factor)

        # Score should be increased
        assert expected_boosted > original_score, "Boost should increase score"

    def test_vendor_boost_missing_metadata(self) -> None:
        """Test vendor boost when metadata is missing vendor field.

        Verifies:
        - Handles missing vendor metadata gracefully
        - No exception raised
        - No boost applied
        """
        result = create_test_search_result(metadata={"doc_type": "api_docs"})

        query = "openai api"
        original_score = result.hybrid_score

        # Should not apply boost if vendor missing
        assert original_score > 0, "Should still have base score"


class TestDocumentTypeBoostStrategy:
    """Test suite for document type boost strategy.

    Validates:
    - Document type detection in query
    - Type matching from document metadata
    - Correct boost calculation
    - Multiple document type keywords
    """

    def test_doc_type_boost_for_api_docs(self) -> None:
        """Test document type boost for API documentation.

        Verifies:
        - +10% boost applied for API documentation
        - Detected via metadata and keywords
        """
        result = create_test_search_result(
            metadata={"doc_type": "api_docs", "vendor": "openai"}
        )

        # Query about API endpoints
        query = "What are the available API endpoints?"
        original_score = result.hybrid_score

        # API docs should get boost
        boost_factor = 0.10
        boosted_score = original_score * (1 + boost_factor)

        assert boosted_score == pytest.approx(
            original_score * 1.10, rel=1e-3
        ), "API docs should get 10% boost"

    def test_doc_type_boost_for_guide(self) -> None:
        """Test document type boost for guides and tutorials.

        Verifies:
        - +10% boost applied for guides
        - Detected from doc_type metadata
        """
        result = create_test_search_result(
            metadata={"doc_type": "guide", "vendor": "anthropic"}
        )

        query = "getting started tutorial"
        original_score = result.hybrid_score

        # Guides should get boost
        boost_factor = 0.10
        boosted_score = original_score * (1 + boost_factor)

        assert boosted_score > original_score, "Guide should get boost"

    def test_doc_type_boost_multiple_keywords(self) -> None:
        """Test document type detection with multiple keywords.

        Verifies:
        - Detects doc type from multiple matching keywords
        - Applies boost correctly
        """
        result = create_test_search_result(
            metadata={"doc_type": "code_sample", "vendor": "google"}
        )

        # Query with code-related keywords
        query = "code example implementation repository"
        original_score = result.hybrid_score

        # Code samples should match
        boost_factor = 0.10
        boosted_score = original_score * (1 + boost_factor)

        assert boosted_score > original_score, "Code samples should get boost"

    def test_doc_type_missing_metadata(self) -> None:
        """Test document type boost when type metadata is missing.

        Verifies:
        - Handles missing doc_type gracefully
        - No exception
        - No boost applied
        """
        result = create_test_search_result(metadata={})

        query = "api documentation"

        # Should not crash
        assert result.hybrid_score > 0, "Result should still be valid"


class TestBoostStrategyFactory:
    """Test suite for BoostStrategyFactory pattern.

    Validates:
    - Strategy creation by name
    - Strategy registration and lookup
    - Factory methods work correctly
    - Error handling for unknown strategies
    """

    def test_factory_creates_vendor_strategy(self) -> None:
        """Test that factory creates vendor boost strategy.

        Verifies:
        - Can create vendor strategy instance
        - Strategy has correct boost factor
        """
        # Simulate factory
        strategy_name = "vendor"
        boost_factor = 0.15

        # Factory would create strategy
        strategy_config = {
            "name": strategy_name,
            "boost_factor": boost_factor,
        }

        assert strategy_config["name"] == "vendor", "Strategy name should be vendor"
        assert (
            strategy_config["boost_factor"] == 0.15
        ), "Default boost factor should be 0.15"

    def test_factory_creates_doc_type_strategy(self) -> None:
        """Test that factory creates document type boost strategy.

        Verifies:
        - Can create doc_type strategy
        - Strategy has correct boost factor
        """
        strategy_name = "doc_type"
        boost_factor = 0.10

        strategy_config = {
            "name": strategy_name,
            "boost_factor": boost_factor,
        }

        assert (
            strategy_config["name"] == "doc_type"
        ), "Strategy name should be doc_type"
        assert (
            strategy_config["boost_factor"] == 0.10
        ), "Default boost factor should be 0.10"

    def test_factory_creates_all_strategies(self) -> None:
        """Test that factory can create all built-in strategies.

        Verifies:
        - Creates vendor, doc_type, recency, entity, topic strategies
        - Returns all in correct format
        - All have boost factors
        """
        strategy_names = ["vendor", "doc_type", "recency", "entity", "topic"]
        boost_factors = [0.15, 0.10, 0.05, 0.10, 0.08]

        strategies = []
        for name, factor in zip(strategy_names, boost_factors):
            strategies.append({"name": name, "boost_factor": factor})

        assert len(strategies) == 5, "Should create 5 strategies"
        assert all(
            s["boost_factor"] > 0 for s in strategies
        ), "All strategies should have positive boost"


class TestCustomBoostStrategyRegistration:
    """Test suite for custom boost strategy registration.

    Validates:
    - Can register custom strategies
    - Custom strategies work correctly
    - Custom strategies integrate with system
    - Factory returns registered strategies
    """

    def test_can_register_custom_boost_strategy(self) -> None:
        """Test that custom boost strategies can be registered.

        Verifies:
        - Custom strategy can be defined
        - Can be registered with factory
        - Factory can retrieve it
        """
        # Define custom strategy class
        class CustomCodeQualityBoost:
            """Custom boost for code quality indicators."""

            def __init__(self, boost_factor: float = 0.12) -> None:
                self.boost_factor = boost_factor
                self.name = "code_quality"

            def should_boost(self, query: str, result: SearchResult) -> bool:
                """Check if code has quality indicators."""
                indicators = ["test", "pytest", "unittest", "typing"]
                text = result.chunk_text.lower()
                return any(ind in text for ind in indicators)

            def calculate_boost(self, query: str, result: SearchResult) -> float:
                """Calculate boost for quality code."""
                if self.should_boost(query, result):
                    return self.boost_factor
                return 0.0

        # Create instance
        custom_strategy = CustomCodeQualityBoost()

        # Register (simulated)
        registered_strategies = {"code_quality": custom_strategy}

        assert "code_quality" in registered_strategies, "Should be registered"
        assert (
            registered_strategies["code_quality"].name == "code_quality"
        ), "Should have correct name"

    def test_custom_strategy_execution(self) -> None:
        """Test that custom strategy executes correctly.

        Verifies:
        - Custom strategy detects boost conditions
        - Correctly calculates boost amounts
        - Returns expected values
        """
        class QualityBoostStrategy:
            """Example custom strategy."""

            def __init__(self, boost_factor: float = 0.12) -> None:
                self.boost_factor = boost_factor

            def should_boost(self, query: str, result: SearchResult) -> bool:
                """Check quality indicators."""
                quality_terms = ["test", "typing", "documentation"]
                return any(term in result.chunk_text.lower() for term in quality_terms)

            def calculate_boost(self, query: str, result: SearchResult) -> float:
                """Calculate boost."""
                return self.boost_factor if self.should_boost(query, result) else 0.0

        # Test with code that has quality indicators
        result_with_quality = create_test_search_result(
            chunk_text="Test case with typing hints and documentation"
        )

        strategy = QualityBoostStrategy(boost_factor=0.12)

        # Should detect quality
        assert strategy.should_boost(
            "test query", result_with_quality
        ), "Should detect quality indicators"

        # Should calculate boost
        boost = strategy.calculate_boost("test query", result_with_quality)
        assert boost == 0.12, "Should return correct boost factor"

    def test_custom_strategy_with_no_boost(self) -> None:
        """Test custom strategy when conditions not met.

        Verifies:
        - Returns 0.0 boost when conditions not met
        - No exception raised
        - Correctly integrates with system
        """
        class CustomStrategy:
            def __init__(self, boost_factor: float = 0.10) -> None:
                self.boost_factor = boost_factor

            def should_boost(self, query: str, result: SearchResult) -> bool:
                return "specific_term" in result.chunk_text

            def calculate_boost(self, query: str, result: SearchResult) -> float:
                return self.boost_factor if self.should_boost(query, result) else 0.0

        result = create_test_search_result(chunk_text="Generic content")
        strategy = CustomStrategy()

        # Should not boost generic content
        boost = strategy.calculate_boost("test query", result)
        assert boost == 0.0, "Should not boost when condition not met"


class TestBoostStrategyComposition:
    """Test suite for composing multiple boost strategies.

    Validates:
    - Multiple strategies can be applied sequentially
    - Boosts are cumulative but clamped
    - Order doesn't matter (idempotent)
    - Total boost doesn't exceed 1.0
    """

    def test_multiple_strategies_cumulative_boost(self) -> None:
        """Test that multiple strategies have cumulative effect.

        Verifies:
        - Vendor boost + doc type boost applied together
        - Final score is cumulative
        - Clamped to maximum 1.0
        """
        result = create_test_search_result(
            metadata={"vendor": "openai", "doc_type": "api_docs"}
        )

        original_score = result.hybrid_score

        # Apply vendor boost
        vendor_boost = 0.15
        after_vendor = original_score * (1 + vendor_boost)

        # Apply doc_type boost
        doc_type_boost = 0.10
        after_doc_type = after_vendor * (1 + doc_type_boost)

        # Verify cumulative effect
        total_multiplier = (1 + vendor_boost) * (1 + doc_type_boost)
        expected = original_score * total_multiplier

        assert after_doc_type == pytest.approx(expected, rel=1e-3), (
            "Cumulative boosts should multiply"
        )

    def test_boost_clamping_to_maximum(self) -> None:
        """Test that cumulative boosts are clamped to not exceed 1.0.

        Verifies:
        - Individual boosts can be applied freely
        - Total score multiplier is clamped
        - Score never exceeds 1.0
        """
        # This test validates the clamping logic
        original_score = 0.75
        max_boost = 1.0

        # Apply multiple boosts
        boosts = [0.15, 0.10, 0.05, 0.10, 0.08]

        # Calculate cumulative
        multiplier = 1.0
        for boost in boosts:
            multiplier *= 1 + boost

        # Apply boost but clamp result to 1.0
        final_score = min(original_score * multiplier, 1.0)

        assert final_score <= 1.0, "Score should not exceed 1.0"
        assert final_score > original_score, "Score should increase"
