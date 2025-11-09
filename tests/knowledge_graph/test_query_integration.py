"""Integration tests for knowledge graph query operations with mock database.

This module tests KnowledgeGraphQueryRepository through the service layer,
validating query behavior against simulated database state.

Test Categories:
1. 1-hop traversal queries with relationship filtering (2 tests)
2. 2-hop traversal with path discovery (2 tests)
3. Bidirectional traversal (2 tests)
4. Query performance characteristics (2 tests)

Total: 8 integration tests validating query execution patterns.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from src.knowledge_graph.graph_service import KnowledgeGraphService
from src.knowledge_graph.cache import Entity
from src.knowledge_graph.cache_config import CacheConfig


# ============================================================================
# Mock Database with Schema Simulation
# ============================================================================


class MockDatabasePool:
    """Mock database pool with schema-aware data storage."""

    def __init__(self) -> None:
        """Initialize with empty schema-based tables."""
        self.entities: dict[UUID, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []

    def add_entity(
        self,
        entity_id: UUID,
        text: str,
        entity_type: str,
        confidence: float,
        mention_count: int
    ) -> None:
        """Add entity to knowledge_entities table."""
        self.entities[entity_id] = {
            "id": entity_id,
            "text": text,
            "entity_type": entity_type,
            "confidence": confidence,
            "mention_count": mention_count,
        }

    def add_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        rel_type: str,
        confidence: float
    ) -> None:
        """Add relationship to entity_relationships table."""
        self.relationships.append({
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relationship_type": rel_type,
            "confidence": confidence,
        })

    def get_connection(self) -> MockConnection:
        """Get connection from pool."""
        return MockConnection(self)


class MockConnection:
    """Mock database connection."""

    def __init__(self, pool: MockDatabasePool) -> None:
        """Initialize with pool reference."""
        self.pool = pool

    def __enter__(self) -> MockConnection:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    def cursor(self) -> MockCursor:
        """Get cursor."""
        return MockCursor(self.pool)


class MockCursor:
    """Mock database cursor with query simulation."""

    def __init__(self, pool: MockDatabasePool) -> None:
        """Initialize with pool reference."""
        self.pool = pool
        self.results: list[tuple[Any, ...]] = []

    def __enter__(self) -> MockCursor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    def execute(self, query: str, params: tuple[Any, ...]) -> None:
        """Execute query with parameter-driven simulation."""
        self.results = []

        # Determine query type from parameters
        if "WITH hop1" in query or "hop2" in query or "WITH" in query:
            # 2-hop or complex query
            self._simulate_2hop(params)
        elif "FULL OUTER JOIN" in query:
            # Bidirectional query
            self._simulate_bidirectional(params)
        else:
            # Default 1-hop query
            self._simulate_1hop(params)

    def _simulate_1hop(self, params: tuple[Any, ...]) -> None:
        """Simulate 1-hop traversal."""
        source_id = params[0]
        min_confidence = params[1] if len(params) > 1 else 0.0

        for rel in self.pool.relationships:
            if (rel["source_entity_id"] == source_id and
                float(rel["confidence"]) >= float(min_confidence)):
                target = self.pool.entities.get(rel["target_entity_id"])
                if target:
                    self.results.append((
                        target["id"],
                        target["text"],
                        target["entity_type"],
                        target["confidence"],
                        rel["relationship_type"],
                        rel["confidence"],
                        None,  # metadata
                    ))

    def _simulate_2hop(self, params: tuple[Any, ...]) -> None:
        """Simulate 2-hop traversal."""
        source_id = params[0]

        # Find 1-hop entities
        hop1_ids: list[UUID] = []
        for rel in self.pool.relationships:
            if rel["source_entity_id"] == source_id:
                hop1_ids.append(rel["target_entity_id"])

        # Find 2-hop entities
        for hop1_id in hop1_ids:
            for rel in self.pool.relationships:
                if (rel["source_entity_id"] == hop1_id and
                    rel["target_entity_id"] != source_id):
                    target = self.pool.entities.get(rel["target_entity_id"])
                    intermediate = self.pool.entities.get(hop1_id)
                    if target and intermediate:
                        self.results.append((
                            target["id"],
                            target["text"],
                            target["entity_type"],
                            target["confidence"],
                            rel["relationship_type"],
                            rel["confidence"],
                            intermediate["id"],
                            intermediate["text"],
                            0.9,  # path_confidence
                            2,    # path_depth
                        ))

    def _simulate_bidirectional(self, params: tuple[Any, ...]) -> None:
        """Simulate bidirectional traversal."""
        source_id = params[0]
        related_ids: set[UUID] = set()

        # Find outbound
        for rel in self.pool.relationships:
            if rel["source_entity_id"] == source_id:
                related_ids.add(rel["target_entity_id"])

        # Find inbound
        for rel in self.pool.relationships:
            if rel["target_entity_id"] == source_id:
                related_ids.add(rel["source_entity_id"])

        # Build results
        for rel_id in related_ids:
            entity = self.pool.entities.get(rel_id)
            if entity:
                self.results.append((
                    entity["id"],
                    entity["text"],
                    entity["entity_type"],
                    entity["confidence"],
                    ["rel-type"],  # outbound
                    [],  # inbound
                    0.9,  # max_confidence
                    1,    # relationship_count
                    1,    # min_distance
                ))

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all results."""
        return self.results


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_pool() -> MockDatabasePool:
    """Create fresh mock database pool."""
    return MockDatabasePool()


@pytest.fixture
def service(mock_pool: MockDatabasePool) -> KnowledgeGraphService:
    """Create service with mock database pool."""
    cache_config = CacheConfig(
        max_entities=100,
        max_relationship_caches=200
    )
    return KnowledgeGraphService(
        db_pool=mock_pool,
        cache_config=cache_config
    )


@pytest.fixture
def populated_pool() -> MockDatabasePool:
    """Create pool with sample data."""
    pool = MockDatabasePool()

    # Create entity hierarchy
    vendor_id = uuid4()
    product1_id = uuid4()
    product2_id = uuid4()
    category_id = uuid4()

    pool.add_entity(vendor_id, "Acme Corp", "VENDOR", 0.95, 20)
    pool.add_entity(product1_id, "Widget", "PRODUCT", 0.90, 15)
    pool.add_entity(product2_id, "Gadget", "PRODUCT", 0.88, 12)
    pool.add_entity(category_id, "Electronics", "CATEGORY", 0.92, 30)

    # Create relationships
    pool.add_relationship(vendor_id, product1_id, "produces", 0.95)
    pool.add_relationship(vendor_id, product2_id, "produces", 0.90)
    pool.add_relationship(product1_id, category_id, "belongs_to", 0.92)
    pool.add_relationship(product2_id, category_id, "belongs_to", 0.88)

    return pool


# ============================================================================
# Category 1: 1-hop Traversal with Relationship Filtering (2 tests)
# ============================================================================


class TestOneHopTraversalQueries:
    """Tests for 1-hop traversal query patterns."""

    def test_service_1hop_traversal_with_real_data(
        self,
        mock_pool: MockDatabasePool,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test 1-hop traversal with real graph data.

        Workflow:
        1. Setup vendor with 2 products
        2. Traverse 1-hop from vendor
        3. Verify both products returned
        4. Verify relationship metadata
        """
        # Create service with populated pool
        cache_config = CacheConfig(
            max_entities=100,
            max_relationship_caches=200
        )
        service = KnowledgeGraphService(
            db_pool=populated_pool,
            cache_config=cache_config
        )

        vendor_id = list(populated_pool.entities.keys())[0]

        # Traverse from vendor
        results = service.traverse_1hop(vendor_id, "produces")

        assert len(results) == 2
        result_texts = {r.text for r in results}
        assert "Widget" in result_texts
        assert "Gadget" in result_texts

        # Verify all are PRODUCT type
        assert all(r.type == "PRODUCT" for r in results)

    def test_service_1hop_with_confidence_filtering(
        self,
        mock_pool: MockDatabasePool,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test 1-hop traversal with confidence threshold filtering.

        Workflow:
        1. Traverse with high confidence threshold (0.91)
        2. Only high-confidence relationships returned
        3. Lower confidence product excluded
        """
        # Create service with populated pool
        cache_config = CacheConfig(
            max_entities=100,
            max_relationship_caches=200
        )
        service = KnowledgeGraphService(
            db_pool=populated_pool,
            cache_config=cache_config
        )

        vendor_id = list(populated_pool.entities.keys())[0]

        # Traverse with high confidence filter
        results = service.traverse_1hop(
            vendor_id,
            "produces",
            min_confidence=0.91
        )

        # Should only find Widget (0.95), not Gadget (0.90)
        assert len(results) == 1
        assert results[0].text == "Widget"


# ============================================================================
# Category 2: 2-hop Traversal (2 tests)
# ============================================================================


class TestTwoHopTraversalQueries:
    """Tests for 2-hop traversal query patterns."""

    def test_service_2hop_traversal_discovers_indirect(
        self,
        service: KnowledgeGraphService,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test 2-hop traversal discovering indirect relationships.

        Workflow:
        1. Vendor -> Product -> Category (2-hops)
        2. Verify categories discovered
        3. Path information correct
        """
        vendor_id = list(populated_pool.entities.keys())[0]

        # Traverse 2 hops
        results = service.traverse_2hop(vendor_id)

        assert len(results) > 0
        # Should find Electronics category
        result_texts = {r.text for r in results}
        assert "Electronics" in result_texts

    def test_service_2hop_prevents_cycles(
        self,
        service: KnowledgeGraphService,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test 2-hop doesn't include source entity.

        Workflow:
        1. Query 2-hop from vendor
        2. Verify vendor not in results
        3. Cycle prevention working
        """
        vendor_id = list(populated_pool.entities.keys())[0]
        vendor_text = populated_pool.entities[vendor_id]["text"]

        results = service.traverse_2hop(vendor_id)

        # Verify vendor not in results
        result_texts = {r.text for r in results}
        assert vendor_text not in result_texts


# ============================================================================
# Category 3: Bidirectional Traversal (2 tests)
# ============================================================================


class TestBidirectionalTraversalQueries:
    """Tests for bidirectional traversal query patterns."""

    def test_service_bidirectional_finds_inbound_and_outbound(
        self,
        service: KnowledgeGraphService,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test bidirectional finds both inbound and outbound relationships.

        Workflow:
        1. Product has outbound (belongs_to) and inbound (produces) rels
        2. Traverse bidirectional from product
        3. Verify both vendor and category found
        """
        # Get product
        product_id = [
            e for e in populated_pool.entities.keys()
            if populated_pool.entities[e]["entity_type"] == "PRODUCT"
        ][0]

        results = service.traverse_bidirectional(product_id)

        # Should find vendor (inbound) and category (outbound)
        result_texts = {r.text for r in results}
        assert "Acme Corp" in result_texts  # vendor (inbound)
        assert "Electronics" in result_texts  # category (outbound)

    def test_service_bidirectional_with_only_outbound(
        self,
        service: KnowledgeGraphService,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test bidirectional with entity having only outbound rels.

        Workflow:
        1. Vendor only has outbound relationships
        2. Traverse bidirectional from vendor
        3. Verify only outbound entities found
        """
        vendor_id = list(populated_pool.entities.keys())[0]

        results = service.traverse_bidirectional(vendor_id)

        # Should only find products (outbound)
        result_types = {r.type for r in results}
        assert "PRODUCT" in result_types
        assert "VENDOR" not in result_types


# ============================================================================
# Category 4: Query Performance Characteristics (2 tests)
# ============================================================================


class TestQueryPerformanceCharacteristics:
    """Tests for query performance patterns."""

    def test_service_caches_traversal_results(
        self,
        service: KnowledgeGraphService,
        populated_pool: MockDatabasePool
    ) -> None:
        """Test traversal results are cached for repeated queries.

        Workflow:
        1. First traversal - DB query, results cached
        2. Second traversal - cache hit
        3. Verify cache statistics
        """
        vendor_id = list(populated_pool.entities.keys())[0]

        # Get stats before
        stats_before = service.get_cache_stats()
        hits_before = stats_before["hits"]

        # First traversal
        result1 = service.traverse_1hop(vendor_id, "produces")
        assert len(result1) == 2

        # Second traversal (should hit cache)
        result2 = service.traverse_1hop(vendor_id, "produces")
        assert len(result2) == 2

        # Verify cache hit
        stats_after = service.get_cache_stats()
        hits_after = stats_after["hits"]
        assert hits_after > hits_before
        assert result1 == result2

    def test_service_large_result_set_handling(
        self,
        service: KnowledgeGraphService,
        mock_pool: MockDatabasePool
    ) -> None:
        """Test service handles large result sets from traversal.

        Workflow:
        1. Create hub entity with 100 outbound relationships
        2. Traverse 1-hop from hub
        3. Verify all 100 returned
        4. Verify cache handles large set
        """
        hub_id = uuid4()
        mock_pool.add_entity(hub_id, "Hub", "ORG", 0.95, 1)

        # Create 100 target entities
        for i in range(100):
            target_id = uuid4()
            mock_pool.add_entity(
                target_id,
                f"Target{i}",
                "PRODUCT",
                0.90,
                1
            )
            mock_pool.add_relationship(hub_id, target_id, "links-to", 0.90)

        # Traverse
        results = service.traverse_1hop(hub_id, "links-to")

        # Verify all 100 returned
        assert len(results) == 100
        assert all(r.type == "PRODUCT" for r in results)

        # Verify cache stats reasonable
        stats = service.get_cache_stats()
        assert stats["size"] <= stats["max_size"]
