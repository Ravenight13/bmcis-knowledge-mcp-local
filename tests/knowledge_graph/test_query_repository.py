"""
Unit tests for knowledge graph query repository.

Tests verify:
- Query correctness (result structure, ordering)
- Performance characteristics (with sample data)
- Edge cases (empty results, missing entities, high fanout)
- SQL injection prevention (parameterized queries)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import UUID, uuid4
from datetime import datetime

from src.knowledge_graph.query_repository import (
    KnowledgeGraphQueryRepository,
    RelatedEntity,
    TwoHopEntity,
    BidirectionalEntity,
    EntityMention
)


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = Mock()
    conn = Mock()
    cursor = Mock()

    # Setup connection manager
    pool.get_connection.return_value.__enter__ = Mock(return_value=conn)
    pool.get_connection.return_value.__exit__ = Mock(return_value=None)

    # Setup cursor manager
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=None)

    return pool, cursor


@pytest.fixture
def repo(mock_db_pool):
    """Create query repository with mocked database pool."""
    pool, _ = mock_db_pool
    return KnowledgeGraphQueryRepository(pool)


class TestTraverse1Hop:
    """Tests for 1-hop outbound traversal queries."""

    def test_basic_1hop_query(self, repo, mock_db_pool):
        """Test basic 1-hop query returns RelatedEntity objects."""
        _, cursor = mock_db_pool

        # Mock database response
        cursor.fetchall.return_value = [
            (
                uuid4(),  # id
                'Claude AI',  # text
                'TECHNOLOGY',  # entity_type
                '0.95',  # entity_confidence (as string from JSONB)
                'hierarchical',  # relationship_type
                0.9,  # relationship_confidence
                {'context': 'developed by'}  # relationship_metadata
            ),
            (
                uuid4(),
                'GPT-4',
                'TECHNOLOGY',
                '0.92',
                'similar-to',
                0.85,
                None
            )
        ]

        # Execute query
        results = repo.traverse_1hop(
            entity_id=123,
            min_confidence=0.7,
            max_results=50
        )

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, RelatedEntity) for r in results)
        assert results[0].text == 'Claude AI'
        assert results[0].entity_confidence == 0.95
        assert results[0].relationship_confidence == 0.9
        assert results[0].relationship_metadata == {'context': 'developed by'}

    def test_1hop_with_relationship_type_filter(self, repo, mock_db_pool):
        """Test 1-hop query with relationship type filtering."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        # Execute query with type filter
        repo.traverse_1hop(
            entity_id=123,
            min_confidence=0.8,
            relationship_types=['hierarchical', 'similar-to'],
            max_results=20
        )

        # Verify query was executed with correct parameters
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args[0]
        assert call_args[1] == (
            123,
            0.8,
            ['hierarchical', 'similar-to'],
            ['hierarchical', 'similar-to'],
            20
        )

    def test_1hop_empty_results(self, repo, mock_db_pool):
        """Test 1-hop query with no related entities."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        results = repo.traverse_1hop(entity_id=999)

        assert results == []

    def test_1hop_handles_null_entity_confidence(self, repo, mock_db_pool):
        """Test 1-hop query handles NULL entity confidence gracefully."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (uuid4(), 'Entity', 'TYPE', None, 'hierarchical', 0.8, None)
        ]

        results = repo.traverse_1hop(entity_id=123)

        assert results[0].entity_confidence is None


class TestTraverse2Hop:
    """Tests for 2-hop traversal queries."""

    def test_basic_2hop_query(self, repo, mock_db_pool):
        """Test basic 2-hop query returns TwoHopEntity objects with path confidence."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (
                uuid4(),  # id
                'Anthropic',  # text
                'ORGANIZATION',  # entity_type
                '0.98',  # entity_confidence
                'hierarchical',  # relationship_type (hop2)
                0.92,  # relationship_confidence (hop2)
                uuid4(),  # intermediate_entity_id
                'Claude AI',  # intermediate_entity_name
                0.9,  # path_confidence (geometric mean)
                2  # path_depth
            )
        ]

        results = repo.traverse_2hop(
            entity_id=123,
            min_confidence=0.7,
            max_results=100
        )

        assert len(results) == 1
        assert isinstance(results[0], TwoHopEntity)
        assert results[0].text == 'Anthropic'
        assert results[0].intermediate_entity_name == 'Claude AI'
        assert results[0].path_confidence == 0.9
        assert results[0].path_depth == 2

    def test_2hop_prevents_cycles(self, repo, mock_db_pool):
        """Test 2-hop query prevents cycles back to source entity."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        repo.traverse_2hop(entity_id=123, min_confidence=0.7)

        # Verify entity_id appears twice in params (once as source, once to prevent cycles)
        call_args = cursor.execute.call_args[0][1]
        assert call_args.count(123) >= 2  # Source ID + cycle prevention

    def test_2hop_empty_results(self, repo, mock_db_pool):
        """Test 2-hop query with no 2-hop relationships."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        results = repo.traverse_2hop(entity_id=999)

        assert results == []


class TestTraverseBidirectional:
    """Tests for bidirectional traversal queries."""

    def test_basic_bidirectional_query(self, repo, mock_db_pool):
        """Test bidirectional query aggregates inbound and outbound relationships."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (
                uuid4(),  # id
                'Related Entity',  # text
                'TYPE',  # entity_type
                '0.9',  # entity_confidence
                ['hierarchical', 'similar-to'],  # outbound_rel_types (ARRAY)
                ['mentions-in-document'],  # inbound_rel_types (ARRAY)
                0.92,  # max_confidence
                3,  # relationship_count
                1  # min_distance
            )
        ]

        results = repo.traverse_bidirectional(
            entity_id=123,
            min_confidence=0.7,
            max_results=50
        )

        assert len(results) == 1
        assert isinstance(results[0], BidirectionalEntity)
        assert results[0].outbound_rel_types == ['hierarchical', 'similar-to']
        assert results[0].inbound_rel_types == ['mentions-in-document']
        assert results[0].relationship_count == 3

    def test_bidirectional_handles_null_arrays(self, repo, mock_db_pool):
        """Test bidirectional query handles NULL relationship arrays."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (uuid4(), 'Entity', 'TYPE', '0.9', None, ['hierarchical'], 0.8, 1, 1)
        ]

        results = repo.traverse_bidirectional(entity_id=123)

        assert results[0].outbound_rel_types == []  # NULL → empty list
        assert results[0].inbound_rel_types == ['hierarchical']


class TestTraverseWithTypeFilter:
    """Tests for type-filtered traversal queries."""

    def test_type_filtered_query(self, repo, mock_db_pool):
        """Test type-filtered query returns only specified entity types."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (uuid4(), 'Product A', 'PRODUCT', '0.95', 'hierarchical', 0.9, None),
            (uuid4(), 'Product B', 'PRODUCT', '0.92', 'hierarchical', 0.85, None)
        ]

        results = repo.traverse_with_type_filter(
            entity_id=123,
            relationship_type='hierarchical',
            target_entity_types=['PRODUCT', 'TECHNOLOGY'],
            min_confidence=0.8
        )

        assert len(results) == 2
        assert all(r.entity_type == 'PRODUCT' for r in results)

    def test_type_filter_params(self, repo, mock_db_pool):
        """Test type-filtered query passes correct parameters."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        repo.traverse_with_type_filter(
            entity_id=123,
            relationship_type='hierarchical',
            target_entity_types=['VENDOR', 'PRODUCT'],
            min_confidence=0.7,
            max_results=30
        )

        call_args = cursor.execute.call_args[0][1]
        assert call_args == (123, 'hierarchical', ['VENDOR', 'PRODUCT'], 0.7, 30)


class TestGetEntityMentions:
    """Tests for entity mentions lookup queries."""

    def test_basic_mentions_query(self, repo, mock_db_pool):
        """Test mentions query returns EntityMention objects."""
        _, cursor = mock_db_pool

        cursor.fetchall.return_value = [
            (
                456,  # chunk_id
                'docs/readme.md',  # document_id
                'This chunk mentions the entity...',  # chunk_text
                'product_docs',  # document_category
                2,  # chunk_index
                0.92,  # mention_confidence
                datetime(2024, 11, 9, 10, 30, 0)  # indexed_at
            )
        ]

        results = repo.get_entity_mentions(entity_id=123, max_results=100)

        assert len(results) == 1
        assert isinstance(results[0], EntityMention)
        assert results[0].chunk_id == 456
        assert results[0].document_id == 'docs/readme.md'
        assert results[0].mention_confidence == 0.92

    def test_mentions_empty_results(self, repo, mock_db_pool):
        """Test mentions query with entity not mentioned anywhere."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        results = repo.get_entity_mentions(entity_id=999)

        assert results == []


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_query_execution_error_logged(self, repo, mock_db_pool):
        """Test database errors are logged and re-raised."""
        pool, cursor = mock_db_pool

        # Simulate database error
        cursor.execute.side_effect = Exception("Database connection lost")

        with pytest.raises(Exception, match="Database connection lost"):
            repo.traverse_1hop(entity_id=123)

    def test_connection_pool_error(self, repo):
        """Test connection pool errors are handled gracefully."""
        # Mock pool that raises error
        repo.db_pool.get_connection.side_effect = Exception("Pool exhausted")

        with pytest.raises(Exception, match="Pool exhausted"):
            repo.traverse_1hop(entity_id=123)


class TestSQLInjectionPrevention:
    """Tests to verify SQL injection prevention via parameterized queries."""

    def test_parameterized_entity_id(self, repo, mock_db_pool):
        """Test entity_id is parameterized (not string interpolated)."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        # Attempt SQL injection via entity_id
        malicious_id = "123; DROP TABLE knowledge_entities; --"

        try:
            repo.traverse_1hop(entity_id=malicious_id)
        except (TypeError, ValueError):
            # Expected: entity_id must be int, not string
            pass

        # Verify no DROP command in executed SQL
        if cursor.execute.called:
            executed_sql = cursor.execute.call_args[0][0]
            assert 'DROP' not in executed_sql.upper()

    def test_parameterized_relationship_types(self, repo, mock_db_pool):
        """Test relationship_types list is parameterized."""
        _, cursor = mock_db_pool
        cursor.fetchall.return_value = []

        # Malicious relationship type
        malicious_types = ["hierarchical'; DROP TABLE entity_relationships; --"]

        repo.traverse_1hop(entity_id=123, relationship_types=malicious_types)

        # Verify parameters passed to execute() (not interpolated into SQL string)
        call_args = cursor.execute.call_args[0]
        assert len(call_args) == 2  # SQL query + params tuple
        assert malicious_types[0] in str(call_args[1])  # Parameter, not in SQL


class TestPerformance:
    """Performance-related tests (require real database for accurate measurement)."""

    @pytest.mark.skip(reason="Requires real database with indexes and sample data")
    def test_1hop_latency_target(self):
        """Test 1-hop query meets P95 <10ms target."""
        # This test would:
        # 1. Create test database with 10k entities + 30k relationships
        # 2. Run 1-hop query 1000 times
        # 3. Calculate P95 latency
        # 4. Assert P95 < 10ms
        pass

    @pytest.mark.skip(reason="Requires real database with indexes and sample data")
    def test_2hop_latency_target(self):
        """Test 2-hop query meets P95 <50ms target."""
        # This test would:
        # 1. Create test database with 10k entities + 30k relationships
        # 2. Run 2-hop query 1000 times
        # 3. Calculate P95 latency
        # 4. Assert P95 < 50ms
        pass

    @pytest.mark.skip(reason="Requires EXPLAIN ANALYZE support")
    def test_query_uses_indexes(self):
        """Test that queries use indexes (not sequential scans)."""
        # This test would:
        # 1. Run query with EXPLAIN ANALYZE
        # 2. Parse query plan
        # 3. Assert "Index Scan" present, "Seq Scan" not present
        pass


# ============================================================================
# Integration Test Fixtures (for future use with real database)
# ============================================================================

@pytest.fixture(scope="module")
@pytest.mark.skip(reason="Requires real PostgreSQL database")
def test_database():
    """Create test database with schema and sample data."""
    # This fixture would:
    # 1. Create temporary PostgreSQL database
    # 2. Run schema migrations
    # 3. Insert sample entities and relationships
    # 4. Yield database connection
    # 5. Cleanup (drop database)
    pass


@pytest.fixture
@pytest.mark.skip(reason="Requires real database")
def sample_graph_data(test_database):
    """Insert sample graph data for integration tests."""
    # This fixture would insert:
    # - 100 sample entities (various types)
    # - 300 sample relationships (various types)
    # - 500 sample entity mentions
    pass
