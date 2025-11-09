"""
Knowledge Graph Query Repository
=================================

Provides optimized SQL query methods for graph traversal operations.

Performance targets:
- 1-hop queries: P50 <5ms, P95 <10ms
- 2-hop queries: P50 <20ms, P95 <50ms

Architecture:
- Raw SQL CTEs for maximum performance
- Parameterized queries to prevent injection
- Type-safe result dictionaries
- Connection pooling support via db_pool
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelatedEntity:
    """Result structure for entity traversal queries."""
    id: int
    text: str
    entity_type: str
    entity_confidence: Optional[float]
    relationship_type: str
    relationship_confidence: float
    relationship_metadata: Optional[Dict[str, Any]] = None


@dataclass
class TwoHopEntity:
    """Result structure for 2-hop traversal queries."""
    id: int
    text: str
    entity_type: str
    entity_confidence: Optional[float]
    relationship_type: str
    relationship_confidence: float
    intermediate_entity_id: int
    intermediate_entity_name: str
    path_confidence: float
    path_depth: int = 2


@dataclass
class BidirectionalEntity:
    """Result structure for bidirectional traversal."""
    id: int
    text: str
    entity_type: str
    entity_confidence: Optional[float]
    outbound_rel_types: List[str]
    inbound_rel_types: List[str]
    max_confidence: float
    relationship_count: int
    min_distance: int


@dataclass
class EntityMention:
    """Result structure for entity mention lookups."""
    chunk_id: int
    document_id: str
    chunk_text: str
    document_category: Optional[str]
    chunk_index: int
    mention_confidence: float
    indexed_at: str


class KnowledgeGraphQueryRepository:
    """
    Repository for knowledge graph traversal queries.

    Uses raw SQL CTEs with parameterized patterns for optimal performance.
    All queries return structured result objects for type safety.
    """

    def __init__(self, db_pool):
        """
        Initialize query repository with database connection pool.

        Args:
            db_pool: PostgreSQL connection pool (from core.database.pool)
        """
        self.db_pool = db_pool

    def traverse_1hop(
        self,
        entity_id: int,
        min_confidence: float = 0.7,
        relationship_types: Optional[List[str]] = None,
        max_results: int = 50
    ) -> List[RelatedEntity]:
        """
        Get all entities directly related to source entity (1-hop outbound).

        Performance: P50 <5ms, P95 <10ms (with index on source_entity_id)

        Args:
            entity_id: Source entity ID
            min_confidence: Minimum relationship confidence (default: 0.7)
            relationship_types: Optional filter by relationship types
            max_results: Limit results (default: 50)

        Returns:
            List of RelatedEntity objects, sorted by relationship confidence (descending)

        Example:
            >>> repo.traverse_1hop(entity_id=123, min_confidence=0.8, max_results=10)
            [RelatedEntity(id=456, text='Claude AI', ...)]
        """
        query = """
        WITH related_entities AS (
            SELECT
                r.target_entity_id,
                r.relationship_type,
                r.confidence AS relationship_confidence,
                r.metadata AS relationship_metadata
            FROM entity_relationships r
            WHERE r.source_entity_id = %s
              AND r.confidence >= %s
              AND (%s IS NULL OR r.relationship_type = ANY(%s))
        )
        SELECT
            e.id,
            e.entity_name AS text,
            e.entity_type,
            e.metadata->>'confidence' AS entity_confidence,
            re.relationship_type,
            re.relationship_confidence,
            re.relationship_metadata
        FROM related_entities re
        JOIN knowledge_entities e ON e.id = re.target_entity_id
        ORDER BY re.relationship_confidence DESC
        LIMIT %s
        """

        params = (
            entity_id,
            min_confidence,
            relationship_types,
            relationship_types,
            max_results
        )

        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [
                        RelatedEntity(
                            id=row[0],
                            text=row[1],
                            entity_type=row[2],
                            entity_confidence=float(row[3]) if row[3] else None,
                            relationship_type=row[4],
                            relationship_confidence=row[5],
                            relationship_metadata=row[6]
                        )
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"1-hop traversal failed for entity {entity_id}: {e}")
            raise

    def traverse_2hop(
        self,
        entity_id: int,
        min_confidence: float = 0.7,
        relationship_types: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[TwoHopEntity]:
        """
        Get entities reachable in 2 relationship steps from source entity.

        Performance: P50 <20ms, P95 <50ms (depends on fanout)

        Args:
            entity_id: Source entity ID
            min_confidence: Minimum relationship confidence (default: 0.7)
            relationship_types: Optional filter by relationship types
            max_results: Limit results (default: 100)

        Returns:
            List of TwoHopEntity objects, sorted by path confidence (descending)
            Path confidence = geometric mean of hop1_confidence * hop2_confidence

        Example:
            >>> repo.traverse_2hop(entity_id=123, max_results=20)
            [TwoHopEntity(id=789, text='GPT-4', intermediate_entity_id=456, ...)]
        """
        query = """
        WITH hop1 AS (
            SELECT DISTINCT
                r1.target_entity_id AS entity_id,
                r1.confidence AS hop1_confidence,
                r1.relationship_type AS hop1_rel_type
            FROM entity_relationships r1
            WHERE r1.source_entity_id = %s
              AND r1.confidence >= %s
              AND (%s IS NULL OR r1.relationship_type = ANY(%s))
        ),
        hop2 AS (
            SELECT
                r2.target_entity_id AS entity_id,
                e2.entity_name AS text,
                e2.entity_type,
                e2.metadata->>'confidence' AS entity_confidence,
                r2.relationship_type AS hop2_rel_type,
                r2.confidence AS hop2_confidence,
                h1.entity_id AS intermediate_entity_id,
                ei.entity_name AS intermediate_entity_name,
                h1.hop1_confidence,
                h1.hop1_rel_type,
                SQRT(h1.hop1_confidence * r2.confidence) AS path_confidence
            FROM hop1 h1
            JOIN entity_relationships r2 ON r2.source_entity_id = h1.entity_id
            JOIN knowledge_entities e2 ON e2.id = r2.target_entity_id
            JOIN knowledge_entities ei ON ei.id = h1.entity_id
            WHERE r2.confidence >= %s
              AND r2.target_entity_id != %s
              AND (%s IS NULL OR r2.relationship_type = ANY(%s))
        )
        SELECT
            h2.entity_id,
            h2.text,
            h2.entity_type,
            h2.entity_confidence,
            h2.hop2_rel_type AS relationship_type,
            h2.hop2_confidence AS relationship_confidence,
            h2.intermediate_entity_id,
            h2.intermediate_entity_name,
            h2.path_confidence,
            2 AS path_depth
        FROM hop2 h2
        ORDER BY h2.path_confidence DESC
        LIMIT %s
        """

        params = (
            entity_id,
            min_confidence,
            relationship_types,
            relationship_types,
            min_confidence,
            entity_id,  # Prevent cycles
            relationship_types,
            relationship_types,
            max_results
        )

        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [
                        TwoHopEntity(
                            id=row[0],
                            text=row[1],
                            entity_type=row[2],
                            entity_confidence=float(row[3]) if row[3] else None,
                            relationship_type=row[4],
                            relationship_confidence=row[5],
                            intermediate_entity_id=row[6],
                            intermediate_entity_name=row[7],
                            path_confidence=row[8],
                            path_depth=row[9]
                        )
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"2-hop traversal failed for entity {entity_id}: {e}")
            raise

    def traverse_bidirectional(
        self,
        entity_id: int,
        min_confidence: float = 0.7,
        max_depth: int = 1,
        max_results: int = 50
    ) -> List[BidirectionalEntity]:
        """
        Get all entities connected to source (both incoming and outgoing).

        Performance: P50 <15ms (1-hop), P95 <30ms

        Args:
            entity_id: Source entity ID
            min_confidence: Minimum relationship confidence (default: 0.7)
            max_depth: Maximum traversal depth (1 or 2, default: 1)
            max_results: Limit results (default: 50)

        Returns:
            List of BidirectionalEntity objects, sorted by relationship count
            and max confidence (descending)

        Example:
            >>> repo.traverse_bidirectional(entity_id=123, max_results=30)
            [BidirectionalEntity(id=456, outbound_rel_types=['similar-to'], ...)]
        """
        query = """
        WITH outbound AS (
            SELECT
                r.target_entity_id AS related_entity_id,
                r.relationship_type,
                r.confidence,
                'outbound' AS direction,
                1 AS distance
            FROM entity_relationships r
            WHERE r.source_entity_id = %s
              AND r.confidence >= %s
        ),
        inbound AS (
            SELECT
                r.source_entity_id AS related_entity_id,
                r.relationship_type,
                r.confidence,
                'inbound' AS direction,
                1 AS distance
            FROM entity_relationships r
            WHERE r.target_entity_id = %s
              AND r.confidence >= %s
        ),
        combined AS (
            SELECT
                COALESCE(o.related_entity_id, i.related_entity_id) AS entity_id,
                ARRAY_AGG(DISTINCT o.relationship_type) FILTER (WHERE o.relationship_type IS NOT NULL) AS outbound_rel_types,
                ARRAY_AGG(DISTINCT i.relationship_type) FILTER (WHERE i.relationship_type IS NOT NULL) AS inbound_rel_types,
                GREATEST(COALESCE(MAX(o.confidence), 0), COALESCE(MAX(i.confidence), 0)) AS max_confidence,
                COUNT(*) AS relationship_count,
                MIN(COALESCE(o.distance, i.distance)) AS min_distance
            FROM outbound o
            FULL OUTER JOIN inbound i ON o.related_entity_id = i.related_entity_id
            GROUP BY COALESCE(o.related_entity_id, i.related_entity_id)
        )
        SELECT
            c.entity_id,
            e.entity_name AS text,
            e.entity_type,
            e.metadata->>'confidence' AS entity_confidence,
            c.outbound_rel_types,
            c.inbound_rel_types,
            c.max_confidence,
            c.relationship_count,
            c.min_distance
        FROM combined c
        JOIN knowledge_entities e ON e.id = c.entity_id
        ORDER BY c.relationship_count DESC, c.max_confidence DESC
        LIMIT %s
        """

        params = (
            entity_id,
            min_confidence,
            entity_id,
            min_confidence,
            max_results
        )

        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [
                        BidirectionalEntity(
                            id=row[0],
                            text=row[1],
                            entity_type=row[2],
                            entity_confidence=float(row[3]) if row[3] else None,
                            outbound_rel_types=row[4] if row[4] else [],
                            inbound_rel_types=row[5] if row[5] else [],
                            max_confidence=row[6],
                            relationship_count=row[7],
                            min_distance=row[8]
                        )
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Bidirectional traversal failed for entity {entity_id}: {e}")
            raise

    def traverse_with_type_filter(
        self,
        entity_id: int,
        relationship_type: str,
        target_entity_types: List[str],
        min_confidence: float = 0.7,
        max_results: int = 50
    ) -> List[RelatedEntity]:
        """
        Get related entities of specific type(s).

        Performance: P50 <8ms, P95 <15ms (with entity_type filter)

        Args:
            entity_id: Source entity ID
            relationship_type: Specific relationship type to filter
            target_entity_types: Entity types to include (e.g., ['VENDOR', 'PRODUCT'])
            min_confidence: Minimum relationship confidence (default: 0.7)
            max_results: Limit results (default: 50)

        Returns:
            List of RelatedEntity objects of specified types, sorted by confidence

        Example:
            >>> repo.traverse_with_type_filter(
            ...     entity_id=123,
            ...     relationship_type='hierarchical',
            ...     target_entity_types=['PRODUCT', 'TECHNOLOGY']
            ... )
            [RelatedEntity(id=456, entity_type='PRODUCT', ...)]
        """
        query = """
        SELECT
            e.id,
            e.entity_name AS text,
            e.entity_type,
            e.metadata->>'confidence' AS entity_confidence,
            r.relationship_type,
            r.confidence AS relationship_confidence,
            r.metadata AS relationship_metadata
        FROM entity_relationships r
        JOIN knowledge_entities e ON e.id = r.target_entity_id
        WHERE r.source_entity_id = %s
          AND r.relationship_type = %s
          AND e.entity_type = ANY(%s)
          AND r.confidence >= %s
        ORDER BY r.confidence DESC
        LIMIT %s
        """

        params = (
            entity_id,
            relationship_type,
            target_entity_types,
            min_confidence,
            max_results
        )

        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [
                        RelatedEntity(
                            id=row[0],
                            text=row[1],
                            entity_type=row[2],
                            entity_confidence=float(row[3]) if row[3] else None,
                            relationship_type=row[4],
                            relationship_confidence=row[5],
                            relationship_metadata=row[6]
                        )
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Type-filtered traversal failed for entity {entity_id}: {e}")
            raise

    def get_entity_mentions(
        self,
        entity_id: int,
        max_results: int = 100
    ) -> List[EntityMention]:
        """
        Get documents and chunks where entity is mentioned.

        Performance: P50 <10ms, P95 <20ms (with index on entity_id)

        Args:
            entity_id: Entity ID to find mentions for
            max_results: Limit results (default: 100)

        Returns:
            List of EntityMention objects, sorted by confidence and indexed_at

        Example:
            >>> repo.get_entity_mentions(entity_id=123, max_results=50)
            [EntityMention(chunk_id=1, document_id='doc.md', ...)]
        """
        query = """
        SELECT
            ce.chunk_id AS chunk_id,
            kb.source_file AS document_id,
            kb.chunk_text AS chunk_text,
            kb.source_category AS document_category,
            kb.chunk_index,
            ce.confidence AS mention_confidence,
            kb.created_at AS indexed_at
        FROM chunk_entities ce
        JOIN knowledge_base kb ON kb.id = ce.chunk_id
        WHERE ce.entity_id = %s
        ORDER BY ce.confidence DESC, kb.created_at DESC
        LIMIT %s
        """

        params = (entity_id, max_results)

        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [
                        EntityMention(
                            chunk_id=row[0],
                            document_id=row[1],
                            chunk_text=row[2],
                            document_category=row[3],
                            chunk_index=row[4],
                            mention_confidence=row[5],
                            indexed_at=str(row[6])
                        )
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Entity mentions lookup failed for entity {entity_id}: {e}")
            raise

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _build_cte_query(self, base_query: str, cte_clauses: List[str]) -> str:
        """
        Build CTE query from base query and CTE clauses.

        Internal helper for constructing complex CTE-based queries.
        """
        cte_str = ", ".join(cte_clauses)
        return f"WITH {cte_str}\n{base_query}"

    def _execute_query(
        self,
        query: str,
        params: tuple,
        result_class: type
    ) -> List[Any]:
        """
        Execute parameterized query and return typed results.

        Internal helper for executing queries with connection pooling.
        """
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [result_class(*row) for row in rows]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
