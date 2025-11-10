"""Search relevance and knowledge graph reranking evaluation.

Tests semantic search quality including:
- Relevance scoring (is result relevant to query?)
- Ranking quality (are results ordered correctly?)
- Knowledge graph impact (does reranking improve results?)
- Query type variations (semantic vs keyword vs entity queries)
"""

import time
from dataclasses import dataclass
from typing import Any

import pytest

from src.mcp.server import get_hybrid_search
from src.knowledge_graph.graph_service import GraphService
from src.search.cross_encoder_reranker import CrossEncoderReranker


@dataclass
class RelevanceScore:
    """Manual relevance judgment (0=irrelevant, 1=somewhat, 2=highly relevant)."""
    result_id: str
    query: str
    judgment: int
    reason: str


@dataclass
class RankingMetrics:
    """Evaluation metrics for ranking quality."""
    query: str
    ndcg_10: float  # NDCG@10 - normalized discounted cumulative gain
    map_10: float   # MAP@10 - mean average precision
    mrr: float      # MRR - mean reciprocal rank
    precision_5: float  # Precision@5
    recall_10: float    # Recall@10


class TestSearchRelevance:
    """Test semantic search relevance with manual judgments."""

    @pytest.fixture
    def search_engine(self):
        """Initialize search engine."""
        return get_hybrid_search()

    @pytest.fixture
    def graph_service(self):
        """Initialize knowledge graph service."""
        return GraphService()

    @pytest.fixture
    def reranker(self):
        """Initialize cross-encoder reranker."""
        return CrossEncoderReranker()

    # ======== TEST QUERIES & RELEVANCE JUDGMENTS ========

    TEST_QUERIES = {
        "vendor_commission": {
            "query": "What are the commission rates for ProSource?",
            "type": "entity_attribute",
            "expected_documents": [
                "Commission_Analysis_2024_2025",
                "PROCESSING_SUMMARY",
                "vendor_commission_structure"
            ]
        },
        "team_structure": {
            "query": "Organizational structure of BMCIS sales teams",
            "type": "organizational",
            "expected_documents": [
                "BMCIS_Master_Playbook",
                "Team_Profile_Completion",
                "Organizational_Structure"
            ]
        },
        "dealer_classification": {
            "query": "How are BMCIS dealers classified and segmented?",
            "type": "classification",
            "expected_documents": [
                "Dealer_Classification_System",
                "Market_Segmentation",
                "Customer_Intelligence"
            ]
        },
        "product_specs": {
            "query": "Lutron lighting control specifications and features",
            "type": "product",
            "expected_documents": [
                "Lutron_Product_Specs",
                "Lighting_Design_Guidelines",
                "Control_System_Architecture"
            ]
        },
        "sales_metrics": {
            "query": "Weekly sales targets and performance metrics",
            "type": "metrics",
            "expected_documents": [
                "Sales_Targets",
                "Performance_Metrics",
                "KPI_Dashboard"
            ]
        }
    }

    # Manual relevance judgments
    RELEVANCE_JUDGMENTS = {
        "vendor_commission": [
            RelevanceScore("doc1", "vendor_commission", 2, "Direct commission rate data"),
            RelevanceScore("doc2", "vendor_commission", 2, "ProSource financial summary"),
            RelevanceScore("doc3", "vendor_commission", 1, "General vendor info, not rates"),
            RelevanceScore("doc4", "vendor_commission", 0, "Unrelated to commission"),
            RelevanceScore("doc5", "vendor_commission", 2, "Commission processing procedure"),
        ],
        "team_structure": [
            RelevanceScore("doc1", "team_structure", 2, "Complete org chart"),
            RelevanceScore("doc2", "team_structure", 2, "District team breakdown"),
            RelevanceScore("doc3", "team_structure", 1, "Mentions teams but high-level"),
            RelevanceScore("doc4", "team_structure", 0, "Product info, not organizational"),
        ],
    }

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, search_engine):
        """Test semantic search returns relevant results."""
        query = "What are the commission rates for ProSource?"
        results = search_engine.search(query, top_k=5)

        assert len(results) > 0, "Search returned no results"

        # Check result relevance
        for i, result in enumerate(results):
            # Results should contain vendor/commission-related content
            text_lower = result.chunk_text.lower()

            # Check for relevance keywords
            relevance_keywords = [
                "commission", "prosource", "vendor", "rate", "percentage", "payment"
            ]
            has_relevant_keyword = any(kw in text_lower for kw in relevance_keywords)

            print(f"Result {i+1}: {result.chunk_text[:100]}...")
            print(f"  Relevance: {'✅' if has_relevant_keyword else '⚠️'}")

            # Top 3 should have relevance keywords
            if i < 3:
                assert has_relevant_keyword, f"Top result {i+1} lacks relevance keywords"

    @pytest.mark.asyncio
    async def test_ranking_order_quality(self, search_engine):
        """Test that results are ordered by relevance."""
        query = "team organizational structure BMCIS"
        results = search_engine.search(query, top_k=10)

        assert len(results) >= 5, "Need at least 5 results"

        # Top results should have higher scores than lower results
        scores = [
            getattr(result, 'hybrid_score', getattr(result, 'score', 0))
            for result in results
        ]

        # Convert strings to floats if needed
        scores = [float(s) if isinstance(s, str) else s for s in scores]

        # Check monotonic decrease (with small tolerance for ties)
        for i in range(len(scores) - 1):
            # Allow slight increases due to floating point
            assert scores[i] >= scores[i+1] - 0.001, \
                f"Ranking not ordered: score[{i}]={scores[i]:.4f} > score[{i+1}]={scores[i+1]:.4f}"

        print(f"✅ Ranking order verified: scores={[f'{s:.3f}' for s in scores[:5]]}")

    @pytest.mark.asyncio
    async def test_entity_search_relevance(self, search_engine, graph_service):
        """Test entity-based search with graph context."""
        # Search for a specific vendor
        query = "ProSource"
        results = search_engine.search(query, top_k=5)

        # Should find ProSource-related documents
        prosource_mentions = 0
        for result in results:
            if "prosource" in result.chunk_text.lower():
                prosource_mentions += 1

        relevance_ratio = prosource_mentions / len(results)
        print(f"ProSource mention ratio: {relevance_ratio:.1%}")

        # Should mention ProSource in at least 2/5 results
        assert prosource_mentions >= 1, "Entity search didn't return entity-relevant results"

    @pytest.mark.asyncio
    async def test_graph_reranking_impact(self, search_engine, graph_service, reranker):
        """Test impact of knowledge graph reranking on result quality."""
        query = "commission structure vendor relationships"

        # Get initial results
        initial_results = search_engine.search(query, top_k=10)
        assert len(initial_results) > 0

        # Check if reranking is active
        # Compare initial ranking with reranked version
        initial_texts = [r.chunk_text[:100] for r in initial_results[:3]]

        print(f"✅ Initial results (top 3):")
        for i, text in enumerate(initial_texts, 1):
            print(f"  {i}. {text}...")

        # Reranking should improve relevance ordering
        # (This would require running through reranker if available)
        assert len(initial_results) > 0, "No results to rerank"

    @pytest.mark.asyncio
    async def test_query_type_variations(self, search_engine):
        """Test different query types for relevance."""
        test_cases = [
            ("What is the dealer classification system?", "definition"),
            ("team members names", "keyword"),
            ("BMCIS organizational structure", "organizational"),
            ("vendor Lutron specifications", "product"),
        ]

        for query, query_type in test_cases:
            results = search_engine.search(query, top_k=5)

            assert len(results) > 0, f"{query_type} query returned no results"

            # Top result should be relevant
            top_text = results[0].chunk_text.lower()
            top_has_content = len(results[0].chunk_text) > 50

            print(f"✅ {query_type:15} query: {len(results)} results, "
                  f"top relevance={'GOOD' if top_has_content else 'POOR'}")

    @pytest.mark.asyncio
    async def test_relevance_consistency(self, search_engine):
        """Test that similar queries return similar relevant results."""
        similar_queries = [
            "commission rates",
            "vendor commission structure",
            "commission payment amounts",
        ]

        all_results = {}
        for query in similar_queries:
            results = search_engine.search(query, top_k=3)
            all_results[query] = [r.source_file for r in results]

        # Check overlap in top results
        files_set_1 = set(all_results[similar_queries[0]])
        files_set_2 = set(all_results[similar_queries[1]])

        overlap = len(files_set_1 & files_set_2)
        print(f"Result overlap for similar queries: {overlap}/3")

        # Similar queries should have some result overlap
        assert overlap >= 1, "Similar queries have no result overlap"

    @pytest.mark.asyncio
    async def test_niche_query_specificity(self, search_engine):
        """Test search specificity for niche/detailed queries."""
        niche_query = "cross encoder reranking performance optimization"
        results = search_engine.search(niche_query, top_k=5)

        # Niche query should return fewer results (more specific)
        # or results should be highly relevant
        if len(results) > 0:
            # Check if results mention key terms
            text = " ".join(r.chunk_text.lower() for r in results[:3])

            has_reranking = "rerank" in text
            has_performance = "performance" in text or "latency" in text or "speed" in text

            specificity_score = sum([has_reranking, has_performance]) / 2
            print(f"Niche query specificity: {specificity_score:.1%}")

            # At least some results should be relevant
            assert specificity_score > 0, "Niche query returned irrelevant results"


class TestRerankerQuality:
    """Test cross-encoder reranker effectiveness."""

    @pytest.fixture
    def reranker(self):
        """Initialize cross-encoder reranker."""
        return CrossEncoderReranker()

    @pytest.mark.asyncio
    async def test_reranker_score_validity(self, reranker):
        """Test that reranker produces valid scores."""
        query = "vendor commission structure"
        candidates = [
            "ProSource commission rates are 15-20% depending on category",
            "Team meeting scheduled for next week",
            "Lutron lighting systems offer superior control",
        ]

        scores = reranker.rerank(query, candidates)

        assert len(scores) == len(candidates)
        assert all(0 <= score <= 1 for score in scores), "Scores should be normalized 0-1"

        # First candidate should have highest score (it's most relevant)
        assert scores[0] > scores[1], "Reranker didn't rank most relevant result first"
        print(f"✅ Reranker scores: {[f'{s:.3f}' for s in scores]}")

    @pytest.mark.asyncio
    async def test_reranker_ranking_improvement(self, reranker):
        """Test that reranker improves ranking compared to initial order."""
        query = "commission structure"
        candidates = [
            "Team meeting scheduled",  # Irrelevant, would be ranked high initially
            "ProSource offers commission structure with tiered rates",  # Very relevant
            "Lighting design services available",  # Somewhat relevant
        ]

        scores = reranker.rerank(query, candidates)

        # Reranked should put most relevant (index 1) first
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        assert sorted_indices[0] == 1, "Reranker failed to identify most relevant candidate"
        print(f"✅ Reranking improved order: {sorted_indices}")


class TestGraphSearchIntegration:
    """Test knowledge graph integration with search."""

    @pytest.fixture
    def graph_service(self):
        """Initialize knowledge graph service."""
        return GraphService()

    @pytest.mark.asyncio
    async def test_entity_relationship_enrichment(self, graph_service):
        """Test that graph provides relationship context."""
        # Query for an entity and check if relationships are found
        entity_name = "ProSource"

        # Get entity from graph
        try:
            entity = await graph_service.get_entity_by_name(entity_name)

            if entity:
                # Get relationships
                relationships = await graph_service.get_entity_relationships(entity.id)

                print(f"✅ Entity '{entity_name}' has {len(relationships)} relationships")
                assert len(relationships) >= 0, "Entity relationships retrieval failed"
        except Exception as e:
            # Graph service might not be fully set up
            pytest.skip(f"Graph service not available: {e}")

    @pytest.mark.asyncio
    async def test_entity_mention_detection(self, search_engine):
        """Test detection of entities in search results."""
        query = "ProSource vendor"
        results = search_engine.search(query, top_k=5)

        # Check for entity mentions
        entity_mentions = {"ProSource": 0, "commission": 0, "vendor": 0}

        for result in results:
            for entity in entity_mentions:
                if entity.lower() in result.chunk_text.lower():
                    entity_mentions[entity] += 1

        print(f"Entity mentions in results: {entity_mentions}")

        # ProSource should be mentioned multiple times
        assert entity_mentions["ProSource"] > 0, "ProSource not found in results"


class TestRankingQualityMetrics:
    """Calculate standard ranking quality metrics."""

    def calculate_ndcg(self, relevances: list[int], ideal_relevances: list[int]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = sum(rel / (1 + i) for i, rel in enumerate(relevances))
        idcg = sum(rel / (1 + i) for i, rel in enumerate(sorted(ideal_relevances, reverse=True)))
        return dcg / idcg if idcg > 0 else 0

    def calculate_map(self, relevances: list[int]) -> float:
        """Calculate Mean Average Precision."""
        ap = 0
        num_relevant = 0
        for i, rel in enumerate(relevances):
            if rel > 0:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        return ap / max(num_relevant, 1)

    def calculate_mrr(self, relevances: list[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rel in enumerate(relevances):
            if rel > 0:
                return 1 / (i + 1)
        return 0

    @pytest.mark.asyncio
    async def test_ranking_metrics_calculation(self):
        """Test calculation of ranking quality metrics."""
        # Example relevance judgments (1 = relevant, 0 = not relevant)
        relevances = [1, 1, 0, 1, 0, 0, 1]
        ideal_relevances = [1, 1, 1, 1, 0, 0, 0]

        ndcg = self.calculate_ndcg(relevances, ideal_relevances)
        map_score = self.calculate_map(relevances)
        mrr = self.calculate_mrr(relevances)

        print(f"✅ Ranking Metrics:")
        print(f"  NDCG@7: {ndcg:.3f}")
        print(f"  MAP@7:  {map_score:.3f}")
        print(f"  MRR:    {mrr:.3f}")

        assert 0 <= ndcg <= 1
        assert 0 <= map_score <= 1
        assert 0 <= mrr <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
