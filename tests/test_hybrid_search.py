import pytest

from src.retrieval.hybrid_search import hybrid_rank_fusion, search_hybrid


def test_hybrid_rank_fusion_unifies_duplicate_chunks():
    vector_results = [
        {"article_id": "Article 5", "chunk_id": 1, "score": 0.90},
        {"article_id": "Article 6", "chunk_id": 2, "score": 0.80},
    ]
    bm25_results = [
        {"article_id": "Article 5", "chunk_id": 1, "score": 0.70},
        {"article_id": "Article 7", "chunk_id": 3, "score": 0.60},
    ]

    fused = hybrid_rank_fusion(vector_results, bm25_results, k=60)

    assert len(fused) == 3
    assert fused[0]["article_id"] == "Article 5"
    assert fused[0]["combined_score"] > fused[1]["combined_score"]
    assert fused[1]["combined_score"] >= fused[2]["combined_score"]


def test_search_hybrid_returns_top_k_unique_results():
    chunks = [
        {"article_id": "Article 5", "chunk_id": 1, "title": "Prohibited practices", "text": "Manipulative AI"},
        {"article_id": "Article 6", "chunk_id": 2, "title": "Transparency", "text": "Explainability requirements"},
        {"article_id": "Article 7", "chunk_id": 3, "title": "Other rules", "text": "General obligations"},
    ]

    results = search_hybrid(
        query="What are prohibited practices in Article 5?",
        chunks=chunks,
        vector_results=[
            {"article_id": "Article 5", "chunk_id": 1, "score": 0.91},
            {"article_id": "Article 6", "chunk_id": 2, "score": 0.81},
        ],
        bm25_results=[
            {"article_id": "Article 5", "chunk_id": 1, "score": 0.73},
            {"article_id": "Article 7", "chunk_id": 3, "score": 0.66},
        ],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["article_id"] == "Article 5"
    assert results[0]["chunk_id"] == 1


class FakeReranker:
    def predict(self, pairs):
        return [1.0 if "Transparency" in passage else 0.2 for _, passage in pairs]


def test_search_hybrid_can_rerank_fused_results():
    chunks = [
        {"article_id": "Article 5", "chunk_id": 1, "title": "Prohibited practices", "text": "Manipulative AI"},
        {"article_id": "Article 6", "chunk_id": 2, "title": "Transparency", "text": "Transparency requirements"},
    ]

    results = search_hybrid(
        query="What are the transparency requirements?",
        chunks=chunks,
        vector_results=[
            {"article_id": "Article 5", "chunk_id": 1, "score": 0.91},
            {"article_id": "Article 6", "chunk_id": 2, "score": 0.81},
        ],
        reranker=FakeReranker(),
        top_k=1,
    )

    assert results[0]["article_id"] == "Article 6"
    assert results[0]["reranker_score"] == 1.0


def test_search_hybrid_rejects_invalid_reranker_output():
    class InvalidReranker:
        def predict(self, pairs):
            return []

    with pytest.raises(ValueError, match="one score per result"):
        search_hybrid(
            query="query",
            chunks=[{"article_id": "Article 5", "chunk_id": 1, "text": "text"}],
            vector_results=[{"article_id": "Article 5", "chunk_id": 1, "score": 1.0}],
            reranker=InvalidReranker(),
        )
