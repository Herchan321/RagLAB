from src.eval_metrics import anchor_retrieval_summary, mrr_at_k, ndcg_at_k


def test_mrr_at_k_returns_reciprocal_rank():
    relevant = ["Article 5", "Article 6"]
    retrieved = ["Article 7", "Article 5", "Article 8"]

    assert mrr_at_k(relevant, retrieved, 3) == 1 / 2


def test_ndcg_at_k_counts_rank_position():
    relevant = ["Article 5", "Article 6"]
    retrieved = ["Article 6", "Article 5", "Article 7"]

    assert ndcg_at_k(relevant, retrieved, 3) > 0.0
    assert ndcg_at_k(relevant, retrieved, 3) <= 1.0


def test_anchor_ndcg_is_bounded_and_chunker_independent():
    metrics = anchor_retrieval_summary(
        {"fact-1": ["alpha"], "fact-2": ["beta"]},
        ["fact-1", "fact-2"],
        ["alpha beta", "unrelated"],
        2,
    )

    assert metrics["anchor_recall_at_k"] == 1.0
    assert 0.0 <= metrics["anchor_ndcg_at_k"] <= 1.0
