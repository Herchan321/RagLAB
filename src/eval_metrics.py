"""Evaluation metrics for RAG-style systems."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _normalize_items(items: Iterable[str]) -> list[str]:
	return [str(item).strip().lower() for item in items if str(item).strip()]


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
	"""Compute Recall@k over string identifiers."""
	if k <= 0:
		return 0.0
	relevant_set = set(_normalize_items(relevant))
	if not relevant_set:
		return 0.0
	top_k = _normalize_items(retrieved[:k])
	return len(relevant_set.intersection(top_k)) / len(relevant_set)


def precision_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
	"""Compute Precision@k over string identifiers."""
	if k <= 0:
		return 0.0
	top_k = _normalize_items(retrieved[:k])
	if not top_k:
		return 0.0
	relevant_set = set(_normalize_items(relevant))
	return sum(1 for item in top_k if item in relevant_set) / len(top_k)


def article_coverage(reference_articles: Sequence[str], retrieved_articles: Sequence[str]) -> float:
	"""Fraction of reference articles covered by retrieved articles."""
	reference_set = set(_normalize_items(reference_articles))
	if not reference_set:
		return 0.0
	retrieved_set = set(_normalize_items(retrieved_articles))
	return len(reference_set.intersection(retrieved_set)) / len(reference_set)


def groundedness(answer_claims: Sequence[str], supporting_facts: Sequence[str]) -> float:
	"""Fraction of claims supported by provided facts."""
	claims = _normalize_items(answer_claims)
	if not claims:
		return 0.0
	support = set(_normalize_items(supporting_facts))
	supported = sum(1 for claim in claims if claim in support)
	return supported / len(claims)


def hallucination_rate(answer_claims: Sequence[str], supporting_facts: Sequence[str]) -> float:
	"""Fraction of claims not supported by provided facts."""
	return 1.0 - groundedness(answer_claims, supporting_facts)


def unsupported_claim_rate(answer_claims: Sequence[str], supporting_facts: Sequence[str]) -> float:
	"""Alias for hallucination rate."""
	return hallucination_rate(answer_claims, supporting_facts)


def _unique_items_in_order(items: Sequence[str]) -> list[str]:
	"""Return unique items preserving their first appearance order."""
	seen: set[str] = set()
	ordered: list[str] = []
	for item in _normalize_items(items):
		if item not in seen:
			seen.add(item)
			ordered.append(item)
	return ordered


def mrr_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
	"""Mean reciprocal rank@k for the first relevant result in the ranking."""
	if k <= 0:
		return 0.0
	relevant_set = set(_normalize_items(relevant))
	if not relevant_set:
		return 0.0

	top_k = _unique_items_in_order(retrieved[:k])
	for rank, item in enumerate(top_k, start=1):
		if item in relevant_set:
			return 1.0 / rank
	return 0.0


def ndcg_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
	"""Normalized discounted cumulative gain@k using binary relevance."""
	if k <= 0:
		return 0.0
	relevant_set = set(_normalize_items(relevant))
	if not relevant_set:
		return 0.0

	top_k = _unique_items_in_order(retrieved[:k])
	if not top_k:
		return 0.0

	dcg = 0.0
	for rank, item in enumerate(top_k, start=1):
		if item in relevant_set:
			dcg += 1.0 / math.log2(rank + 1)

	ideal_count = min(len(relevant_set), k)
	idcg = 0.0
	for rank in range(1, ideal_count + 1):
		idcg += 1.0 / math.log2(rank + 1)

	if idcg == 0.0:
		return 0.0
	return min(1.0, dcg / idcg)


def metric_summary(
	relevant: Sequence[str],
	retrieved: Sequence[str],
	k: int,
	reference_articles: Sequence[str],
	answer_claims: Sequence[str],
	supporting_facts: Sequence[str],
) -> dict[str, float]:
	"""Convenience summary for common RAG evaluation metrics."""
	return {
		"recall_at_k": recall_at_k(relevant, retrieved, k),
		"precision_at_k": precision_at_k(relevant, retrieved, k),
		"mrr_at_k": mrr_at_k(relevant, retrieved, k),
		"ndcg_at_k": ndcg_at_k(relevant, retrieved, k),
		"article_coverage": article_coverage(reference_articles, retrieved),
		"groundedness": groundedness(answer_claims, supporting_facts),
		"hallucination_rate": hallucination_rate(answer_claims, supporting_facts),
		"unsupported_claim_rate": unsupported_claim_rate(answer_claims, supporting_facts),
	}


def anchor_retrieval_summary(
    anchors_by_fact: dict[str, Sequence[str]],
    fact_ids: Sequence[str],
    retrieved_texts: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Measure retrieval against text anchors, independently of chunk IDs."""
    empty = {
        "anchor_recall_at_k": 0.0,
        "anchor_precision_at_k": 0.0,
        "anchor_mrr_at_k": 0.0,
        "anchor_ndcg_at_k": 0.0,
    }
    if k <= 0:
        return empty
    required = [fact_id for fact_id in fact_ids if anchors_by_fact.get(fact_id)]
    if not required:
        return empty
    found = []
    relevant_flags = []
    for text in list(retrieved_texts)[:k]:
        normalized = str(text).lower()
        matches = {
            fact_id
            for fact_id in required
            if any(anchor.lower() in normalized for anchor in anchors_by_fact[fact_id])
        }
        found.extend(matches)
        relevant_flags.append(bool(matches))
    found_set = set(found)
    relevant_count = sum(relevant_flags)
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(relevant_flags, 1) if is_relevant),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_flags, 1)
        if is_relevant
    )
    ideal_count = min(len(relevant_flags), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "anchor_recall_at_k": len(found_set) / len(required),
        "anchor_precision_at_k": relevant_count / len(relevant_flags) if relevant_flags else 0.0,
        "anchor_mrr_at_k": reciprocal_rank,
        "anchor_ndcg_at_k": dcg / idcg if idcg else 0.0,
    }
