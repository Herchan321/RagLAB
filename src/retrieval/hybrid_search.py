"""Hybrid retrieval combining semantic/vector and lexical/BM25 results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _result_key(result: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable key for a chunk based on its article and chunk ID."""
    article_id = str(result.get("article_id") or "").strip()
    chunk_id = str(result.get("chunk_id") or "").strip()
    return article_id, chunk_id


def hybrid_rank_fusion(
    vector_results: Iterable[Dict[str, Any]],
    bm25_results: Iterable[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse vector and BM25 rankings using reciprocal rank fusion.

    The fusion keeps the best-ranked item from each retrieval source and combines
    their positions to produce a stable hybrid score. Results are deduplicated by
    article/chunk ID so the same passage is not repeated across sources.
    """
    fused: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for source_name, source_results in (
        ("vector", vector_results),
        ("bm25", bm25_results),
    ):
        for rank, result in enumerate(source_results, start=1):
            if rank > k:
                break

            key = _result_key(result)
            entry = fused.setdefault(
                key,
                {
                    "article_id": result.get("article_id"),
                    "chunk_id": result.get("chunk_id"),
                    "title": result.get("title"),
                    "text": result.get("text"),
                    "combined_score": 0.0,
                    "max_score": 0.0,
                    "ranks": {},
                    "sources": set(),
                },
            )

            score = float(result.get("score", 0.0) or 0.0)
            if score > entry["max_score"]:
                entry["max_score"] = score

            if source_name not in entry["sources"]:
                entry["sources"].add(source_name)
                entry["ranks"][source_name] = rank
                entry["combined_score"] += 1.0 / (k + rank)
            else:
                continue

    final_results = []
    for item in fused.values():
        item["sources"] = sorted(item["sources"])
        item["ranks"] = dict(sorted(item["ranks"].items()))
        item["combined_score"] = float(item["combined_score"])
        item["max_score"] = float(item.get("max_score", 0.0) or 0.0)
        final_results.append(item)

    final_results.sort(
        key=lambda item: (item["combined_score"], item["max_score"], item.get("chunk_id", "")),
        reverse=True,
    )
    return final_results


def search_hybrid(
    query: str,
    chunks: Iterable[Dict[str, Any]],
    vector_results: Iterable[Dict[str, Any]] | None = None,
    bm25_results: Iterable[Dict[str, Any]] | None = None,
    top_k: int = 5,
    k_rrf: int = 60,
    reranker: Any | None = None,
) -> List[Dict[str, Any]]:
    """Search across precomputed vector and BM25 results and return the best hybrid matches.

    Args:
        query: Query text (kept for API compatibility; the fusion step is ranking-based).
        chunks: Full chunk collection used to enrich the returned matching passages.
        vector_results: Ranked semantic retrieval results.
        bm25_results: Ranked lexical retrieval results.
        top_k: Maximum number of fused results to return.
        k_rrf: Reciprocal rank fusion constant.
        reranker: Optional cross-encoder-like object exposing ``predict``.
            When provided, it reranks the fused candidates by query/passage
            relevance after RRF.
    """
    if vector_results is None:
        vector_results = []
    if bm25_results is None:
        bm25_results = []

    chunk_lookup = {}
    for chunk in chunks:
        key = _result_key(chunk)
        chunk_lookup[key] = chunk

    fused = hybrid_rank_fusion(vector_results, bm25_results, k=k_rrf)

    results = []
    for item in fused:
        key = _result_key(item)
        chunk = chunk_lookup.get(key, {})
        merged = {
            **chunk,
            "article_id": item.get("article_id") or chunk.get("article_id"),
            "chunk_id": item.get("chunk_id") or chunk.get("chunk_id"),
            "title": item.get("title") or chunk.get("title"),
            "text": item.get("text") or chunk.get("text"),
            "combined_score": item.get("combined_score", 0.0),
            "sources": item.get("sources", []),
            "ranks": item.get("ranks", {}),
            "query": query,
        }
        results.append(merged)

    if reranker is not None and results:
        pairs = [(query, result.get("text") or "") for result in results]
        reranker_scores = reranker.predict(pairs)
        if len(reranker_scores) != len(results):
            raise ValueError("reranker.predict must return one score per result.")
        for result, score in zip(results, reranker_scores):
            result["reranker_score"] = float(score)
        results.sort(
            key=lambda item: (item["reranker_score"], item["combined_score"]),
            reverse=True,
        )

    return results[:top_k]
