"""Utilities for basic vector-based retrieval.

This module provides a lightweight in-memory vector search implementation that can
be used for RAG-style retrieval tasks when a full vector database is not needed.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


def _as_vector(vector: Sequence[float]) -> List[float]:
    """Normalize a vector into a list of floats."""
    if np is not None:
        arr = np.asarray(vector, dtype=float)
        return arr.astype(float).tolist()
    return [float(value) for value in vector]


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors."""
    a = _as_vector(vector_a)
    b = _as_vector(vector_b)

    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    if not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = (sum(x * x for x in a)) ** 0.5
    norm_b = (sum(x * x for x in b)) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def vector_search(
    embeddings: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Return the top-k nearest matches using cosine similarity.

    Args:
        embeddings: Sequence of document embedding vectors.
        query_vector: Query embedding vector.
        top_k: Number of results to return.

    Returns:
        A list of (index, similarity_score) pairs sorted in descending score order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    scored = []
    for index, embedding in enumerate(embeddings):
        score = cosine_similarity(embedding, query_vector)
        scored.append((index, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


class VectorSearch:
    """Simple in-memory vector retrieval index."""

    def __init__(self) -> None:
        self._items: List[Any] = []
        self._embeddings: List[List[float]] = []

    def add(self, item: Any, embedding: Sequence[float]) -> None:
        """Add a document and its embedding to the index."""
        self._items.append(item)
        self._embeddings.append(_as_vector(embedding))

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Tuple[Any, float]]:
        """Search the index for the closest documents to the query vector."""
        results = []
        for index, embedding in enumerate(self._embeddings):
            score = cosine_similarity(embedding, query_vector)
            results.append((self._items[index], score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self._items) > 0
