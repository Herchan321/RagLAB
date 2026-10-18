import math
import re
from collections import Counter


def _tokenize(text: str):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _build_bm25_stats(chunks):
    doc_texts = []
    doc_freq = {}
    total_docs = len(chunks)

    for chunk in chunks:
        text = f"{chunk.get('article_id', '')} {chunk.get('title', '')} {chunk.get('text', '')}"
        tokens = _tokenize(text)
        doc_texts.append(tokens)

        unique_terms = set(tokens)
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avgdl = sum(len(tokens) for tokens in doc_texts) / max(1, total_docs)
    idf = {}
    for term, freq in doc_freq.items():
        idf[term] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    return doc_texts, idf, avgdl


def search_bm25(query: str, chunks, top_k: int = 5, k1: float = 1.5, b: float = 0.75):
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    doc_texts, idf, avgdl = _build_bm25_stats(chunks)
    scored = []

    for idx, chunk in enumerate(chunks):
        tokens = doc_texts[idx]
        token_counts = Counter(tokens)
        doc_len = len(tokens)

        score = 0.0
        for term in set(query_terms):
            if term not in token_counts:
                continue
            tf = token_counts[term]
            df = idf.get(term, 0.0)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += df * (tf * (k1 + 1) / denominator)

        scored.append({
            "score": score,
            "article_id": chunk.get("article_id"),
            "title": chunk.get("title"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
