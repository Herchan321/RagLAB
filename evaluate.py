import json
import re

import faiss
from sentence_transformers import SentenceTransformer

from src.benchmark_protocol import load_frozen_questions
from src.eval_metrics import metric_summary
from src.retrieval import load_chunks, search
from src.retrieval.bm25_search import search_bm25
from src.retrieval.hybrid_search import search_hybrid


def extract_article_numbers(question: str):
    matches = re.findall(r"articles?\s*(\d+)", question.lower())
    numbers = []
    for group in matches:
        if isinstance(group, tuple):
            for value in group:
                if value:
                    numbers.append(int(value))
        else:
            if group:
                numbers.append(int(group))
    return numbers


def evaluate_question(
    question: str,
    expected_articles: list[str],
    expected_claims: list[str],
    supporting_facts: list[str],
    reranker=None,
):
    chunks = load_chunks()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("data/processed/faiss.index")

    vector_results = search(question, model, index, chunks, top_k=8)
    vector_articles = [item["article_id"] for item in vector_results]

    bm25_results = search_bm25(question, chunks, top_k=8)
    bm25_articles = [item["article_id"] for item in bm25_results]

    hybrid_results = search_hybrid(
        query=question,
        chunks=chunks,
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=5,
        reranker=reranker,
    )
    hybrid_articles = [item["article_id"] for item in hybrid_results]

    metrics = {
        "vector": metric_summary(
            relevant=expected_articles,
            retrieved=vector_articles,
            k=5,
            reference_articles=expected_articles,
            answer_claims=expected_claims,
            supporting_facts=supporting_facts,
        ),
        "bm25": metric_summary(
            relevant=expected_articles,
            retrieved=bm25_articles,
            k=5,
            reference_articles=expected_articles,
            answer_claims=expected_claims,
            supporting_facts=supporting_facts,
        ),
        "hybrid": metric_summary(
            relevant=expected_articles,
            retrieved=hybrid_articles,
            k=5,
            reference_articles=expected_articles,
            answer_claims=expected_claims,
            supporting_facts=supporting_facts,
        ),
    }

    return {
        "vector": {"results": vector_results, "metrics": metrics["vector"]},
        "bm25": {"results": bm25_results, "metrics": metrics["bm25"]},
        "hybrid": {"results": hybrid_results, "metrics": metrics["hybrid"]},
    }


if __name__ == '__main__':
    questions = load_frozen_questions()

    all_results = []

    for item in questions:
        evaluation = evaluate_question(
            item["question"],
            item["expected_articles"],
            item["supporting_facts"],
            item["supporting_facts"],
        )
        all_results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "vector": evaluation["vector"],
            "bm25": evaluation["bm25"],
            "hybrid": evaluation["hybrid"],
        })

    print(json.dumps(all_results, ensure_ascii=False, indent=2))
