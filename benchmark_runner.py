"""Run isolated retrieval/RAG systems on the versioned evaluation dataset.

This development runner never shares results between systems. Gemini generation
is opt-in with ``--generate``; without it, retrieval metrics remain reproducible
and no external API call is made.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from src.agentic_rag import (
    AgenticRAGWorkflow,
    QuestionRouter,
    RetrievalPlanner,
    RetrievalVerifier,
    retrieve_evidence,
)
from src.benchmark_protocol import SYSTEMS, empty_system_result, load_anchors, load_frozen_questions
from src.eval_metrics import anchor_retrieval_summary, metric_summary
from src.retrieval import INDEX_PATH, MODEL_NAME, load_chunks, search
from src.retrieval.bm25_search import search_bm25
from src.retrieval.hybrid_search import search_hybrid


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "results.dev.json"


def _passages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "article_id": result.get("article_id"),
            "chunk_id": result.get("chunk_id"),
            "title": result.get("title"),
            "text": result.get("text"),
            "score": result.get("score", result.get("combined_score")),
            "sources": result.get("sources", []),
        }
        for result in results
    ]


def _evaluation_passage_id(result: dict[str, Any]) -> str | None:
    article_id = str(result.get("article_id") or "")
    article_number = article_id.removeprefix("Article ").strip()
    if not article_number.isdigit():
        return None
    chunk_id = str(result.get("chunk_id") or "")
    try:
        chunk_number = int(chunk_id.rsplit("_", 1)[-1])
    except ValueError:
        chunk_number = 1
    if article_number == "51":
        return f"AIA-Art51-p{min(chunk_number, 2)}"
    if article_number == "6":
        return f"AIA-Art6-p{min(chunk_number, 2)}"
    if article_number == "99":
        return "AIA-Art99-p3" if chunk_number == 3 else None
    if article_number in {"5", "16"}:
        return f"AIA-Art{article_number}-p1"
    return None


def _retrieval_metrics(
    question: dict[str, Any],
    results: list[dict[str, Any]],
    anchors_by_fact: dict[str, list[str]],
    k: int | None = None,
) -> dict[str, float]:
    effective_anchors = {**anchors_by_fact, **question.get("anchors_by_fact", {})}
    # Use actual result count if not specified, otherwise use k=5 for backward compatibility
    eval_k = k or min(len(results), 5)
    anchor_metrics = anchor_retrieval_summary(
        effective_anchors,
        question["fact_ids"],
        [result.get("text", "") for result in results],
        k=eval_k,
    )
    articles = [result.get("article_id", "") for result in results]
    return {
        **anchor_metrics,
        "article_coverage": metric_summary(
            relevant=question["expected_articles"],
            retrieved=articles,
            k=eval_k,
            reference_articles=question["expected_articles"],
            answer_claims=[],
            supporting_facts=[],
        )["article_coverage"],
    }


def _system_role(system: str) -> str:
    return {
        "no_retrieval": "generation_baseline",
        "naive_rag": "rag_generation",
        "vector": "retrieval_only",
        "bm25": "retrieval_only",
        "hybrid": "rag_generation",
        "agentic": "agentic_rag",
    }[system]


def _generate_answer(client: Any, model: str, question: str, results: list[dict[str, Any]]) -> str:
    if not results:
        prompt = (
            "Answer only from the supplied EU AI Act corpus. "
            "State that there is not enough information to answer.\n\n"
            f"QUESTION:\n{question}\n\nSOURCE PASSAGES:\n(none)"
        )
    else:
        context = "\n\n".join(
            f"[{result.get('article_id')} | {result.get('chunk_id')}]\n{result.get('text', '')}"
            for result in results
        )
        prompt = (
            "Answer only from the supplied EU AI Act passages. Do not invent facts. "
            "If evidence is insufficient, say so.\n\n"
            f"QUESTION:\n{question}\n\nSOURCE PASSAGES:\n{context}"
        )
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def _run_system(
    system: str,
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    index: Any,
    *,
    anchors_by_fact: dict[str, list[str]] | None = None,
    generator: Any = None,
    generation_model: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    result = empty_system_result(system, question["question"])
    result["system_role"] = _system_role(system)
    result["generation_status"] = "not_run"
    result["generation_metrics"] = None
    started = time.perf_counter()
    retrieved: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    try:
        if system == "no_retrieval":
            trace.append({"step": "no_retrieval", "passages": 0})
        else:
            vector = search(question["question"], model, index, chunks, top_k=8)
            if system in {"naive_rag", "vector"}:
                retrieved = vector[:5]
                trace.append(
                    {
                        "step": "vector_search",
                        "count": len(retrieved),
                        "used_for_generation": system == "naive_rag",
                    }
                )
            elif system == "bm25":
                retrieved = search_bm25(question["question"], chunks, top_k=5)
                trace.append({"step": "bm25_search", "count": len(retrieved)})
            elif system == "hybrid":
                bm25 = search_bm25(question["question"], chunks, top_k=8)
                retrieved = search_hybrid(
                    question["question"], chunks, vector, bm25, top_k=5
                )
                trace.append({"step": "vector_search", "count": len(vector)})
                trace.append({"step": "bm25_search", "count": len(bm25)})
                trace.append({"step": "rrf_fusion", "count": len(retrieved)})
            elif system == "agentic":
                if generator is None:
                    router = QuestionRouter()
                    planner = RetrievalPlanner()
                    verifier = RetrievalVerifier()
                    question_type = router.route(question["question"])
                    vector = search(question["question"], model, index, chunks, top_k=8)
                    needs_complementary = (
                        planner.plan(question_type) == "vector_plus_complementary"
                    )
                    if needs_complementary:
                        bm25 = search_bm25(question["question"], chunks, top_k=8)
                        retrieved = search_hybrid(
                            question["question"], chunks, vector, bm25, top_k=8
                        )
                    else:
                        retrieved = vector
                    evidence_ok, reason = verifier.verify(retrieved, question["question"])
                    trace.extend(
                        [
                            {"step": "route", "question_type": question_type},
                            {
                                "step": "plan",
                                "needs_complementary": needs_complementary,
                            },
                            {
                                "step": "complementary_search",
                                "executed": needs_complementary,
                                "count": len(retrieved) if needs_complementary else 0,
                            },
                            {"step": "verify_evidence", "ok": evidence_ok, "reason": reason},
                        ]
                    )
                else:
                    workflow = AgenticRAGWorkflow(client=generator, model=generation_model)
                    state = workflow.run(
                        question["question"],
                        chunks=chunks,
                        embedding_model=model,
                        index=index,
                    )
                    if hasattr(state, "final_results"):
                        retrieved = state.final_results
                        result["answer"] = state.answer
                        trace.extend(
                            [
                                {"step": "route", "question_type": state.question_type},
                                {"step": "verify_evidence", "ok": state.evidence_ok, "reason": state.reason},
                            ]
                        )
                    else:
                        retrieved = state.get("final_results", [])
                        result["answer"] = state.get("answer", "")
                        trace.extend(
                            [
                                {"step": "route", "question_type": state.get("question_type")},
                                {"step": "verify_evidence", "ok": state.get("evidence_ok"), "reason": state.get("reason")},
                            ]
                        )
            else:
                raise ValueError(f"Unsupported system: {system}")

        if generator is not None and system != "agentic":
            result["answer"] = _generate_answer(
                generator, generation_model, question["question"], retrieved
            )
            result["generation_status"] = "completed"
            result["generation_metrics"] = {
                "status": "not_implemented",
                "groundedness": None,
                "hallucination_rate": None,
                "citation_accuracy": None,
            }
        elif generator is None and system in {"no_retrieval", "naive_rag", "hybrid", "agentic"}:
            result["generation_status"] = "not_run"
        result["retrieved_passages"] = _passages(retrieved)
        if system != "no_retrieval":
            result["retrieval_metrics"] = _retrieval_metrics(
                question, retrieved, anchors_by_fact or {}
            )
    except Exception as exc:
        trace.append({"step": "error", "type": type(exc).__name__, "message": str(exc)})
        result["execution_error"] = str(exc)
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["trace"] = trace
    return result


def run_system(
    system: str,
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    index: Any,
    *,
    anchors_by_fact: dict[str, list[str]] | None = None,
    generator: Any = None,
    generation_model: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    """Public wrapper around the isolated system runner."""
    return _run_system(
        system,
        question,
        chunks,
        model,
        index,
        anchors_by_fact=anchors_by_fact,
        generator=generator,
        generation_model=generation_model,
    )


def run_benchmark(
    questions: list[dict[str, Any]],
    systems: tuple[str, ...],
    *,
    generate: bool = False,
    generation_model: str = "gemini-3.5-flash-lite",
    eval_k: int | None = None,
) -> list[dict[str, Any]]:
    chunks = load_chunks()
    anchors_by_fact = load_anchors()
    embedding_model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    client = None
    if generate:
        from google import genai

        client = genai.Client()

    rows = []
    for question in questions:
        for system in systems:
            result = _run_system(
                system,
                question,
                chunks,
                embedding_model,
                index,
                anchors_by_fact=anchors_by_fact,
                generator=client,
                generation_model=generation_model,
            )
            
            # Apply flexible metrics calculation
            if eval_k is not None and system == "agentic" and len(result["retrieved_passages"]) > eval_k:
                # For agentic, calculate metrics on all retrieved passages
                result["retrieval_metrics"] = _retrieval_metrics(
                    question,
                    result["retrieved_passages"],
                    anchors_by_fact or {},
                    k=None  # Use actual count
                )
            else:
                # For other systems or if eval_k is set, use k=5
                result["retrieval_metrics"] = _retrieval_metrics(
                    question,
                    result["retrieved_passages"],
                    anchors_by_fact or {},
                    k=5
                )
                
            rows.append(
                {
                    "question_id": question["id"],
                    "category": question["category"],
                    "split": question["split"],
                    "result": result,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generation-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--eval_k", type=int, help="If set and system is 'agentic', evaluate metrics on all retrieved passages instead of just top-k. Useful for studying agentic retrieval behavior.")
    args = parser.parse_args()

    questions = load_frozen_questions(args.questions) if args.questions else load_frozen_questions()
    dataset_id = "eu-ai-act-qa-v2"
    if args.questions:
        dataset_id = json.loads(args.questions.read_text(encoding="utf-8")).get(
            "dataset_id", dataset_id
        )
    if args.split != "all":
        questions = [question for question in questions if question["split"] == args.split]
    if not questions:
        raise SystemExit(f"No questions found for split {args.split!r}.")

    rows = run_benchmark(
        questions,
        tuple(args.systems),
        generate=args.generate,
        generation_model=args.generation_model,
        eval_k=args.eval_k if hasattr(args, 'eval_k') and args.eval_k is not None else None,
    )
    payload = {
        "dataset_id": dataset_id,
        "split": args.split,
        "systems": args.systems,
        "generation_enabled": args.generate,
        "generation_metrics_status": "available" if args.generate else "not_collected",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} isolated runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
