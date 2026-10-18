from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists() or os.getenv("GEMINI_API_KEY"):
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

from google import genai

from src.retrieval import load_chunks, search
from src.retrieval.bm25_search import search_bm25
from src.retrieval.hybrid_search import search_hybrid


QuestionType = Literal[
    "single_article",
    "multi_article_comparison",
    "legal_threshold",
    "list_categories",
    "missing_evidence",
]


@dataclass
class AgenticState:
    question: str
    question_type: QuestionType | None = None
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    complementary_results: List[Dict[str, Any]] = field(default_factory=list)
    final_results: List[Dict[str, Any]] = field(default_factory=list)
    needs_complementary: bool = False
    evidence_ok: bool = False
    reason: str = ""
    answer: str = ""


class QuestionRouter:
    """Classify the user question before choosing a retrieval strategy."""

    def route(self, question: str) -> QuestionType:
        text = _norm(question)

        if re.search(r"articles?\s*\d+.*(?:and|&|,|/|vs|versus).*articles?\s*\d+", text):
            return "multi_article_comparison"

        if re.search(r"\b(compare|difference|differ|contrast|versus|vs)\b", text):
            return "multi_article_comparison"

        if _is_sanction_question(text):
            return "legal_threshold"

        if re.search(r"\b(all categories|all obligations|all conditions|all exceptions|list all)\b", text):
            return "list_categories"

        return "single_article"


class RetrievalPlanner:
    """Choose the retrieval mode based on question type."""

    def plan(self, question_type: QuestionType) -> str:
        if question_type in {"multi_article_comparison", "legal_threshold", "list_categories"}:
            return "vector_plus_complementary"
        return "vector_only"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).lower()


SANCTION_RE = re.compile(r"\b(fines?|penalt(?:y|ies)|sanctions?|amendes?)\b")


def _is_sanction_question(question: str) -> bool:
    return bool(SANCTION_RE.search(_norm(question)))


def _sanction_articles(question: str) -> set[str]:
    q = _norm(question)
    arts = {"Article 99"}
    if re.search(r"general[- ]purpose|\bgpai\b", q):
        arts.add("Article 101")
    if re.search(r"union institutions|eu institutions|union bodies|agencies", q):
        arts.add("Article 100")
    return arts


def _chunk_no(c) -> int:
    chunk_id = str(c.get("chunk_id") or "")
    try:
        return int(chunk_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _force_sanction_chunks(chunks: List[Dict[str, Any]], question: str):
    if not _is_sanction_question(question):
        return []

    arts = _sanction_articles(question)
    forced = [dict(c, retrieval_source="forced_sanction") for c in chunks if c.get("article_id") in arts]
    return sorted(forced, key=lambda c: (c.get("article_id", ""), _chunk_no(c)))


def _key(result: Dict[str, Any]):
    return (result.get("article_id"), result.get("chunk_id"))


def retrieve_evidence(question: str, chunks: List[Dict[str, Any]], embedding_model, index):
    vector = search(question, embedding_model, index, chunks, top_k=8)
    bm25 = search_bm25(question, chunks, top_k=8)
    hybrid = search_hybrid(
        query=question,
        chunks=chunks,
        vector_results=vector,
        bm25_results=bm25,
        top_k=8,
    )

    merged = []
    seen = set()
    for result in vector + hybrid:
        key = _key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)

    if not _is_sanction_question(question):
        return vector, hybrid, merged[:8]

    forced = _force_sanction_chunks(chunks, question)
    forced_keys = {_key(c) for c in forced}
    support = []
    if re.search(r"prohibit|article 5", _norm(question)):
        support = [
            r for r in merged if r.get("article_id") == "Article 5" and _key(r) not in forced_keys
        ][:2]
    return vector, hybrid, forced + support


class RetrievalVerifier:
    """Check whether the retrieved evidence is sufficient to answer safely."""

    def verify(self, results: List[Dict[str, Any]], question: str) -> tuple[bool, str]:
        if not results:
            return False, "No evidence was retrieved for this question."

        articles = {r.get("article_id") for r in results if r.get("article_id")}
        question_text = _norm(question)
        query_terms = {
            token
            for token in re.findall(r"[a-z]{4,}", question_text)
            if token
            not in {
                "what",
                "which",
                "when",
                "where",
                "does",
                "under",
                "according",
                "between",
                "difference",
                "article",
                "articles",
                "system",
                "systems",
                "question",
                "high",
                "risk",
            }
        }
        evidence_text = _norm(" ".join(r.get("text", "") for r in results))
        overlap = {term for term in query_terms if term in evidence_text}
        referenced_articles = {
            f"Article {number}"
            for number in re.findall(r"articles?\s*(\d+)", question_text)
        }
        article_supported = bool(referenced_articles.intersection(articles))
        if query_terms and not article_supported and len(overlap) * 2 < len(query_terms):
            return False, "The retrieved passages do not contain enough terms supporting the question."

        asks_for_multiple_articles = bool(
            re.search(r"articles?\s*\d+.*(?:and|&|,|/|vs|versus).*articles?\s*\d+", question_text)
            or re.search(r"\b(compare|difference|differ|contrast|versus|vs)\b", question_text)
        )

        if asks_for_multiple_articles and len(articles) < 2:
            return False, "The retrieved evidence does not cover both relevant articles."

        if _is_sanction_question(question_text):
            blob = " ".join(_norm(r.get("text", "")) for r in results)
            if "prohibit" in question_text:
                if "non-compliance with the prohibition" not in blob or "35 000 000" not in blob:
                    return False, "Article 99(3) (fine for Article 5 prohibitions) is not in the evidence."
            elif not re.search(r"eur\s*\d[\d\s]*\d", blob):
                return False, "No fine amount appears in the retrieved evidence."

        return True, "The evidence is sufficient to answer grounded in the supplied passages."


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a careful legal assistant for the EU AI Act.
Answer only from the supplied passages below.
Do not invent legal obligations, paragraph numbers, fines, or exceptions.
If the passages do not contain enough information, say exactly what is missing.

QUESTION:
{question}

SOURCE PASSAGES:
{context}
"""


class AnswerGenerator:
    """Generate a grounded answer with explicit legal references."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def generate(self, results: List[Dict[str, Any]], question: str) -> str:
        if not results:
            return (
                "The supplied passages do not contain enough information to answer this question. "
                "The corpus is missing the required legal basis."
            )

        context = "\n\n".join(
            f"[{result.get('article_id', 'Unknown')} | {result.get('chunk_id', 'unknown')}]\n{result.get('text', '')}"
            for result in results
        )
        prompt = build_prompt(question, context)

        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured; set it before calling the model.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or "No answer generated."


class AgenticRAGWorkflow:
    """A controlled workflow: route -> plan retrieval -> verify evidence -> generate answer."""

    def __init__(self, model=None, client=None):
        self.router = QuestionRouter()
        self.planner = RetrievalPlanner()
        self.verifier = RetrievalVerifier()
        self.model = model or "gemini-3.5-flash-lite"
        if client is not None:
            self.client = client
        elif os.getenv("GEMINI_API_KEY"):
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = None
        self.generator = AnswerGenerator(self.client, self.model)

    def _manual_run(self, question: str, chunks=None, embedding_model=None, index=None):
        state = AgenticState(question=question)
        state.question_type = self.router.route(question)
        state.reason = f"Question classified as: {state.question_type}"

        chunks = chunks or load_chunks()
        if embedding_model is None or index is None:
            raise ValueError("embedding_model and index are required for vector search.")

        _, _, final_results = retrieve_evidence(question, chunks, embedding_model, index)
        state.vector_results = search(question, embedding_model, index, chunks, top_k=8)
        state.complementary_results = search_hybrid(
            query=question,
            chunks=chunks,
            vector_results=state.vector_results,
            bm25_results=search_bm25(question, chunks, top_k=8),
            top_k=8,
        )
        state.final_results = final_results
        state.needs_complementary = self.planner.plan(state.question_type) == "vector_plus_complementary"

        state.evidence_ok, state.reason = self.verifier.verify(state.final_results, question)
        if not state.evidence_ok:
            state.answer = (
                "The supplied passages do not contain enough information to answer this question safely. "
                f"Missing evidence: {state.reason}"
            )
            return state

        state.answer = self.generator.generate(state.final_results, question)
        return state

    def run(self, question: str, chunks=None, embedding_model=None, index=None):
        chunks = chunks or load_chunks()
        if embedding_model is None or index is None:
            raise ValueError("embedding_model and index are required for vector search.")

        try:
            graph = build_agentic_rag_graph(chunks=chunks, embedding_model=embedding_model, index=index, generator=self.generator)
            return graph.invoke({"question": question})
        except RuntimeError:
            return self._manual_run(question, chunks=chunks, embedding_model=embedding_model, index=index)


def _route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    router = QuestionRouter()
    state["question_type"] = router.route(state["question"])
    state["reason"] = f"Question classified as: {state['question_type']}"
    return state


def _vector_search_node(state: Dict[str, Any], *, chunks, embedding_model, index) -> Dict[str, Any]:
    results = search(state["question"], embedding_model, index, chunks, top_k=8)
    state["vector_results"] = results
    state["final_results"] = results
    state["needs_complementary"] = (
        state.get("question_type") in {"multi_article_comparison", "legal_threshold", "list_categories"}
    )
    return state


def _complementary_search_node(state: Dict[str, Any], *, chunks, embedding_model=None, index=None) -> Dict[str, Any]:
    vector_results = state.get("vector_results", [])
    if embedding_model is None or index is None:
        state["final_results"] = vector_results
        state["complementary_results"] = []
        return state

    bm25_results = search_bm25(state["question"], chunks, top_k=8)
    hybrid_results = search_hybrid(
        query=state["question"],
        chunks=chunks,
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=8,
    )

    merged = []
    seen = set()
    for result in vector_results + hybrid_results:
        key = (result.get("article_id"), result.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)

    if _is_sanction_question(state["question"]):
        forced = _force_sanction_chunks(chunks, state["question"])
        support = []
        if re.search(r"prohibit|article 5", _norm(state["question"])):
            forced_keys = {(c.get("article_id"), c.get("chunk_id")) for c in forced}
            support = [
                r for r in merged if r.get("article_id") == "Article 5" and (r.get("article_id"), r.get("chunk_id")) not in forced_keys
            ][:2]
        merged = forced + support

    state["complementary_results"] = hybrid_results
    state["final_results"] = merged[:8]
    return state


def _verify_evidence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    verifier = RetrievalVerifier()
    ok, reason = verifier.verify(state.get("final_results", []), state["question"])
    state["evidence_ok"] = ok
    state["reason"] = reason
    return state


def _generate_answer_node(state: Dict[str, Any], *, generator: AnswerGenerator | None = None) -> Dict[str, Any]:
    generator = generator or AnswerGenerator(None, "gemini-3.5-flash-lite")
    state["answer"] = generator.generate(state.get("final_results", []), state["question"])
    return state


def _refuse_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["answer"] = (
        "The supplied passages do not contain enough information to answer this question safely. "
        f"Missing evidence: {state.get('reason', 'Insufficient evidence.')}"
    )
    return state


def _decide_after_vector(state: Dict[str, Any]) -> str:
    if state.get("needs_complementary"):
        return "complementary_search"
    return "verify_evidence"


def _decide_after_verify(state: Dict[str, Any]) -> str:
    return "generate_answer" if state.get("evidence_ok") else "refuse_answer"


def build_agentic_rag_graph(*, chunks, embedding_model, index, generator: AnswerGenerator | None = None):
    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LangGraph is not installed. Install it with: pip install langgraph") from exc

    workflow = StateGraph(dict)
    workflow.add_node("route_question", _route_question)
    workflow.add_node(
        "vector_search",
        lambda state: _vector_search_node(state, chunks=chunks, embedding_model=embedding_model, index=index),
    )
    workflow.add_node(
        "complementary_search",
        lambda state: _complementary_search_node(state, chunks=chunks, embedding_model=embedding_model, index=index),
    )
    workflow.add_node("verify_evidence", _verify_evidence_node)
    workflow.add_node("generate_answer", lambda state: _generate_answer_node(state, generator=generator))
    workflow.add_node("refuse_answer", _refuse_answer_node)

    workflow.set_entry_point("route_question")
    workflow.add_edge("route_question", "vector_search")
    workflow.add_conditional_edges(
        "vector_search",
        _decide_after_vector,
        {
            "complementary_search": "complementary_search",
            "verify_evidence": "verify_evidence",
        },
    )
    workflow.add_edge("complementary_search", "verify_evidence")
    workflow.add_conditional_edges(
        "verify_evidence",
        _decide_after_verify,
        {
            "generate_answer": "generate_answer",
            "refuse_answer": "refuse_answer",
        },
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("refuse_answer", END)

    return workflow.compile()


def main():
    import argparse

    import faiss
    from sentence_transformers import SentenceTransformer

    from src.retrieval import INDEX_PATH, MODEL_NAME

    parser = argparse.ArgumentParser(description="Run the agentic RAG workflow on the EU AI Act corpus.")
    parser.add_argument("--question", type=str, help="Question to answer, or else the script will prompt for one.")
    args = parser.parse_args()

    chunks = load_chunks()
    index = faiss.read_index(str(INDEX_PATH))
    embedding_model = SentenceTransformer(MODEL_NAME)

    question = (args.question or input("Pose ta question sur l'AI Act : ")).strip()
    if not question:
        print("La question ne peut pas être vide.")
        return

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
    if client is None:
        print("GEMINI_API_KEY is not configured. Set it to enable Gemini generation.")
        return
    workflow = AgenticRAGWorkflow(model="gemini-3.5-flash-lite", client=client)
    state = workflow.run(question, chunks=chunks, embedding_model=embedding_model, index=index)

    print("\n=== AGENTIC RAG ANSWER ===")
    if isinstance(state, dict):
        answer = state.get("answer", "Aucune réponse générée.")
        final_results = state.get("final_results", [])
    else:
        answer = getattr(state, "answer", "Aucune réponse générée.")
        final_results = getattr(state, "final_results", [])

    print(answer)

    if final_results:
        print("\n=== SELECTED EVIDENCE ===")
        for i, result in enumerate(final_results[:6], start=1):
            print(f"{i}. {result.get('article_id')} | {result.get('chunk_id')} | {result.get('text','')[:180]}")


if __name__ == "__main__":
    main()
