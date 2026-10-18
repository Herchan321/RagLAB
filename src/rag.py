
import os
import re
import sys
import unicodedata
from pathlib import Path

import faiss

from google import genai
from sentence_transformers import SentenceTransformer

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

from src.retrieval import (
    INDEX_PATH,
    MODEL_NAME,
    expand_context,
    load_chunks,
    search,
)
from src.retrieval.bm25_search import search_bm25
from src.retrieval.hybrid_search import search_hybrid

LLM_MODEL = "gemini-3.5-flash-lite"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).lower()


SANCTION_RE = re.compile(r"\b(fines?|penalt(?:y|ies)|sanctions?|amendes?)\b")


def _resolve_retrieval_mode(mode: str | None):
    normalized = (mode or "vector").strip().lower()
    allowed = {"vector", "hybrid", "compare", "complementary"}
    if normalized not in allowed:
        raise ValueError(f"Mode de retrieval inconnu: {mode!r}. Choisir parmi {sorted(allowed)}.")
    return normalized


def _needs_complementary_retrieval(question: str) -> bool:
    text = (question or "").lower()

    multi_article_patterns = [
        r"articles?\s*\d+\s*(?:and|&|,|/|vs|versus)\s*articles?\s*\d+",
        r"articles?\s*\d+\s*(?:and|&|,|/|vs|versus)\s*\d+",
        r"\bcompare\b|\bdifference\b|\bdiffer\b|\bcontrast\b|\bversus\b|\bvs\b",
    ]
    penalty_patterns = [
        r"\bfine\b|\bpenalty\b|\bamount\b|\bthreshold\b|\bpercentage\b|\bmaximum\b|\bup to\b",
        r"\bwhich paragraph\b|\bwhat paragraph\b|\bwhat article\b|\bwhich article\b",
    ]
    list_patterns = [
        r"\ball categories\b|\ball conditions\b|\ball exceptions\b|\ball obligations\b",
    ]

    patterns = multi_article_patterns + penalty_patterns + list_patterns
    return any(re.search(pattern, text) for pattern in patterns)


def _effective_retrieval_mode(question: str, requested_mode: str | None) -> str:
    mode = _resolve_retrieval_mode(requested_mode)

    if mode != "vector":
        return mode

    if _needs_complementary_retrieval(question):
        return "complementary"

    return "vector"


def _merge_retrieval_results(vector_results, hybrid_results, top_k=8):
    merged = {}

    for result in vector_results or []:
        key = (result.get("article_id"), result.get("chunk_id"))
        merged[key] = dict(result)
        merged[key]["retrieval_source"] = "vector"

    for result in hybrid_results or []:
        key = (result.get("article_id"), result.get("chunk_id"))
        if key in merged:
            existing = merged[key]
            combined_score = result.get("combined_score", 0.0)
            vector_score = existing.get("score", 0.0)
            existing["retrieval_source"] = "vector+hybrid"
            existing["combined_score"] = max(float(combined_score), float(vector_score))
            continue

        merged[key] = dict(result)
        merged[key]["retrieval_source"] = "hybrid"

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("combined_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score", item.get("combined_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    return ordered[:top_k]


def _chunk_num(chunk):
    try:
        return int(str(chunk["chunk_id"]).rsplit("_", 1)[-1])
    except (KeyError, TypeError, ValueError):
        return 0


def _articles_for_sanction_question(question: str) -> set[str]:
    text = (question or "").lower()
    articles = {"Article 99"}
    if re.search(r"general[- ]purpose|\bgpai\b", text):
        articles.add("Article 101")
    if re.search(r"union institutions|eu institutions|agencies|public authorities", text):
        articles.add("Article 100")
    return articles


def _force_article_chunks(chunks, article_ids):
    article_ids = set(article_ids or [])
    forced = [
        dict(chunk, retrieval_source="forced")
        for chunk in (chunks or [])
        if chunk.get("article_id") in article_ids
    ]
    return sorted(forced, key=lambda c: (c.get("article_id", ""), _chunk_num(c)))


def _is_sanction_question(question: str) -> bool:
    return bool(SANCTION_RE.search(_norm(question)))


def _find_article_99_sanction_chunks(chunks, question, top_k=12):
    if not _is_sanction_question(question):
        return []

    sanction_chunks = []
    for chunk in chunks or []:
        if str(chunk.get("article_id") or "").strip() != "Article 99":
            continue
        sanction_chunks.append(
            {
                "article_id": chunk.get("article_id"),
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "text": chunk.get("text"),
                "score": 1.0,
                "retrieval_type": "sanction_override",
            }
        )

    return sanction_chunks[:top_k]


def _is_article_99_sanction_clause(result) -> bool:
    article_id = _norm(result.get("article_id") or "")
    text = _norm(result.get("text") or "")
    clause_markers = (
        "non-compliance with the prohibition",
        "article 5",
        "35 000 000",
        "7 %",
        "7 percent",
        "up to eur 35",
        "up to 7 %",
    )

    if "article 99" not in article_id and "article 99" not in text:
        return False

    return any(marker in text for marker in clause_markers)


def _prioritize_sanction_evidence(vector_results, hybrid_results, question, top_k=8):
    merged = _merge_retrieval_results(vector_results, hybrid_results, top_k=max(top_k, 16))

    if not _is_sanction_question(question):
        return merged[:top_k]

    preferred_articles = {"Article 99"}
    preferred = []
    article_5 = []
    rest = []
    seen = set()

    for result in merged:
        article_id = result.get("article_id")
        key = (article_id, result.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)

        if article_id in preferred_articles and _is_article_99_sanction_clause(result):
            preferred.append(dict(result))
        elif article_id == "Article 5":
            article_5.append(dict(result))
        else:
            rest.append(dict(result))

    if not preferred:
        preferred = [dict(result) for result in merged if result.get("article_id") in preferred_articles]

    ordered = preferred + article_5 + rest
    return ordered[:top_k]


def _enforce_sanction_context(selected_results, question, top_k=8):
    if not selected_results:
        return []

    if not _is_sanction_question(question):
        return selected_results[:top_k]

    ordered = []
    seen = set()

    for result in selected_results:
        article_id = result.get("article_id")
        key = (article_id, result.get("chunk_id"))
        if key in seen:
            continue

        if article_id == "Article 99" and _is_article_99_sanction_clause(result):
            ordered.append(dict(result))
            seen.add(key)
        elif article_id == "Article 5":
            ordered.append(dict(result))
            seen.add(key)

    for result in selected_results:
        key = (result.get("article_id"), result.get("chunk_id"))
        if key not in seen:
            ordered.append(dict(result))
            seen.add(key)

    return ordered[:top_k]


def main():
    # 1. Vérifier la clé API
    api_key = os.getenv("GEMINI_API_KEY")


    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY est absente. "
            "Configure-la dans le terminal CMD."
        )

    # 2. Initialiser le client Gemini
    client = genai.Client(api_key=api_key)

    # 3. Charger les chunks, l'index et le modèle d'embeddings
    chunks = load_chunks()
    index = faiss.read_index(str(INDEX_PATH))
    embedding_model = SentenceTransformer(MODEL_NAME)

    # 4. Poser une question
    question = input("\nPose ta question sur l'AI Act : ").strip()

    if not question:
        print("La question ne peut pas être vide.")
        return

    requested_mode = os.getenv("RAG_RETRIEVAL_MODE")
    retrieval_mode = _resolve_retrieval_mode(requested_mode)
    effective_mode = _effective_retrieval_mode(question, requested_mode)

    # 5. Rechercher les chunks les plus pertinents
    vector_results = search(
        question,
        embedding_model,
        index,
        chunks,
        top_k=8,
    )
    bm25_results = search_bm25(
        question,
        chunks,
        top_k=8,
    )
    hybrid_results = search_hybrid(
        query=question,
        chunks=chunks,
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=8,
    )

    if effective_mode == "hybrid":
        selected_results = hybrid_results or vector_results
    elif effective_mode in {"complementary", "compare"}:
        selected_results = _prioritize_sanction_evidence(
            vector_results,
            hybrid_results,
            question,
            top_k=8,
        )
    else:
        selected_results = vector_results

    if _is_sanction_question(question):
        forced = _force_article_chunks(chunks, _articles_for_sanction_question(question))
        art5 = [r for r in selected_results if r.get("article_id") == "Article 5"][:2]
        forced_keys = {(r.get("article_id"), r.get("chunk_id")) for r in forced}
        support = [r for r in selected_results if r.get("article_id") == "Article 5" and (r.get("article_id"), r.get("chunk_id")) not in forced_keys][:2]
        selected_results = forced + support + art5
        seen = set()
        deduped = []
        for result in selected_results:
            key = (result.get("article_id"), result.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(result))
        selected_results = deduped
    else:
        selected_results = _enforce_sanction_context(selected_results, question, top_k=8)

    if not selected_results:
        print("Aucun passage pertinent trouvé.")
        return

    # 6. Ajouter les chunks voisins pour récupérer le contexte
    context_results = expand_context(
        selected_results,
        chunks,
        window=1,
    )

    print("\n=== DIAGNOSTIC ARTICLE 99 ===")
    for result in context_results:
        if result.get("article_id") == "Article 99":
            print(f"\n{result.get('chunk_id')}")
            print(result.get("text", ""))

    # 7. Construire le contexte juridique
    context = "\n\n".join(
        f"[{result['article_id']} | {result['chunk_id']}]\n"
        f"{result['text']}"
        for result in context_results
    )

    # 8. Construire le prompt
    prompt = f"""
You are a careful legal information assistant specialized in the
European Union Artificial Intelligence Act (Regulation (EU) 2024/1689).

RULES:

1. SOURCE GROUNDING
- Answer exclusively from the legal passages provided below.
- Do not use outside knowledge to fill gaps in the evidence.
- Never invent legal obligations, exceptions, conditions, penalties,
  article references, or definitions.
- If the passages do not contain enough information, clearly state
  what is missing.


2. COMPLETENESS
- When the user requests all categories listed in a specific legal
  provision, inspect the supplied passages for the complete provision.
- If all enumerated points of that provision are present, list all
  of them and explicitly state that the list is complete for that
  provision.
- Do not claim that the list is incomplete merely because it contains
  lettered points such as (a) through (h).
- If one or more parts of the provision are missing or truncated,
  identify the missing parts and state that completeness cannot
  be verified from the supplied passages.
- Never infer completeness solely from the number of retrieved chunks.


3. LEGAL PRECISION
- Preserve the meaning and scope of the legal text.
- Distinguish prohibitions, obligations, permissions, exceptions,
  and conditions.
- Include relevant exceptions and qualifications when they appear
  in the provided passages.
- Do not present an exception as a general rule.
- Do not treat a recital, explanation, or cross-reference as if it
  were itself an operative legal requirement.

4. CITATIONS
- Cite the exact article and paragraph when this information is
  available in the passages.
- Use only references explicitly supported by the supplied text.
- Associate each claim with its relevant legal reference.

5. RESPONSE FORMAT
- Answer in the language used by the user.
- Start with a direct answer.
- Use numbered lists for categories and bullet points for conditions
  or exceptions.
- Keep explanations clear, precise, and concise.
- Put every condition, result, and exception on a separate bullet point.
- Never concatenate two bullet points on the same line.
- Use consistent Markdown formatting throughout the response.

QUESTION:
{question}

LEGAL PASSAGES:
{context}
"""


    # 9. Générer la réponse avec Gemini
    print(f"\nMode de récupération : {effective_mode.upper()}")
    print("\nGénération de la réponse en cours...")

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    print("\n=== RÉPONSE DU RAG ===")
    print(response.text or "Aucune réponse générée.")

    # 10. Afficher les chunks retrouvés par la recherche vectorielle et hybride
    print("\n=== VECTOR RESULTS ===")
    for i, result in enumerate(vector_results, start=1):
        print(
            f"{i}. {result['article_id']} "
            f"| {result['chunk_id']} "
            f"| score={result['score']:.4f}"
        )

    print("\n=== HYBRID RESULTS ===")
    for i, result in enumerate(hybrid_results, start=1):
        combined_score = result.get("combined_score", 0.0)
        print(
            f"{i}. {result['article_id']} "
            f"| {result['chunk_id']} "
            f"| fused_score={combined_score:.4f}"
        )

    print("\n=== PASSAGES UTILISÉS POUR LE CONTEXTE ===")

    for i, result in enumerate(context_results, start=1):
        score = result.get("score", result.get("combined_score"))

        if score is None:
            score_text = "chunk voisin ajouté"
        elif "combined_score" in result and "score" not in result:
            score_text = f"fused_score={float(score):.4f}"
        else:
            score_text = f"score={float(score):.4f}"

        print(
            f"{i}. {result['article_id']} "
            f"| {result['chunk_id']} "
            f"| {score_text}"
        )


if __name__ == "__main__":
    main()
