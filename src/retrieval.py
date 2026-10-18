
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/faiss.index")
METADATA_PATH = Path("data/processed/chunks_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_index(chunks, model):
    texts = [
        f"{chunk.get('article_id', '')} {chunk['title']}. {chunk['text']}"
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    # Similarité cosinus grâce aux vecteurs normalisés.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"\nIndex créé : {len(chunks)} chunks")
    print(f"Dimensions des embeddings : {embeddings.shape[1]}")
    print(f"Index sauvegardé : {INDEX_PATH}")


def _normalize_article_id(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _extract_article_references(query):
    lower_query = query.lower()
    refs = []
    seen = set()

    for match in re.finditer(r"articles?\s*(\d+)", lower_query):
        ref = f"article {match.group(1)}"
        if ref not in seen:
            refs.append(ref)
            seen.add(ref)

    # Handles questions like "Articles 5 and 6" without relying on title strings.
    for match in re.finditer(r"(?:^|\s|,|\()and\s+(\d+)", lower_query):
        ref = f"article {match.group(1)}"
        if ref not in seen:
            refs.append(ref)
            seen.add(ref)

    return refs


def _article_aware_bonus(query, chunk):
    query_lower = query.lower()
    text = f"{chunk.get('article_id', '')} {chunk.get('title', '')} {chunk.get('text', '')}".lower()

    bonus = 0.0
    article_refs = _extract_article_references(query)
    for article_ref in article_refs:
        if _normalize_article_id(chunk.get("article_id")) == article_ref:
            bonus += 100.0
        if f"article {article_ref.split()[-1]}" in text:
            bonus += 50.0

    if "prohibited ai practices" in text:
        bonus += 25.0

    if "prohibited" in query_lower and "prohibited ai practices" in text:
        bonus += 20.0

    for keyword in [
        "manipulative",
        "deceptive",
        "exploit",
        "vulnerability",
        "social scoring",
        "facial recognition",
        "emotion recognition",
        "biometric categorisation",
        "real-time remote biometric",
        "criminal offence",
    ]:
        if keyword in query_lower and keyword in text:
            bonus += 10.0

    return bonus


def search(query, model, index, chunks, top_k=5):
    article_refs = _extract_article_references(query)
    article_ref_set = {ref.lower() for ref in article_refs}

    article_filtered = []
    if article_refs:
        article_filtered = [
            chunk for chunk in chunks
            if _normalize_article_id(chunk.get("article_id")) in article_ref_set
        ]

    if article_filtered:
        ranked = []
        for chunk in article_filtered:
            text = f"{chunk.get('article_id', '')} {chunk.get('title', '')} {chunk.get('text', '')}"
            query_embedding = model.encode(
                [f"{query} {text}"],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
            score, _ = index.search(query_embedding, 1)
            raw_score = float(score[0][0])
            ranked.append({
                "score": raw_score,
                "article_id": chunk["article_id"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)

        if len(article_refs) > 1:
            results = []
            seen_articles = set()
            for item in ranked:
                if item["article_id"] not in seen_articles:
                    results.append(item)
                    seen_articles.add(item["article_id"])
                if len(results) >= top_k:
                    break

            if len(results) < top_k:
                for item in ranked:
                    if item not in results:
                        results.append(item)
                    if len(results) >= top_k:
                        break
            return results

        return ranked[:top_k]

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append({
            "score": float(score),
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

    return results


def expand_context(results, chunks, window=1):
    """
    Ajoute les chunks voisins des résultats récupérés.
    window=1 : un chunk précédent et un chunk suivant.
    Les scores restent réservés aux résultats de recherche initiaux.
    """
    chunk_positions = {
        (chunk["article_id"], chunk["chunk_index"]): chunk
        for chunk in chunks
    }

    expanded = {}

    for result in results:
        key = (result["article_id"], result["chunk_id"])
        expanded[key] = {
            **result,
            "retrieval_type": "semantic"
        }

        # Retrouver le chunk d'origine.
        seed = next(
            (
                chunk for chunk in chunks
                if chunk["article_id"] == result["article_id"]
                and chunk["chunk_id"] == result["chunk_id"]
            ),
            None
        )

        if seed is None:
            continue

        article_id = seed["article_id"]
        center = seed["chunk_index"]

        for offset in range(-window, window + 1):
            neighbor = chunk_positions.get(
                (article_id, center + offset)
            )

            if neighbor is None:
                continue

            neighbor_key = (
                neighbor["article_id"],
                neighbor["chunk_id"]
            )

            if neighbor_key not in expanded:
                expanded[neighbor_key] = {
                    "score": None,
                    "article_id": neighbor["article_id"],
                    "title": neighbor["title"],
                    "chunk_id": neighbor["chunk_id"],
                    "text": neighbor["text"],
                    "retrieval_type": "neighbor"
                }

    # Présenter les passages dans l'ordre des articles et des chunks.
    chunk_order = {
        (chunk["article_id"], chunk["chunk_id"]): chunk["chunk_index"]
        for chunk in chunks
    }

    return sorted(
        expanded.values(),
        key=lambda item: (
            item["article_id"],
            chunk_order.get(
                (item["article_id"], item["chunk_id"]), 0
            )
        )
    )


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHUNKS_PATH.resolve()}"
        )

    chunks = load_chunks()

    print("Chargement du modèle :", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    build_index(chunks, model)
    index = faiss.read_index(str(INDEX_PATH))

    test_questions = [
        "What AI practices are prohibited by the AI Act?",
        "What are the requirements for high-risk AI systems?",
        "What is an AI system according to the regulation?",
        "What obligations apply to providers of general-purpose AI models?",
        "What penalties can companies face for violating the AI Act?",
    ]

    for query in test_questions:
        print("\n" + "=" * 80)
        print("QUESTION :", query)

        results = search(
            query,
            model,
            index,
            chunks,
            top_k=3
        )

        if not results:
            print("Aucun résultat retourné.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Résultat {rank} ---")
            print("Score :", round(result["score"], 4))
            print("Article :", result["article_id"])
            print("Titre :", result["title"])
            print("Chunk :", result["chunk_id"])
            print("Texte :", result["text"][:250])


if __name__ == "__main__":
    main()


