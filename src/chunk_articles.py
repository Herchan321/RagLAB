
import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/articles.jsonl")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_SIZE = 400


def split_text(text: str) -> list[str]:
    """Découpe le texte en chunks en privilégiant les frontières naturelles."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + CHUNK_SIZE, text_length)

        if target_end == text_length:
            chunks.append(text[start:].strip())
            break

        # On évite de chercher une frontière trop proche du début.
        min_end = start + MIN_CHUNK_SIZE

        # 1. Priorité : fin de phrase.
        sentence_matches = list(
            re.finditer(r'[.!?]["\')\]]?\s+', text[min_end:target_end])
        )

        if sentence_matches:
            end = min_end + sentence_matches[-1].end()
        else:
            # 2. Sinon : frontière de liste ou de proposition.
            separator_matches = list(
                re.finditer(r'[;:]\s+', text[min_end:target_end])
            )

            if separator_matches:
                end = min_end + separator_matches[-1].end()
            else:
                # 3. Dernier recours : fin de mot.
                whitespace = text.rfind(" ", min_end, target_end)

                if whitespace > start:
                    end = whitespace
                else:
                    # Cas exceptionnel : segment sans espace.
                    next_space = text.find(" ", target_end)

                    end = (
                        next_space
                        if next_space != -1
                        else text_length
                    )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Chevauchement, puis déplacement vers une frontière de mot.
        next_start = max(start + 1, end - CHUNK_OVERLAP)
        next_space = text.find(" ", next_start, min(next_start + 80, text_length))

        if next_space != -1:
            next_start = next_space + 1

        # Protection contre une boucle infinie.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_PATH.resolve()}"
        )

    articles = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                articles.append(json.loads(line))

    all_chunks = []

    for article in articles:
        chunks = split_text(article["text"])

        for index, chunk_text in enumerate(chunks, start=1):
            all_chunks.append({
                "chunk_id": (
                    f"{article['article_id'].replace(' ', '_')}"
                    f"_chunk_{index:03d}"
                ),
                "article_id": article["article_id"],
                "title": article["title"],
                "chunk_index": index,
                "text": chunk_text,
                "source": article["source"]
            })

    if not all_chunks:
        raise ValueError("Aucun chunk généré. Vérifie articles.jsonl.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [len(chunk["text"]) for chunk in all_chunks]

    print("Nombre d'articles :", len(articles))
    print("Nombre de chunks :", len(all_chunks))
    print("Taille moyenne :", round(sum(lengths) / len(lengths), 1))
    print("Chunk le plus court :", min(lengths))
    print("Chunk le plus long :", max(lengths))
    print("Fichier créé :", OUTPUT_PATH.resolve())

    print("\n=== EXEMPLE DE CHUNK ===")
    example = all_chunks[0]
    print("ID :", example["chunk_id"])
    print("Article :", example["article_id"])
    print("Titre :", example["title"])
    print("Texte :", example["text"][:400])


if __name__ == "__main__":
    main()
