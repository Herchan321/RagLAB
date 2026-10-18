
from pathlib import Path
from bs4 import BeautifulSoup
import json
import re

INPUT_PATH = Path("data/raw/regAI.html")
OUTPUT_PATH = Path("data/processed/articles.jsonl")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier HTML introuvable : {INPUT_PATH.resolve()}"
        )

    soup = BeautifulSoup(
        INPUT_PATH.read_text(encoding="utf-8"),
        "lxml"
    )

    # Le contenu juridique principal se trouve dans docHtml.
    document = soup.find(id="docHtml")

    if document is None:
        raise ValueError("La zone principale docHtml est introuvable.")

    articles = []

    # Les articles principaux utilisent des IDs de type art_1, art_2...
    for element in document.find_all(
        "div",
        class_="eli-subdivision",
        id=re.compile(r"^art_\d+$")
    ):
        article_id = element.get("id")
        number = article_id.removeprefix("art_")

        # Le titre est stocké dans un élément enfant.
        title_element = element.find(
            id=f"{article_id}.tit_1"
        )

        title = (
            title_element.get_text(" ", strip=True)
            if title_element
            else ""
        )
        title = title.rstrip("` ").strip()
        # Copier l'élément pour ne pas modifier le document original.
        content_element = BeautifulSoup(
            str(element), "lxml"
        ).find(id=article_id)

        # Supprimer le titre de la copie pour ne conserver
        # que le contenu juridique dans le champ text.
        if content_element:
            copied_title = content_element.find(
                id=f"{article_id}.tit_1"
            )
            if copied_title:
                copied_title.decompose()

            text = content_element.get_text(" ", strip=True)
        else:
            text = element.get_text(" ", strip=True)

        # Nettoyage des espaces.
        text = re.sub(r"\s+", " ", text).strip()
        title = re.sub(r"\s+", " ", title).strip()

        text = re.sub(rf"^Article\s+{re.escape(number)}\s*",
    "", text).strip()


        articles.append({
            "article_id": f"Article {number}",
            "title": title,
            "text": text,
            "source": "ai_act.html"
        })

    if not articles:
        raise ValueError(
            "Aucun article trouvé. Vérifie la structure du HTML."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )

    print("Nombre d'articles extraits :", len(articles))
    print("Fichier créé :", OUTPUT_PATH.resolve())

    print("\n=== APERÇU DES 3 PREMIERS ARTICLES ===")
    for article in articles[:3]:
        print("\nID :", article["article_id"])
        print("Titre :", article["title"])
        print("Texte :", article["text"][:300])

    print("\n=== CONTRÔLE DE QUALITÉ ===")
    print(
        "Articles sans titre :",
        sum(not a["title"] for a in articles)
    )
    print(
        "Articles sans texte :",
        sum(not a["text"] for a in articles)
    )


if __name__ == "__main__":
    main()
