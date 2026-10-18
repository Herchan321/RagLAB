
from pathlib import Path
from bs4 import BeautifulSoup
import json

INPUT_PATH = Path("data/raw/regAI.html")
OUTPUT_PATH = Path("data/processed/ai_act_sections.json")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier HTML introuvable : {INPUT_PATH.resolve()}"
        )

    html = INPUT_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n", strip=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            {"document_title": title, "text": text},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Titre :", title)
    print("Nombre de caractères :", len(text))
    print("Fichier créé :", OUTPUT_PATH.resolve())
    print("\nDébut du texte extrait :\n")
    print(text[:2000])


if __name__ == "__main__":
    main()
