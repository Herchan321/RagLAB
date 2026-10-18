
import json
from pathlib import Path

path = Path("data/processed/articles.jsonl")

with path.open("r", encoding="utf-8") as file:
    articles = [json.loads(line) for line in file if line.strip()]

ids = [article["article_id"] for article in articles]

print("Nombre d'articles :", len(articles))
print("IDs uniques :", len(set(ids)))
print("Doublons :", len(ids) - len(set(ids)))
print("Article le plus court :", min(
    articles, key=lambda article: len(article["text"])
)["article_id"])
print("Article le plus long :", max(
    articles, key=lambda article: len(article["text"])
)["article_id"])

assert len(articles) == 113, "Nombre d'articles inattendu"
assert len(set(ids)) == len(ids), "Des IDs d'articles sont dupliqués"
assert all(a["title"] and a["text"] for a in articles), \
    "Un article est vide"

print("Validation réussie !")
