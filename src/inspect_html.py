
from pathlib import Path
from bs4 import BeautifulSoup

INPUT_PATH = Path("data/raw/regAI.html")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_PATH.resolve()}"
    )

soup = BeautifulSoup(
    INPUT_PATH.read_text(encoding="utf-8"),
    "lxml"
)

print("=== CIBLES REELLES DES ARTICLES ===")

count = 0

for link in soup.find_all("a", href=True):
    label = link.get_text(" ", strip=True)

    if not label.lower().startswith("article "):
        continue

    href = link["href"]

    if not href.startswith("#"):
        continue

    target_id = href[1:]
    target = soup.find(id=target_id)

    print(f"\nLien du sommaire : {label}")
    print(f"href            : {href}")

    if target is None:
        print("Cible            : INTROUVABLE")
    else:
        print(f"Balise cible     : {target.name}")
        print(f"ID cible         : {target.get('id')}")
        print(f"Classes          : {target.get('class')}")
        print(
            "Texte cible      :",
            target.get_text(" ", strip=True)[:350]
        )

    count += 1

    if count >= 8:
        break

print("\n=== ELEMENTS DONT L'ID COMMENCE PAR art_ ===")

elements = soup.find_all(
    lambda tag: tag.get("id", "").startswith("art_")
)

for element in elements[:15]:
    print(
        element.name,
        "| ID:", element.get("id"),
        "| CLASS:", element.get("class"),
        "| TEXTE:", element.get_text(" ", strip=True)[:200]
    )

print("\nInspection terminée.")
