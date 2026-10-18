# RAG Lab

Projet de Retrieval-Augmented Generation pour le corpus du règlement européen sur l’IA (AI Act), avec focus sur la récupération de passages juridiques pertinents et la génération de réponses fondées sur le texte fourni.

## Architecture

- `src/retrieval.py` : recherche vectorielle de base sur FAISS + embeddings SentenceTransformers
- `src/retrieval/bm25_search.py` : recherche lexicale BM25 sur les mots-clés
- `src/retrieval/hybrid_search.py` : fusion des résultats vectoriels et BM25
- `src/rag.py` : pipeline complet RAG pour interroger le modèle Gemini
- `evaluate.py` : script de benchmark pour comparer les méthodes
- `src/eval_metrics.py` : métriques de récupération et de classement

## Logique de la fusion hybride

La recherche hybride ne choisit pas une méthode unique. Elle combine les résultats de deux moteurs :

1. Recherche vectorielle (sémantique)
2. Recherche lexicale BM25 (mots-clés exacts)

Pour chaque document, on récupère sa position dans chaque moteur, puis on fusionne les rangs avec une formule de Reciprocal Rank Fusion (RRF) :

```text
score_fusion(d) = 1 / (k + rank_vector(d)) + 1 / (k + rank_bm25(d))
```

- si le document est bien classé dans les deux moteurs, son score augmente
- s’il est bon dans un seul moteur, il reste présent mais avec un score plus faible
- si le document est mauvais dans les deux, il descend

Cela permet d’exploiter les deux signaux complémentaires :

- vector = similarité sémantique
- BM25 = correspondance lexicale exacte

## Pourquoi ce choix ?

Dans un corpus juridique, les documents importants peuvent être retrouvés par :

- leur sens général
- leurs termes exacts et leur vocabulaire légal
- leur article et leurs notions juridiques spécifiques

La fusion hybride évite de dépendre d’un seul type de recherche.

## Benchmark

The benchmark is executed via `evaluate.py` on the frozen v2 dataset (questions.json). The dataset is based on the original text of Regulation (EU) 2024/1689, published July 12, 2024.

For interactive comparison, see the web app interface in `app.py`. Example results can be seen by running the dev benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark_runner.py --split dev --output data/evaluation/results.dev.json --eval_k 6
```

**`--eval_k` parameter:** When set and the system is `agentic`, the benchmark calculates retrieval metrics on **all retrieved passages** instead of just the top-`k` (default 5). This is useful for studying the agentic system’s broader retrieval strategy. If not set or if agentic returns ≤ `eval_k` passages, the standard `k=5` behavior applies.

### Analysis of failures

A comparison of retrieval systems on key questions reveals typical failure patterns:

**Naive RAG:** Fetches relevant passages from the correct article (e.g., Article 26) but also includes off-topic neighbor passages from other articles (e.g., Article 14 — human oversight is a provider obligation, not a deployer obligation). This illustrates the weakness of purely semantic search without Article‑aware filtering.

**Hybrid:** Stays within the correct article (e.g., Article 26) but limited by the fixed k=5 cutoff. The AI Act Article 26 contains at least 9 relevant chunks (passages), so none of the systems retrieve the full list of obligations with the default k. Points for improvement (tested on dev, not yet frozen for test): expand context, boost k for hybrid, or use neighbor expansion.

**Agentic:** Returns the same 5 passages as Naive RAG, then adds three passages from other articles (e.g., Articles 16 and 20) that begin with "Providers of high‑risk AI systems…". These describe provider obligations (logging, cooperation with authorities, etc.), not deployer obligations, making them noise for a question about deployer duties. The system includes citations of provider responsibilities, which are irrelevant. See the technical trace in the app to see which queries produced them.

**Deterministic tie‑breaking:** When RRF scores from hybrid are equal (e.g., two passages with score 0.0318), order depends on implementation randomness. To ensure reproducible evaluation, break ties deterministically by `chunk_id`.

For interactive questions (e.g., the dev set), there is no ground truth, so relevance labels cannot be trusted. Scores from different retrieval sources (cosine similarity vs. RRF) use different scales, making a single threshold misleading.

These insights come from a 5‑passage window, which caps Recall for a multi‑chunk Article like Article 26. For real‑world deployment, consider expanding the retrieval window (e.g., neighbor chunks, full article, or larger k).
Il compare actuellement :

- vector
- BM25
- hybrid

Le protocole prévoit les six systèmes suivants :

- `no_retrieval` : question envoyée au modèle sans passage récupéré
- `naive_rag` : top-k vectoriel simple puis génération
- `vector` : récupération vectorielle évaluée séparément
- `bm25` : récupération lexicale évaluée séparément
- `hybrid` : fusion vectorielle + BM25
- `agentic` : routage, récupération complémentaire, vérification puis génération

Les deux premières baselines et le système agentique seront exécutés avec le
même jeu gelé et le même format de résultat. Tant que la génération Gemini
n'est pas activée, `evaluate.py` mesure uniquement la qualité de récupération
des trois moteurs disponibles.

Le runner isolé de développement est
[benchmark_runner.py](C:/Users/dell/Desktop/raglab/benchmark_runner.py). Il
exécute chaque système séparément et écrit les résultats dans
`data/evaluation/results.dev.json` :

```powershell
.\.venv\Scripts\python.exe benchmark_runner.py `
  --split dev `
  --output data/evaluation/results.dev.json
```

La génération Gemini est désactivée par défaut. Pour l'activer explicitement
après configuration de `GEMINI_API_KEY` :

```powershell
.\.venv\Scripts\python.exe benchmark_runner.py `
  --split dev `
  --generate
```

Les résultats de développement ne constituent pas encore les résultats finaux
du benchmark : le split `test` doit d'abord être complété et relu.

Les métriques de ranking du runner sont calculées à partir d'ancres textuelles
dans [anchors.json](C:/Users/dell/Desktop/raglab/data/evaluation/anchors.json),
et non à partir des IDs de chunks. Un passage récupéré est pertinent s'il
contient une ancre d'un fait requis. Les IDs lisibles
(`AIA-Art51-p2`, etc.) restent des repères humains et ne servent pas au calcul.
Cette séparation permet de comparer plusieurs stratégies de chunking. La
couverture des articles est conservée comme métrique séparée. Quand Gemini
n'est pas exécuté, les métriques de génération restent `null`/non collectées ;
elles ne sont jamais remplacées par des zéros.

## Dataset v3

Le dataset v3 de développement est conservé dans
[questions.v3.json](C:/Users/dell/Desktop/raglab/data/evaluation/questions.v3.json).
Il contient 55 questions, dont 29 dans le split `test`, et utilise
`anchors_any` comme vérité textuelle indépendante du chunking. Les
`gold_paragraph_ids` sont uniquement des repères humains.

Les ancres doivent être vérifiées contre le corpus avant de passer une question
en `reviewed` :

```powershell
.\.venv\Scripts\python.exe check_anchors.py `
  data/evaluation/questions.v3.json `
  data/processed/aia_passages.jsonl
```

Le runner peut utiliser v3 sans le geler :

```powershell
.\.venv\Scripts\python.exe benchmark_runner.py `
  --questions data/evaluation/questions.v3.json `
  --split dev `
  --output data/evaluation/results.v3.dev.json
```

Le v3 reste un dataset de développement : ses questions sont encore `draft`
et certaines ancres nécessitent une vérification humaine.

## Application

L'interface Streamlit est dans [app.py](C:/Users/dell/Desktop/raglab/app.py).
Elle propose les onglets **Comparer**, **Assistant**, **Résultats benchmark** et
**Méthode et limites**. La comparaison est limitée à trois systèmes par
question. La génération Gemini est désactivée par défaut ; elle nécessite
`GEMINI_API_KEY`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Les métriques comprennent :

- Recall@k
- Precision@k
- MRR@k
- nDCG@k
- article coverage
- groundedness / hallucination rate

Le but est d’évaluer non seulement si le bon document est retrouvé, mais aussi s’il est bien classé dans le top-k.

## État actuel

Le projet a validé :

- recherche vectorielle robuste pour les questions de droit de l’IA
- intégration BM25
- fusion hybride fonctionnelle
- jeu de test gelé v2 avec sept questions, dont deux dans le split `test`
- faits atomiques annotés et identifiants de passages d’or
- protocole de résultat commun pour les comparaisons en ligne et hors ligne
- benchmark de comparaison avec métriques de classement

## Prochaine étape recommandée

- améliorer la fusion hybride avec un reranker ou un routeur
- comparer plus de questions de test
- intégrer le meilleur moteur dans le pipeline RAG final

## Reranking optionnel

La recherche hybride peut maintenant appliquer un reranker cross-encoder après
la fusion RRF. Le reranker doit exposer une méthode `predict` compatible avec
`sentence_transformers.CrossEncoder` :

```python
from sentence_transformers import CrossEncoder
from src.retrieval.hybrid_search import search_hybrid

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
results = search_hybrid(
    query="What AI practices are prohibited?",
    chunks=chunks,
    vector_results=vector_results,
    bm25_results=bm25_results,
    top_k=5,
    reranker=reranker,
)
```

Le reranking est désactivé par défaut afin de préserver le comportement et les
performances existants. Les résultats rerankés contiennent aussi le champ
`reranker_score`.

