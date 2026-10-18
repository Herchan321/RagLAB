import json

from evaluate import evaluate_question

questions = [
    {
        "question": "What AI practices are prohibited by Article 5 of the EU AI Act? List all categories.",
        "expected_articles": ["Article 5"],
        "expected_claims": [
            "Article 5 prohibits certain AI practices",
            "Article 5 covers prohibited AI practices",
        ],
        "supporting_facts": [
            "Article 5 prohibits certain AI practices",
            "Article 5 covers prohibited AI practices",
        ],
    },
    {
        "question": "How do Articles 5 and 6 differ?",
        "expected_articles": ["Article 5", "Article 6"],
        "expected_claims": [
            "Article 5 prohibits certain AI practices",
            "Article 6 classifies high-risk AI systems",
        ],
        "supporting_facts": [
            "Article 5 prohibits certain AI practices",
            "Article 6 classifies high-risk AI systems",
        ],
    },
    {
        "question": "What fine applies to a fictional violation described nowhere in the supplied passages?",
        "expected_articles": [],
        "expected_claims": ["The supplied passages do not contain enough information"],
        "supporting_facts": ["The supplied passages do not contain enough information"],
    },
]

rows = []
for item in questions:
    ev = evaluate_question(
        item["question"],
        item["expected_articles"],
        item["expected_claims"],
        item["supporting_facts"],
    )
    rows.append(
        {
            "question": item["question"],
            "vector": ev["vector"]["metrics"],
            "bm25": ev["bm25"]["metrics"],
            "hybrid": ev["hybrid"]["metrics"],
        }
    )

print(json.dumps(rows, ensure_ascii=False, indent=2))
