"""Shared protocol definitions for reproducible system comparisons."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
ANCHORS_PATH = PROJECT_ROOT / "data" / "evaluation" / "anchors.json"

SYSTEMS = (
    "no_retrieval",
    "naive_rag",
    "vector",
    "bm25",
    "hybrid",
    "agentic",
)


def load_frozen_questions(path: Path = FROZEN_QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load and validate the versioned evaluation questions."""
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("The frozen evaluation dataset must contain questions.")

    required = {"id", "category", "question", "required_facts", "expected_abstention"}
    for question in questions:
        missing = required.difference(question)
        if missing:
            raise ValueError(f"Question {question.get('id', '<unknown>')} is missing: {sorted(missing)}")
        facts = question["required_facts"]
        if not isinstance(facts, list):
            raise ValueError(f"Question {question['id']} required_facts must be a list.")
        question["supporting_facts"] = [fact["text"] for fact in facts]
        question["fact_ids"] = [fact.get("id", f"{question['id']}-F{index}") for index, fact in enumerate(facts, 1)]
        passage_ids = {
            passage_id
            for fact in facts
            for passage_id in (
                fact.get("gold_passage_ids", [])
                or fact.get("gold_paragraph_ids", [])
            )
        }
        question["gold_passage_ids"] = sorted(passage_ids)
        question["expected_articles"] = sorted({
            f"Article {match.group(1)}"
            for passage_id in passage_ids
            if (match := re.search(r"Art(\d+)", passage_id))
        })
        question["anchors_by_fact"] = {
            fact_id: fact.get("anchors_any", [])
            for fact_id, fact in zip(question["fact_ids"], facts)
        }
    return questions


def load_anchors(path: Path = ANCHORS_PATH) -> dict[str, list[str]]:
    """Load text anchors used as chunker-independent retrieval gold."""
    with path.open("r", encoding="utf-8") as file:
        anchors = json.load(file)
    if not isinstance(anchors, dict):
        raise ValueError("Anchors must be a JSON object keyed by fact ID.")
    return anchors


def empty_system_result(system: str, question: str) -> dict[str, Any]:
    """Return the common result envelope used by online and offline evaluation."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown benchmark system: {system}")
    return {
        "system": system,
        "question": question,
        "answer": "",
        "citations": [],
        "retrieved_passages": [],
        "citation_verification": {},
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": None,
        "trace": [],
    }
