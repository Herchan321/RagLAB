#!/usr/bin/env python3
"""Validate and freeze the versioned evaluation question set."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORIES = {
    "single_passage",
    "definition",
    "numeric_threshold",
    "list_enumeration",
    "multi_passage",
    "cross_document",
    "trap_similar",
    "recital_vs_article",
    "unanswerable",
}
STATUSES = {"draft", "reviewed", "frozen"}
ABSTENTION_TYPES = {"out_of_scope", "near_miss"}
REQUIRED = ["id", "split", "category", "question", "expected_abstention", "status", "required_facts"]
LEAK = re.compile(r"\b(supplied|provided|given)\s+(passages?|context|documents?)\b|\bfictional\b", re.I)


def has_todo(value: Any) -> bool:
    return "TODO" in json.dumps(value, ensure_ascii=False)


def test_hash(questions: list[dict[str, Any]]) -> tuple[str, int]:
    tests = sorted((q for q in questions if q.get("split") == "test"), key=lambda q: q["id"])
    canonical = json.dumps(tests, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), len(tests)


def validate(data: dict[str, Any], passage_ids: set[str] | None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    questions = data.get("questions", [])
    is_v3 = str(data.get("schema_version", "")).startswith("3")
    if is_v3 and has_todo(data.get("corpus", {})):
        warnings.append("dataset corpus metadata still contains TODO")
    duplicate_ids = [value for value, count in Counter(q.get("id") for q in questions).items() if count > 1]
    slugs = [q.get("slug") for q in questions if q.get("slug") is not None]
    duplicate_slugs = [value for value, count in Counter(slugs).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    if duplicate_slugs:
        errors.append(f"duplicate slugs: {duplicate_slugs}")

    fact_ids: list[str] = []
    for question in questions:
        question_id = question.get("id", "?")
        missing = [key for key in REQUIRED if key not in question]
        if missing:
            errors.append(f"{question_id}: missing fields {missing}")
            continue
        if question["split"] not in {"dev", "test"}:
            errors.append(f"{question_id}: invalid split")
        if question["category"] not in CATEGORIES:
            errors.append(f"{question_id}: invalid category")
        if question["status"] not in STATUSES:
            errors.append(f"{question_id}: invalid status")
        if LEAK.search(question["question"]):
            errors.append(f"{question_id}: protocol leakage in question")

        facts = question["required_facts"]
        if not isinstance(facts, list):
            errors.append(f"{question_id}: required_facts must be a list")
            continue
        if question["expected_abstention"]:
            if question["category"] != "unanswerable":
                errors.append(f"{question_id}: abstention requires unanswerable category")
            if facts:
                errors.append(f"{question_id}: abstention question has required facts")
            if question.get("abstention_type") not in ABSTENTION_TYPES:
                errors.append(f"{question_id}: invalid abstention_type")
        elif not facts:
            errors.append(f"{question_id}: answerable question has no required facts")

        for fact in facts:
            fact_id = fact.get("id", "?")
            fact_ids.append(fact_id)
            gold_ids = fact.get("gold_passage_ids") or fact.get("gold_paragraph_ids") or []
            anchors = fact.get("anchors_any") or []
            if not gold_ids:
                warnings.append(f"{fact_id}: no human-readable gold paragraph reference")
            if is_v3 and not anchors and not question["expected_abstention"]:
                errors.append(f"{fact_id}: missing anchors_any")
            if fact.get("anchors_verified") is False:
                warnings.append(f"{fact_id}: anchors are not verified")
            if passage_ids is not None and not is_v3:
                missing_passages = [value for value in gold_ids if value not in passage_ids]
                if missing_passages:
                    errors.append(f"{fact_id}: unknown passages {missing_passages}")
            for pattern in fact.get("auto_check", []):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{fact_id}: invalid auto_check ({exc})")

        if question["status"] in {"reviewed", "frozen"} and has_todo(question):
            errors.append(f"{question_id}: reviewed question still contains TODO")
        elif is_v3 and has_todo(question):
            warnings.append(f"{question_id}: draft question contains TODO")
        if question["split"] == "test" and question["status"] == "draft":
            warnings.append(f"{question_id}: test question is still draft")

    duplicate_facts = [value for value, count in Counter(fact_ids).items() if count > 1]
    if duplicate_facts:
        errors.append(f"duplicate fact ids: {duplicate_facts}")

    freeze = data.get("freeze", {})
    if freeze.get("test_sha256"):
        current_hash, count = test_hash(questions)
        if current_hash != freeze["test_sha256"]:
            errors.append("test split changed since freeze")
        if count != freeze.get("n_test"):
            errors.append("test question count changed since freeze")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("passages", type=Path, nargs="?")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    is_v3 = str(data.get("schema_version", "")).startswith("3")
    passage_ids = None
    if args.passages:
        passage_ids = {
            json.loads(line)["id"]
            for line in args.passages.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    errors, warnings = validate(data, passage_ids)
    print(f"{len(data.get('questions', []))} questions")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    if args.freeze:
        tests = [q for q in data["questions"] if q["split"] == "test"]
        not_ready = [q["id"] for q in tests if q["status"] != "reviewed"]
        if is_v3 and len(tests) < 40:
            print(f"FREEZE FAILED: v3 requires at least 40 test questions, found {len(tests)}")
            return 1
        if is_v3:
            unverified = [
                fact.get("id", "?")
                for question in tests
                for fact in question.get("required_facts", [])
                if fact.get("anchors_verified") is not True
            ]
            if unverified:
                print(f"FREEZE FAILED: unverified anchors: {unverified}")
                return 1
        if not tests or not_ready:
            print(f"FREEZE FAILED: test questions not reviewed: {not_ready or 'none'}")
            return 1
        for question in tests:
            question["status"] = "frozen"
        digest, count = test_hash(data["questions"])
        data["freeze"] = {
            "frozen_at": datetime.date.today().isoformat(),
            "n_test": count,
            "test_sha256": digest,
        }
        args.questions.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Test set frozen: {count} questions, hash {digest[:12]}...")
    else:
        print("Validation succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
