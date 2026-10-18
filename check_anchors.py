"""Check that every annotated text anchor occurs in the frozen corpus passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_anchors(questions_path: Path, passages_path: Path) -> list[str]:
    dataset = json.loads(questions_path.read_text(encoding="utf-8"))
    corpus = passages_path.read_text(encoding="utf-8").lower()
    errors = []
    for question in dataset.get("questions", []):
        for fact in question.get("required_facts", []):
            fact_id = fact.get("id", "?")
            for anchor in fact.get("anchors_any", []):
                if anchor.lower() not in corpus:
                    errors.append(f"{fact_id}: anchor not found: {anchor!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("passages", type=Path)
    args = parser.parse_args()
    errors = check_anchors(args.questions, args.passages)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"{len(errors)} anchor(s) not found.")
        return 1
    print("All anchors found in the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
