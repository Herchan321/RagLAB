from pathlib import Path

from validate_questions import validate


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_questions_reference_existing_stable_passages():
    import json

    questions = json.loads(
        (ROOT / "data" / "evaluation" / "questions.json").read_text(encoding="utf-8")
    )
    passage_ids = {
        json.loads(line)["id"]
        for line in (ROOT / "data" / "processed" / "aia_passages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    errors, warnings = validate(questions, passage_ids)

    assert errors == []
    assert any("Q005" in warning for warning in warnings)
