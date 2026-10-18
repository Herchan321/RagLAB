from pathlib import Path

from check_anchors import check_anchors


ROOT = Path(__file__).resolve().parents[1]


def test_v3_anchor_checker_reports_unverified_or_missing_anchors():
    errors = check_anchors(
        ROOT / "data" / "evaluation" / "questions.v3.json",
        ROOT / "data" / "processed" / "aia_passages.jsonl",
    )

    assert isinstance(errors, list)
    assert any("Q003-F1" in error for error in errors)
