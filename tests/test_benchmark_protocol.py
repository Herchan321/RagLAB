from src.benchmark_protocol import SYSTEMS, empty_system_result, load_frozen_questions


def test_frozen_dataset_has_required_question_categories():
    questions = load_frozen_questions()

    assert len(questions) >= 5
    assert {question["category"] for question in questions} >= {
        "list_enumeration",
        "trap_similar",
        "numeric_threshold",
        "unanswerable",
    }
    assert any(question["expected_abstention"] for question in questions)
    assert all(question["required_facts"] is not None for question in questions)
    assert all("expected_articles" in question for question in questions)


def test_common_result_envelope_lists_all_comparison_fields():
    result = empty_system_result("hybrid", "test question")

    assert result["system"] == "hybrid"
    assert result["question"] == "test question"
    assert set(result) == {
        "system",
        "question",
        "answer",
        "citations",
        "retrieved_passages",
        "citation_verification",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "trace",
    }


def test_common_result_rejects_unknown_system():
    try:
        empty_system_result("unknown", "question")
    except ValueError as error:
        assert "Unknown benchmark system" in str(error)
    else:
        raise AssertionError("Unknown systems must be rejected.")
