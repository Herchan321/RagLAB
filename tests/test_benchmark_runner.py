from benchmark_runner import _run_system


class FakeModel:
    pass


def test_no_retrieval_does_not_receive_passages():
    question = {
        "id": "Q",
        "question": "Question",
        "expected_articles": [],
        "supporting_facts": [],
    }
    result = _run_system(
        "no_retrieval",
        question,
        chunks=[],
        model=FakeModel(),
        index=None,
    )

    assert result["retrieved_passages"] == []
    assert result["trace"] == [{"step": "no_retrieval", "passages": 0}]
    assert result["generation_status"] == "not_run"
    assert "retrieval_metrics" not in result
    assert "execution_error" not in result

