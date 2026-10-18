import faiss
from sentence_transformers import SentenceTransformer

from src.agentic_rag import AgenticRAGWorkflow, QuestionRouter, RetrievalPlanner, RetrievalVerifier
from src.rag import _effective_retrieval_mode, _needs_complementary_retrieval, _prioritize_sanction_evidence
from src.retrieval import load_chunks, search


def test_sanction_questions_keep_article_99_in_selection():
    vector_results = [
        {"article_id": "Article 5", "chunk_id": "A5_001", "score": 0.74, "text": "Prohibited AI practices under Article 5."},
        {"article_id": "Article 5", "chunk_id": "A5_002", "score": 0.68, "text": "The prohibited AI practices continue."},
    ]
    hybrid_results = [
        {"article_id": "Article 99", "chunk_id": "Article_99_chunk_003", "combined_score": 0.0161, "text": "Non-compliance with the prohibition of the AI practices referred to in Article 5 shall be subject to administrative fines of up to EUR 35 000 000 or, if the offender is an undertaking, up to 7 % of its total worldwide annual turnover."},
        {"article_id": "Article 101", "chunk_id": "A101_001", "combined_score": 0.0159, "text": "General-purpose AI model fines."},
    ]

    selected = _prioritize_sanction_evidence(
        vector_results,
        hybrid_results,
        'What fine applies to a prohibited AI practice?',
        top_k=8,
    )

    assert selected[0]['article_id'] == 'Article 99', selected
    assert 'Article_99_chunk_003' in selected[0]['chunk_id'], selected


def test_effective_retrieval_mode_switches_to_complementary_for_compare_questions():
    assert _effective_retrieval_mode('How do Articles 5 and 6 differ?', None) == 'complementary'
    assert _effective_retrieval_mode('What AI practices are prohibited by Article 5 of the EU AI Act?', None) == 'vector'
    assert _effective_retrieval_mode('How do Articles 5 and 6 differ?', 'compare') == 'compare'


def test_question_router_routes_multi_article_and_threshold_questions():
    router = QuestionRouter()
    assert router.route('How do Articles 5 and 6 differ?') == 'multi_article_comparison'
    assert router.route('What fine applies to a prohibited AI practice?') == 'legal_threshold'


def test_retrieval_planner_requires_complementary_for_legal_threshold_questions():
    assert RetrievalPlanner().plan('legal_threshold') == 'vector_plus_complementary'
    assert RetrievalPlanner().plan('single_article') == 'vector_only'


def test_retrieval_verifier_rejects_insufficient_evidence():
    verifier = RetrievalVerifier()
    assert verifier.verify([], 'What fine applies to a prohibited AI practice?')[0] is False


def test_retrieval_verifier_requires_article_99_for_fine_questions():
    verifier = RetrievalVerifier()
    results = [{"article_id": "Article 5", "text": "Prohibited AI practice."}]
    ok, reason = verifier.verify(results, 'What fine applies to a prohibited AI practice?')
    assert ok is False
    assert 'Article 99(3)' in reason


def test_needs_complementary_retrieval_for_multi_article_comparison():
    assert _needs_complementary_retrieval('How do Articles 5 and 6 differ?') is True
    assert _needs_complementary_retrieval('What fine applies to a prohibited AI practice?') is True


def test_does_not_need_complementary_retrieval_for_single_article_query():
    assert _needs_complementary_retrieval('What AI practices are prohibited by Article 5 of the EU AI Act?') is False


def test_search_prioritizes_article_5_for_prohibited_practices_query():
    chunks = load_chunks()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index('data/processed/faiss.index')

    query = 'What AI practices are prohibited by Article 5 of the EU AI Act? List all categories.'
    results = search(query, model, index, chunks, top_k=5)

    assert results[0]['article_id'] == 'Article 5', results


def test_search_includes_both_article_5_and_article_6_for_comparison_queries():
    chunks = load_chunks()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index('data/processed/faiss.index')

    query = 'How do Articles 5 and 6 differ?'
    results = search(query, model, index, chunks, top_k=8)
    article_ids = {result['article_id'] for result in results}

    assert 'Article 5' in article_ids, article_ids
    assert 'Article 6' in article_ids, article_ids
