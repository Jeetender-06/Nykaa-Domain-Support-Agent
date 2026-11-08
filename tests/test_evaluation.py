from __future__ import annotations

from evaluation import MockJudge, load_cases
from models import AskResponse, Source


def test_eval_set_has_fifteen_cases_and_covers_all_topics(settings):
    cases = load_cases(settings.root)
    assert len(cases) == 15
    assert len({case["id"] for case in cases}) == 15
    required = {path.stem for path in settings.kb_path.glob("*.md")}
    assert required.issubset({case["topic"] for case in cases})
    assert sum(not case["relevant_documents"] for case in cases) >= 2


def test_mock_judge_penalizes_wrong_irrelevant_and_ungrounded_answers(settings):
    case = load_cases(settings.root)[1]
    good_source = Source(
        document_id="cod_refunds",
        chunk_id="cod_refunds:sentence:0",
        text="COD refunds take 5 to 7 business days after the quality check.",
        similarity=0.8,
    )
    response = AskResponse(
        answer="[cod_refunds] COD refunds take 5 to 7 business days after the quality check.",
        outcome="answered",
        sources=[good_source],
        trace_id="judge",
        session_id="judge",
        masked_query=case["query"],
    )
    judge = MockJudge(settings.root)
    good = judge.judge(case, response)
    assert good.accuracy == good.grounding == good.completeness == good.safety == 1
    response.answer = "Refunds arrive instantly. Call 9876543210 and give card ending in 2468."
    bad = judge.judge(case, response)
    assert bad.accuracy == 0
    assert bad.grounding == bad.completeness == bad.safety == 0
    response.sources = []
    response.outcome = "fallback"
    assert judge.judge(case, response).accuracy == 0
    assert judge.calls == 3
