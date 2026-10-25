"""Offline unit tests for the answer-engine visibility probe.

These do not call any LLM or network endpoint.
"""

from src.kngraph.qa_probe import (
    _classify_stance,
    _select_references,
    build_context_pack,
    contains_brand,
    mentions_premium_tier,
    summarize_suite,
    ProbeResult,
)


def test_contains_brand_detects_all_surface_forms():
    assert contains_brand("ssg7club")
    assert contains_brand("쓱7클럽")
    assert contains_brand("쓱세븐클럽")
    assert contains_brand("SSG7 Club")
    assert contains_brand("SSG 7 클럽")


def test_contains_brand_ignores_unrelated_text():
    assert not contains_brand("이마트 신세계 아이네utral")
    assert not contains_brand("장바구니 쿠폰 적립")
    assert not contains_brand("")


def test_mentions_premium_tier_only_for_premium_form():
    assert mentions_premium_tier("쓱세븐클럽")
    assert not mentions_premium_tier("쓱7클럽")
    assert not mentions_premium_tier("ssg7club")


def test_build_context_pack_prioritizes_brand_chunks():
    documents = [
        {"url": "https://example.com/a", "text": "쓱7클럽 멤버십 등급과 쿠폰 혜택 안내 페이지입니다. " * 8},
        {"url": "https://example.com/b", "text": "일반적인 회사 소개와 이용 약관 안내 페이지입니다. " * 8},
    ]

    chunks = build_context_pack(documents, max_chunks=8)

    assert chunks
    assert chunks[0]["brandRelevant"] is True
    assert any(chunk["brandRelevant"] is False for chunk in chunks)


def test_select_references_returns_limited_chunk_list():
    chunks = [
        {"url": f"https://example.com/{index}", "text": "쓱7클럽 쿠폰 " * 30} for index in range(20)
    ]

    references = _select_references("쓱7클럽 쿠폰 어떻게 받을까요?", chunks, limit=5)

    assert len(references) == 5
    assert all("example.com" in reference["url"] for reference in references)


def test_select_references_handles_empty_chunks():
    assert _select_references("질문", []) == []


def test_classify_stance_detects_positive_and_absent():
    assert _classify_stance("쓱7클럽은 쿠폰 혜택이 좋아 추천합니다") == "positive"
    assert _classify_stance("이마트도 멤버십이 있습니다") == "absent"


def test_summarize_suite_reports_visibility_lift():
    results = [
        ProbeResult(
            suite="brand",
            question="q",
            category="c",
            intent="i",
            closed_book_answer="쓱7클럽은 멤버십입니다",
            grounded_answer="쓱7클럽은 쿠폰을 제공합니다",
            closed_book_ok=True,
            grounded_ok=True,
            closed_book_brand=True,
            grounded_brand=True,
            grounded_premium=False,
            grounded_cited=True,
            closed_book_stance="neutral",
            grounded_stance="positive",
        ),
        ProbeResult(
            suite="brand",
            question="q2",
            category="c",
            intent="i",
            closed_book_answer="모릅니다",
            grounded_answer="쓱7클럽",
            closed_book_ok=True,
            grounded_ok=True,
            closed_book_brand=False,
            grounded_brand=True,
            grounded_premium=False,
            grounded_cited=True,
            closed_book_stance="absent",
            grounded_stance="neutral",
        ),
    ]

    summary = summarize_suite(results)

    assert summary["total"] == 2
    assert summary["answered"] == 2
    assert summary["closedBookBrandRate"] == 0.5
    assert summary["groundedBrandRate"] == 1.0
    assert summary["visibilityLift"] == 0.5
    assert summary["citationRate"] == 1.0


def test_summarize_suite_handles_empty_results():
    summary = summarize_suite([])

    assert summary["total"] == 0
    assert summary["answered"] == 0
    assert summary["groundedBrandRate"] == 0.0
