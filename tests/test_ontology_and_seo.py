"""Offline unit tests for ontology normalization and the SEO audit."""

from src.kngraph.ontology import (
    build_core_entity_report,
    build_entity_key,
    build_official_name,
    build_ontology_stats,
    build_relation_program,
    normalize_relation_type,
)
from src.kngraph.seo_audit import audit_page_html, summarize_audits


# --------------------------------------------------------------------------
# Ontology normalization
# --------------------------------------------------------------------------


def test_build_entity_key_folds_spacing_and_punctuation():
    assert build_entity_key("26SS 신세계 단독상품") == build_entity_key("26SS신세계단독상품")
    assert build_entity_key("SSG.COM") == build_entity_key("ssg com")
    assert build_entity_key("") == ""


def test_build_entity_key_strips_legal_suffix():
    assert build_entity_key("에스에스지닷컴(주)") == build_entity_key("에스에스지닷컴")


def test_build_official_name_drops_legal_suffix():
    assert build_official_name("루비디(주)") == "루비디"
    assert build_official_name("  SSG  COM  ") == "SSG COM"


def test_normalize_relation_type_maps_common_predicates():
    assert normalize_relation_type("has discount rate") == "DISCOUNTS"
    assert normalize_relation_type("is operated by") == "OPERATES"
    assert normalize_relation_type("provides") == "OFFERS"
    assert normalize_relation_type("is a member of") == "MEMBER_OF"
    assert normalize_relation_type("zzzz nonsense") == "RELATED_TO"
    assert normalize_relation_type("") == "RELATED_TO"


def test_build_ontology_stats_counts_alias_folds():
    entities = ["SSG.COM", "ssg com", "쓱7클럽", "쓱세븐클럽"]
    relations = [("쓱7클럽", "offers", "쿠폰")]

    stats = build_ontology_stats(entities, relations)

    assert stats["entityCount"] == 4
    # "SSG.COM" and "ssg com" fold into one canonical name.
    assert stats["canonicalCount"] == 3
    assert stats["aliasCount"] == 1
    assert stats["relationTypeCount"] == 1
    assert stats["relationTypes"][0]["type"] == "OFFERS"


def test_build_ontology_stats_produces_normalized_facts():
    stats = build_ontology_stats(
        ["쓱7클럽", "쿠폰"],
        [("쓱7클럽", "offers", "쿠폰")],
    )

    assert stats["facts"]
    fact = stats["facts"][0]
    assert fact["subject"] == "쓱7클럽"
    assert fact["object"] == "쿠폰"
    assert fact["relationType"] == "OFFERS"


def test_build_relation_program_keeps_high_degree_nodes():
    entities = ["A", "B", "C", "D"]
    relations = [("A", "links", "B"), ("A", "links", "C"), ("D", "links", "A")]

    program = build_relation_program(entities, relations, max_nodes=2)

    assert program["totalNodes"] == 4
    assert program["totalEdges"] == 3
    assert len(program["nodes"]) == 2
    # The highest-degree node is kept.
    assert program["nodes"][0]["id"] == "A"
    assert program["adjacency"]["A"]


# --------------------------------------------------------------------------
# SEO audit
# --------------------------------------------------------------------------

GOOD_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
  <title>쓱7클럽 멤버십 안내와 혜택 총정리</title>
  <meta name="description" content="쓱7클럽 등급별 쿠폰, 적립, 사은품 혜택과 가입 방법을 안내합니다. 배송과 반품 정책도 함께 확인하세요." />
  <meta property="og:title" content="쓱7클럽 멤버십 안내" />
  <meta property="og:description" content="쓱7클럽 멤버십 등급별 혜택과 가입 방법을 안내하는 페이지입니다." />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://example.com/membership" />
  <meta property="og:image" content="https://example.com/og.png" />
  <meta property="og:site_name" content="Example" />
  <link rel="canonical" href="https://example.com/membership" />
  <script type="application/ld+json">
  {"@context": "https://schema.org", "@type": "Organization", "name": "Example"}
  </script>
</head>
<body>
  <h1>쓱7클럽 멤버십 안내</h1>
  <h2>등급별 혜택</h2>
  <h3>쿠폰 혜택</h3>
  <p>내용</p>
</body>
</html>
"""

POOR_PAGE = """<!DOCTYPE html>
<html>
<head><title>홈</title></head>
<body><p>내용</p></body>
</html>
"""


def test_audit_page_html_passes_a_complete_page():
    audit = audit_page_html(GOOD_PAGE, url="https://example.com/membership")

    assert audit["title"] == "쓱7클럽 멤버십 안내와 혜택 총정리"
    assert audit["h1Count"] == 1
    assert audit["levelSkips"] == []
    assert audit["jsonLdTypes"] == ["Organization"]
    assert audit["canonical"] == "https://example.com/membership"
    assert audit["status"] == "good"
    assert all(check["passed"] for check in audit["checks"])


def test_audit_page_html_flags_missing_signals():
    audit = audit_page_html(POOR_PAGE, url="https://example.com/")

    assert audit["status"] == "poor"
    assert audit["h1Count"] == 0
    assert audit["jsonLdTypes"] == []
    failed = {check["label"] for check in audit["checks"] if not check["passed"]}
    assert "description" in failed
    assert "og:title" in failed
    assert "구조화 데이터" in failed


def test_audit_page_html_detects_heading_level_skips():
    html = "<html><head><title>제목이 충분히 긴 페이지 제목입니다</title></head><body><h1>A</h1><h3>B</h3></body></html>"

    audit = audit_page_html(html, url="https://example.com/")

    assert audit["levelSkips"]
    assert audit["levelSkips"][0] == {"from": 1, "to": 3}


def test_summarize_audits_aggregates_counts():
    audits = [
        audit_page_html(GOOD_PAGE, url="https://example.com/a"),
        audit_page_html(POOR_PAGE, url="https://example.com/b"),
    ]

    summary = summarize_audits(audits)

    assert summary["pageCount"] == 2
    assert summary["goodCount"] == 1
    assert summary["poorCount"] == 1
    assert summary["missingCounts"]["description"] == 1


def test_summarize_audits_handles_empty_input():
    summary = summarize_audits([])

    assert summary["pageCount"] == 0
    assert summary["averageScore"] == 0


# --------------------------------------------------------------------------
# Core entity tracking
# --------------------------------------------------------------------------


def test_core_entity_report_separates_entity_from_products():
    entities = [
        "쓱7클럽",
        "쓱7클럽 특가 할리스 카페라떼 300m",
        "[쓱7클럽] 순두부또띠아 225g",
        "이마트",
    ]
    relations = [
        ("쓱7클럽", "offers", "쿠폰"),
        ("쓱7클럽", "is operated by", "SSG.COM"),
        ("SSG.COM", "has membership", "쓱7클럽"),
    ]

    report = build_core_entity_report(entities, relations)
    group = report["groups"][0]

    assert report["covered"] is True
    # Only the standalone membership is the entity itself.
    assert group["exactEntities"] == ["쓱7클럽"]
    # Product names that merely contain the brand string are counted apart.
    assert group["containingProductCount"] == 2
    assert group["mentionAsSubject"] == 2
    assert group["mentionAsObject"] == 1
    assert group["totalMentions"] == 3


def test_core_entity_report_detects_surface_form_variants():
    entities = ["쓱7클럽", "ssg7club", "SSG7클럽"]
    relations = [("쓱7클럽", "offers", "쿠폰")]

    report = build_core_entity_report(entities, relations)
    group = report["groups"][0]

    assert len(group["exactEntities"]) == 3
    assert group["distinctSurfaceForms"] == 3
    # Different spellings mean the label is not fully consistent.
    assert group["isConsistent"] is False


def test_core_entity_report_marks_missing_entity():
    report = build_core_entity_report(["이마트", "쿠폰"], [])
    group = report["groups"][0]

    assert group["found"] is False
    assert group["totalMentions"] == 0
    assert group["exactEntities"] == []


def test_core_entity_report_is_in_ontology_stats():
    stats = build_ontology_stats(
        ["쓱7클럽", "쿠폰"],
        [("쓱7클럽", "offers", "쿠폰")],
    )

    assert "coreEntities" in stats
    assert stats["coreEntities"]["covered"] is True
