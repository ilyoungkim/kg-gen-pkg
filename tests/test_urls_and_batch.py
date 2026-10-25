"""Offline unit tests for the urls.yaml loader and the batch slug builder.

These do not call any LLM or network endpoint.
"""

from pathlib import Path

import pytest

from src.kngraph.batch import build_slug, select_cases
from src.kngraph.urls import cases_by_url, load_cases, unique_urls


SAMPLE_YAML = """\
cases:
  - case_id: 1
    source: google
    query: abd
    in_sitemap: false
    url: "https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005"
  - case_id: 2
    source: google
    query: abd
    in_sitemap: false
    url: "https://www.ssg.com/page/pc/membershiponly.ssg"
  - case_id: 3
    source: naver
    query: abd
    in_sitemap: true
    url: "https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005"
"""


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "urls.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_cases_parses_all_fields(tmp_path: Path):
    cases = load_cases(_write_yaml(tmp_path, SAMPLE_YAML))

    assert len(cases) == 3
    assert cases[0].case_id == 1
    assert cases[0].source == "google"
    assert cases[0].in_sitemap is False
    assert cases[0].host == "event.ssg.com"


def test_unique_urls_keeps_first_seen_order(tmp_path: Path):
    cases = load_cases(_write_yaml(tmp_path, SAMPLE_YAML))

    urls = unique_urls(cases)

    assert urls == [
        "https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005",
        "https://www.ssg.com/page/pc/membershiponly.ssg",
    ]


def test_cases_by_url_groups_duplicate_urls(tmp_path: Path):
    cases = load_cases(_write_yaml(tmp_path, SAMPLE_YAML))

    grouped = cases_by_url(cases)

    shared = "https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005"
    assert [case.case_id for case in grouped[shared]] == [1, 3]


def test_load_cases_rejects_duplicate_case_ids(tmp_path: Path):
    invalid = SAMPLE_YAML.replace("case_id: 2", "case_id: 1")

    with pytest.raises(ValueError, match="case_id"):
        load_cases(_write_yaml(tmp_path, invalid))


def test_load_cases_rejects_missing_field(tmp_path: Path):
    invalid = SAMPLE_YAML.replace("    in_sitemap: false\n", "", 1)

    with pytest.raises(ValueError, match="expected fields"):
        load_cases(_write_yaml(tmp_path, invalid))


def test_load_cases_rejects_relative_url(tmp_path: Path):
    invalid = SAMPLE_YAML.replace(
        '"https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005"',
        '"/relative/path"',
        1,
    )

    with pytest.raises(ValueError, match="absolute HTTP"):
        load_cases(_write_yaml(tmp_path, invalid))


def test_load_cases_rejects_empty_list(tmp_path: Path):
    with pytest.raises(ValueError, match="non-empty list"):
        load_cases(_write_yaml(tmp_path, "cases: []\n"))


def test_select_cases_filters_by_host(tmp_path: Path):
    cases = load_cases(_write_yaml(tmp_path, SAMPLE_YAML))

    selected = select_cases(cases, only_host="www.ssg.com")

    assert [case.case_id for case in selected] == [2]


def test_select_cases_filters_by_case_id(tmp_path: Path):
    cases = load_cases(_write_yaml(tmp_path, SAMPLE_YAML))

    selected = select_cases(cases, only_case_ids=[1, 3])

    assert [case.case_id for case in selected] == [1, 3]


def test_build_slug_is_filesystem_safe():
    slug = build_slug("https://event.ssg.com/eventDetail.ssg?nevntId=1&domainSiteNo=6005")

    assert slug.startswith("event.ssg.com/")
    assert "?" not in slug
    assert "/" not in slug.split("/", 1)[1]


def test_build_slug_separates_urls_sharing_a_path():
    first = build_slug("https://event.ssg.com/eventDetail.ssg?nevntId=1")
    second = build_slug("https://event.ssg.com/eventDetail.ssg?nevntId=2")

    assert first != second


def test_build_slug_handles_root_path():
    slug = build_slug("https://www.ssg.com")

    assert slug == "www.ssg.com/root"
