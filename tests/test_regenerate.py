"""Offline unit tests for the regenerate-from-sources helpers."""

import json
from pathlib import Path

from src.kngraph.regenerate import build_links, collect_source_files, load_sources


def _write_sources(path: Path, documents: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_sources_reads_documents(tmp_path: Path):
    path = _write_sources(
        tmp_path / "sources.json",
        [{"url": "https://example.com/a", "text": "본문", "html": "<html></html>"}],
    )

    documents = load_sources([path])

    assert len(documents) == 1
    assert documents[0]["url"] == "https://example.com/a"
    assert documents[0]["html"] == "<html></html>"


def test_load_sources_deduplicates_by_url(tmp_path: Path):
    first = _write_sources(
        tmp_path / "one" / "sources.json",
        [{"url": "https://example.com/a", "text": "본문"}],
    )
    second = _write_sources(
        tmp_path / "two" / "sources.json",
        [
            {"url": "https://example.com/a", "text": "본문"},
            {"url": "https://example.com/b", "text": "다른 본문"},
        ],
    )

    documents = load_sources([first, second])

    assert [document["url"] for document in documents] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_load_sources_skips_entries_without_text(tmp_path: Path):
    path = _write_sources(
        tmp_path / "sources.json",
        [
            {"url": "https://example.com/a", "text": ""},
            {"url": "", "text": "본문"},
            {"url": "https://example.com/c", "text": "본문"},
        ],
    )

    documents = load_sources([path])

    assert len(documents) == 1
    assert documents[0]["url"] == "https://example.com/c"


def test_load_sources_ignores_missing_files(tmp_path: Path):
    assert load_sources([tmp_path / "nope.json"]) == []


def test_load_sources_ignores_malformed_json(tmp_path: Path):
    path = tmp_path / "sources.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_sources([path]) == []


def test_collect_source_files_finds_nested_files(tmp_path: Path):
    _write_sources(tmp_path / "a" / "sources.json", [{"url": "https://x.com", "text": "t"}])
    _write_sources(tmp_path / "b" / "c" / "sources.json", [{"url": "https://y.com", "text": "t"}])
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")

    found = collect_source_files(tmp_path)

    assert len(found) == 2
    assert all(path.name == "sources.json" for path in found)


def test_collect_source_files_handles_missing_root(tmp_path: Path):
    assert collect_source_files(tmp_path / "missing") == []


def test_build_links_lists_document_urls():
    links = build_links(
        [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]
    )

    assert [link["url"] for link in links] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
