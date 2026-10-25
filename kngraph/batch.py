"""Batch runner: analyze every URL listed in ``urls.yaml``.

The ssg7 project family keeps a shared ``urls.yaml`` of collected URL cases.
This module turns that list into knowledge graphs: it deduplicates the URLs,
runs the normal crawl -> analyze -> visualize pipeline for each one, and
writes a summary report so a batch run can be audited in one place.

Each URL keeps its own output directory so a failure on one URL never
destroys the results of the others.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from kngraph.urls import Case, cases_by_url, load_cases, unique_urls


DEFAULT_OUTPUT_ROOT = Path("outputs") / "batch"


@dataclass
class BatchItem:
    """One URL's outcome in a batch run."""

    url: str
    host: str
    slug: str
    case_ids: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    status: str = "pending"
    output: str | None = None
    html: str | None = None
    build_version: str | None = None
    graph_summary: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "host": self.host,
            "slug": self.slug,
            "caseIds": self.case_ids,
            "sources": self.sources,
            "status": self.status,
            "output": self.output,
            "html": self.html,
            "buildVersion": self.build_version,
            "graphSummary": self.graph_summary,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "error": self.error,
        }


def build_slug(url: str) -> str:
    """Build a filesystem-safe, collision-free directory name for a URL."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "unknown").replace(":", "_")
    path = re.sub(r"[^0-9A-Za-z]+", "-", parsed.path or "root").strip("-")
    if not path:
        path = "root"
    # The query string distinguishes sibling pages that share a path
    # (eventDetail.ssg?nevntId=... ), so keep a short hash of it.
    if parsed.query:
        suffix = re.sub(r"[^0-9A-Za-z]+", "-", parsed.query).strip("-")
        if suffix:
            path = f"{path}-{suffix[:48]}"
    return f"{host}/{path}"


def select_cases(
    cases: Sequence[Case],
    *,
    only_host: str | None = None,
    only_case_ids: Sequence[int] | None = None,
) -> tuple[Case, ...]:
    """Filter the case list by host and/or explicit case ids."""
    selected = list(cases)
    if only_host:
        normalized = only_host.strip().lower()
        selected = [case for case in selected if case.host.lower() == normalized]
    if only_case_ids:
        wanted = set(only_case_ids)
        selected = [case for case in selected if case.case_id in wanted]
    return tuple(selected)


def run_batch(
    *,
    urls_path: str | Path = "urls.yaml",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model: str | None = None,
    page: int | None = None,
    renew: bool = False,
    verbose: bool = False,
    config_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    only_host: str | None = None,
    only_case_ids: Sequence[int] | None = None,
    dry_run: bool = False,
    browser: bool = False,
    browser_headless: bool = False,
    chrome_path: str | None = None,
    cdp_url: str | None = None,
    profile_dir: str | None = None,
    merge_output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the analyze pipeline for every URL in ``urls_path``.

    Returns a summary dict with one :class:`BatchItem` per unique URL.

    When ``merge_output`` is set, every per-URL ``sources.json`` produced by
    the run is merged into a single ``graph.json``/``graph.html`` under that
    directory (no re-crawl, one LLM pass over the combined documents).
    """
    from kngraph.runner import analyze

    cases = select_cases(
        load_cases(Path(urls_path)),
        only_host=only_host,
        only_case_ids=only_case_ids,
    )
    if not cases:
        raise ValueError("urls.yaml 에서 선택된 케이스가 없습니다.")

    grouped = cases_by_url(cases)
    root = Path(output_root)
    items: list[BatchItem] = []
    started_at = time.time()

    for index, url in enumerate(unique_urls(cases), start=1):
        url_cases = grouped[url]
        slug = build_slug(url)
        item = BatchItem(
            url=url,
            host=url_cases[0].host,
            slug=slug,
            case_ids=[case.case_id for case in url_cases],
            sources=[case.source for case in url_cases],
        )
        items.append(item)

        target_dir = root / slug
        graph_path = target_dir / "graph.json"
        html_path = target_dir / "graph.html"

        if dry_run:
            item.status = "planned"
            print(f"[{index}/{len(grouped)}] (dry-run) {url} -> {graph_path}")
            continue

        print(f"[{index}/{len(grouped)}] {url} -> {graph_path}")
        item_start = time.time()
        try:
            result = analyze(
                url=url,
                model=model,
                renew=renew,
                output=str(graph_path),
                html=str(html_path),
                page=page,
                verbose=verbose,
                open_browser=False,
                config_path=config_path,
                work_dir=work_dir,
                browser=browser,
                browser_headless=browser_headless,
                chrome_path=chrome_path,
                cdp_url=cdp_url,
                profile_dir=profile_dir,
            )
            item.status = "ok"
            item.output = result.get("output")
            item.html = result.get("html")
            item.build_version = result.get("build_version")
            item.graph_summary = result.get("graph_summary") or {}
        except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the batch
            item.status = "failed"
            item.error = str(exc)
            print(f"    실패: {exc}")
        item.elapsed_seconds = time.time() - item_start

    succeeded = [item for item in items if item.status == "ok"]
    failed = [item for item in items if item.status == "failed"]

    summary = {
        "urlsPath": str(urls_path),
        "outputRoot": str(root),
        "crawlMode": "browser" if browser else "http",
        "totalCases": len(cases),
        "uniqueUrls": len(items),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "elapsedSeconds": round(time.time() - started_at, 1),
        "items": [item.to_dict() for item in items],
    }

    if merge_output and not dry_run:
        merged = merge_batch_outputs(
            output_root=root,
            merge_output=merge_output,
            config_path=config_path,
            model=model,
            verbose=verbose,
        )
        summary["mergedOutput"] = merged
    elif merge_output:
        print(f"(dry-run) 병합 예정: {root} -> {Path(merge_output)}", flush=True)

    return summary


def merge_batch_outputs(
    *,
    output_root: str | Path,
    merge_output: str | Path,
    config_path: str | Path | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Merge every per-URL ``sources.json`` under ``output_root`` into one graph.

    Collects all ``sources.json`` files produced by the batch run (one per
    URL slug directory), passes them to :func:`kngraph.regenerate.regenerate`
    so the documents are deduplicated by URL and combined into a single
    ``graph.json`` + ``graph.html`` under ``merge_output``. No re-crawl; the
    LLM extraction runs once over the union of the collected pages.
    """
    from kngraph.regenerate import collect_source_files, regenerate

    root = Path(output_root)
    source_files = collect_source_files(root)
    if not source_files:
        print("병합할 sources.json 을 찾지 못했습니다.", flush=True)
        return {"error": "no sources.json found"}

    merge_dir = Path(merge_output)
    merge_dir.mkdir(parents=True, exist_ok=True)
    graph_path = merge_dir / "graph.json"
    html_path = merge_dir / "graph.html"

    print(
        f"병합 시작: {len(source_files)}개 sources.json -> {html_path}",
        flush=True,
    )
    result = regenerate(
        sources=[str(path) for path in source_files],
        output=str(graph_path),
        html=str(html_path),
        model=model,
        config_path=config_path,
        reuse_graph=False,
        verbose=verbose,
        open_browser=False,
    )

    merged = {
        "output": result.get("output"),
        "html": result.get("html"),
        "buildVersion": result.get("build_version"),
        "sourceDocuments": result.get("source_documents"),
        "graphSummary": result.get("graph_summary"),
    }
    print(f"병합 완료: {html_path}", flush=True)
    return merged


def write_batch_report(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write the batch summary as JSON and Markdown."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "batch_report.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# KG 배치 실행 리포트")
    lines.append("")
    lines.append(f"- URL 목록: `{summary.get('urlsPath', '')}`")
    lines.append(f"- 출력 루트: `{summary.get('outputRoot', '')}`")
    lines.append(f"- 수집 방식 | Crawl mode: {summary.get('crawlMode', 'http')}")
    lines.append(
        f"- 케이스 {summary.get('totalCases', 0)}건 / 고유 URL {summary.get('uniqueUrls', 0)}개 "
        f"(성공 {summary.get('succeeded', 0)}, 실패 {summary.get('failed', 0)})"
    )
    lines.append(f"- 총 소요: {summary.get('elapsedSeconds', 0)}초")
    lines.append("")
    lines.append("| # | URL | 케이스 | 상태 | 엔티티 | 관계 | 결과 |")
    lines.append("|---|---|---|---|---|---|---|")
    for index, item in enumerate(summary.get("items", []), start=1):
        graph_summary = item.get("graphSummary") or {}
        entities = graph_summary.get("entity_count", "-")
        relations = graph_summary.get("relation_count", "-")
        result_path = item.get("html") or item.get("output") or "-"
        error = item.get("error") or ""
        status = item.get("status", "")
        lines.append(
            f"| {index} | {item.get('url', '')} | {','.join(str(c) for c in item.get('caseIds', []))} "
            f"| {status} | {entities} | {relations} | {result_path} {error} |"
        )
    lines.append("")

    markdown_path = destination / "batch_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(markdown_path)}
