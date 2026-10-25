"""Rebuild knowledge graph outputs from already collected source files.

Crawling is the slow and failure-prone part of the pipeline: a site may
block plain HTTP clients, and every re-run pays for the crawl again.

This module takes the pages that were already collected (the ``sources.json``
files written by a previous run, each holding the page text and its raw
HTML) and regenerates the graph and the HTML dashboard from them — without
touching the network.

Two modes:

``regenerate``
    Re-run the LLM extraction over the collected text to rebuild
    ``graph.json``, then render ``graph.html``.

``--reuse-graph``
    Skip the LLM pass and reuse the existing ``graph.json``; only the
    dashboard is re-rendered. This is the fast path when the graph is
    already correct and only the presentation (or a new feature) changed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from kngraph.config import (
    DEFAULT_CHUNK_SIZE,
    SOURCE_TEXT_LIMIT,
    load_config_files,
    resolve_model_settings,
)


def load_sources(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load and merge ``sources.json`` files, deduplicating by URL."""
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            url = str(item.get('url', '')).strip()
            text = str(item.get('text', '')).strip()
            if not url or not text or url in seen:
                continue
            seen.add(url)
            documents.append(item)

    return documents


def collect_source_files(root: str | Path) -> list[Path]:
    """Find every ``sources.json`` under ``root``."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(base.rglob('sources.json'))


def build_links(documents: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Rebuild a light link list from the collected documents."""
    links: list[dict[str, str]] = []
    for document in documents:
        url = str(document.get('url', ''))
        if url:
            links.append({'text': '', 'url': url})
    return links


def regenerate(
    *,
    sources: Sequence[str | Path],
    output: str | Path,
    html: str | Path | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    reuse_graph: bool = False,
    verbose: bool = False,
    open_browser: bool = False,
    work_dir: str | Path | None = None,
    probe_report: str | Path | None = None,
) -> dict[str, Any]:
    """Regenerate the graph and dashboard from collected source files."""
    load_config_files(search_dir=work_dir)

    source_documents = load_sources(sources)
    if not source_documents:
        raise ValueError('재생성할 원문 파일을 찾지 못했습니다. sources.json 경로를 확인하세요.')

    output_path = Path(output)
    html_path = Path(html) if html else output_path.with_suffix('.html')

    if config_path is not None:
        config_file = Path(config_path)
    elif work_dir is not None:
        config_file = Path(work_dir) / 'config.json'
    else:
        config_file = Path('config.json')
    with config_file.open('r', encoding='utf-8') as config_handle:
        config = json.load(config_handle)

    from kngraph.kngraph import DeduplicateMethod, KNGraph
    from kngraph.runner import build_system_guideline, generate_graph_from_documents

    graph = None
    if reuse_graph and output_path.exists():
        print({'reused_graph': str(output_path)})
        graph = KNGraph.from_file(str(output_path))
    else:
        model_name, api_key, api_base = resolve_model_settings(model)
        print({'model': model_name, 'has_api_key': bool(api_key)})
        kg = KNGraph(model=model_name, api_key=api_key, api_base=api_base) if api_key else KNGraph(
            model=model_name, api_base=api_base
        )

        guideline = build_system_guideline(config, build_links(source_documents))
        started = time.time()

        if len(source_documents) > 1:
            graph = generate_graph_from_documents(
                kg, source_documents, guideline, verbose=verbose
            )
        else:
            document = source_documents[0]
            page_input = f"[PAGE URL] {document['url']}\n{document['text']}"
            page_context = f"{guideline}\n- 현재 처리 페이지: {document['url']}"
            graph = kg.generate(
                input_data=page_input,
                context=page_context,
                chunk_size=DEFAULT_CHUNK_SIZE,
            )

        print({'graph_seconds': round(time.time() - started, 1)})

    from kngraph.runner import (
        allocate_build_version,
        build_output_metadata,
        load_qa_report,
        save_graph_with_metadata,
    )

    build_info = allocate_build_version()

    graph_output_metadata = build_output_metadata(
        file_type='json',
        target_url=str(source_documents[0].get('url', '')),
        build_info=build_info,
    )
    html_output_metadata = build_output_metadata(
        file_type='html',
        target_url=str(source_documents[0].get('url', '')),
        build_info=build_info,
    )

    save_graph_with_metadata(output_path, graph, graph_output_metadata)

    qa_pairs, qa_report = load_qa_report(
        output_path,
        extra_paths=[probe_report] if probe_report else None,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    KNGraph.visualize(
        graph,
        str(html_path),
        open_in_browser=open_browser,
        source_documents=source_documents,
        build_info=build_info,
        output_metadata=html_output_metadata,
        source_html_documents=[
            {'url': document.get('url', ''), 'html': document.get('html', '')}
            for document in source_documents
            if document.get('html')
        ],
        qa_pairs=qa_pairs,
        qa_report=qa_report,
    )

    return {
        'output': str(output_path),
        'html': str(html_path),
        'build_version': build_info['version'],
        'source_documents': len(source_documents),
        'pages_with_html': sum(1 for document in source_documents if document.get('html')),
        'graph_summary': {
            'entity_count': len(getattr(graph, 'entities', [])),
            'edge_count': len(getattr(graph, 'edges', [])),
            'relation_count': len(getattr(graph, 'relations', [])),
        },
    }
