"""Crawl -> analyze -> visualize pipeline (extracted from main.py)."""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import nltk

from kngraph.config import (
    BUILD_NUMBER_START,
    BUILD_VERSION_FILE_NAME,
    BUILD_VERSION_SERIES,
    DEFAULT_CHUNK_SIZE,
    OUTPUT_FILE_LICENSE,
    OUTPUT_GENERATOR_NAME,
    SOURCE_TEXT_LIMIT,
    load_config_files,
    resolve_model_settings,
)
from kngraph.qa import normalize_graph_generation_error, start_qa_server
from kngraph.utils.scraping import get_web_content


class UserFacingError(RuntimeError):
    pass


def log_progress(step: int, total_steps: int, message: str) -> None:
    print(f"[{step}/{total_steps}] {message}")


def log_verbose(enabled: bool, message: str, payload: dict | None = None) -> None:
    if not enabled:
        return
    if payload is None:
        print(f"[verbose] {message}")
        return
    print(f"[verbose] {message}: {payload}")


def ensure_nltk_resources():
    nltk_data_dir = Path(__file__).resolve().parent / '.nltk_data'
    nltk_data_dir.mkdir(exist_ok=True)

    nltk_data_path = str(nltk_data_dir)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]

    original_ssl_context = getattr(ssl, '_create_default_https_context', None)
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(resource_name, quiet=True, download_dir=nltk_data_path)
                nltk.data.find(resource_path)
    finally:
        if original_ssl_context is not None:
            ssl._create_default_https_context = original_ssl_context


def ensure_local_kngraph_path():
    # No-op inside the installed package; kept for API compatibility.
    return None


def build_visualization_path(output_path: Path, visualization_path: str | None) -> Path:
    if visualization_path:
        return Path(visualization_path)
    return output_path.with_suffix('.html')


def resolve_target_url(args) -> str | None:
    return args.url or os.getenv('TARGET_URL')


def build_output_path(target_url: str, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)

    parsed = urlparse(target_url)
    directory_name = parsed.netloc or parsed.path or 'default'
    safe_directory_name = directory_name.replace('/', '_').replace(':', '_')
    return Path('outputs') / safe_directory_name / 'graph.json'


def build_source_cache_path(output_path: Path) -> Path:
    return output_path.with_name('sources.json')


def build_version_state_path() -> Path:
    return Path.cwd() / BUILD_VERSION_FILE_NAME


def load_build_version_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None

    try:
        with state_path.open('r', encoding='utf-8') as state_file:
            state = json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(state, dict):
        return None

    return state


def allocate_build_version() -> dict[str, Any]:
    state_path = build_version_state_path()
    previous_state = load_build_version_state(state_path)

    if previous_state:
        previous_build_number = int(previous_state.get('build', 0) or 0)
        if previous_state.get('series') == BUILD_VERSION_SERIES:
            build_number = max(BUILD_NUMBER_START, previous_build_number + 1)
        else:
            # Series changed (e.g., 0.9 -> 1.0): keep the previous build number
            # so the version immediately becomes new_series.previous_build (e.g., 1.0.12).
            build_number = max(BUILD_NUMBER_START, previous_build_number)
    else:
        build_number = BUILD_NUMBER_START

    build_info = {
        'series': BUILD_VERSION_SERIES,
        'build': build_number,
        'version': f'{BUILD_VERSION_SERIES}.{build_number}',
        'builtAt': datetime.now(timezone.utc).isoformat(),
    }

    with state_path.open('w', encoding='utf-8') as state_file:
        json.dump(build_info, state_file, ensure_ascii=False, indent=2)

    return build_info


def build_output_metadata(
    *,
    file_type: str,
    target_url: str,
    build_info: dict[str, Any],
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()

    if file_type == 'json':
        meaning = 'Generated knowledge graph source data for cache reuse and downstream visualization.'
        attributes = [
            'entities',
            'edges',
            'relations',
            'entity_clusters',
            'edge_clusters',
            'entity_metadata',
        ]
    else:
        meaning = 'Generated interactive knowledge graph visualization dashboard.'
        attributes = [
            'interactive graph canvas',
            'stats summary',
            'root entity focus',
            'runtime Q&A tab',
            'build version badge',
        ]

    return {
        'meaning': meaning,
        'fileType': file_type,
        'createdAt': created_at,
        'sourceUrl': target_url,
        'license': OUTPUT_FILE_LICENSE,
        'generatedBy': OUTPUT_GENERATOR_NAME,
        'buildVersion': build_info.get('version'),
        'attributes': attributes,
    }


def build_json_file_comment(output_metadata: dict[str, Any]) -> str:
    return (
        f"Generated {output_metadata['fileType']} for {output_metadata['sourceUrl']} | "
        f"Build {output_metadata['buildVersion']} | {output_metadata['meaning']}"
    )


def save_graph_with_metadata(output_path: Path, graph, output_metadata: dict[str, Any]) -> None:
    graph_payload = graph.model_dump(mode='json')
    payload = {
        '_comment': build_json_file_comment(output_metadata),
        '_meta': output_metadata,
        **graph_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def normalize_source_documents(page_documents: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_documents: list[dict[str, str]] = []
    for page_document in page_documents:
        page_url = page_document.get('url', '').strip()
        page_text = page_document.get('text', '').strip()
        crawl_source = str(page_document.get('crawl_source', '')).strip()
        if not page_url or not page_text:
            continue
        normalized_document = {
            'url': page_url,
            'text': page_text[:SOURCE_TEXT_LIMIT],
        }
        if crawl_source:
            normalized_document['crawl_source'] = crawl_source
        normalized_documents.append(normalized_document)
    return normalized_documents


def source_documents_have_crawl_source(source_documents: list[dict[str, str]]) -> bool:
    if not source_documents:
        return False

    return all(str(source_document.get('crawl_source', '')).strip() for source_document in source_documents)


def load_source_documents(source_cache_path: Path) -> list[dict[str, str]]:
    if not source_cache_path.exists():
        return []

    with source_cache_path.open('r', encoding='utf-8') as source_file:
        source_documents = json.load(source_file)

    if not isinstance(source_documents, list):
        return []

    return normalize_source_documents(source_documents)


def save_source_documents(source_cache_path: Path, page_documents: list[dict[str, str]]) -> list[dict[str, str]]:
    source_documents = normalize_source_documents(page_documents)
    source_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with source_cache_path.open('w', encoding='utf-8') as source_file:
        json.dump(source_documents, source_file, ensure_ascii=False, indent=2)
    return source_documents


def build_system_guideline(config: dict, web_links: list[dict[str, str]]) -> str:
    return f"""
    다음 지침에 따라 지식 그래프를 추출하세요:
    - 중요 키워드: {config['target_keywords']}
    - 허용된 관계 타입: {config['allowed_relationships']}
    - 추출 강조 사항: {config['extraction_focus']}
    - 추가 데이터: 웹 페이지 내 발견된 하이퍼링크 정보 {web_links[:10]} (상위 10개)
    """


def log_collected_pages(page_documents: list[dict[str, str]]) -> None:
    collected_urls = [page_document['url'] for page_document in page_documents]
    print({'collected_page_count': len(collected_urls)})
    print({'collected_page_urls': collected_urls})


def summarize_graph(graph) -> dict[str, int]:
    return {
        'entity_count': len(getattr(graph, 'entities', [])),
        'edge_count': len(getattr(graph, 'edges', [])),
        'relation_count': len(getattr(graph, 'relations', [])),
    }


def generate_graph_from_documents(kg, page_documents, system_guideline: str, verbose: bool = False, no_dspy: bool = False):
    from kngraph.kngraph import DeduplicateMethod, KNGraph
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    def _determine_workers() -> int:
        env = os.getenv('KG_GENERATION_WORKERS')
        if env:
            try:
                val = int(env)
                return max(1, val)
            except Exception:
                pass
        cpu = os.cpu_count() or 1
        if cpu >= 8:
            return 4
        if cpu >= 4:
            return 2
        return 1

    page_graphs: list = []
    total_pages = len(page_documents)
    overall_start = time.time()

    workers = _determine_workers()
    if workers <= 1 or total_pages <= 1:
        # Fallback to sequential processing
        for index, page_document in enumerate(page_documents, start=1):
            page_url = page_document['url']
            page_text = page_document['text']
            log_progress(4, 6, f'페이지별 그래프 생성 시작 ({index}/{total_pages})')
            log_verbose(
                verbose,
                '페이지 그래프 생성 입력',
                {
                    'page_index': index,
                    'page_count': total_pages,
                    'url': page_url,
                    'text_length': len(page_text),
                },
            )
            page_input = f"[PAGE URL] {page_url}\n{page_text}"
            page_context = f"{system_guideline}\n- 현재 처리 페이지: {page_url}"

            page_start = time.time()
            page_graph = kg.generate(
                input_data=page_input,
                context=page_context,
                chunk_size=DEFAULT_CHUNK_SIZE,
                deduplication_method=None,
                no_dspy=no_dspy,
            )
            page_elapsed = time.time() - page_start
            log_progress(4, 6, f'페이지별 그래프 생성 완료 ({index}/{total_pages}) - {page_elapsed:.1f}s')
            page_graphs.append(page_graph)

        total_elapsed = time.time() - overall_start
        log_progress(4, 6, f'페이지 그래프 전체 생성 완료 - 총 {total_elapsed:.1f}s')
    else:
        log_progress(4, 6, f'멀티스레드로 페이지 그래프 생성 중 - 워커: {workers} (총 {total_pages}개)')
        # Use ThreadPoolExecutor and create separate KNGraph instances per thread for safety
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, page_document in enumerate(page_documents, start=1):
                def _task(idx, doc):
                    page_url = doc['url']
                    page_text = doc['text']
                    log_progress(4, 6, f'페이지별 그래프 생성 시작 ({idx}/{total_pages})')
                    log_verbose(
                        verbose,
                        '페이지 그래프 생성 입력',
                        {
                            'page_index': idx,
                            'page_count': total_pages,
                            'url': page_url,
                            'text_length': len(page_text),
                        },
                    )
                    page_input = f"[PAGE URL] {page_url}\n{page_text}"
                    page_context = f"{system_guideline}\n- 현재 처리 페이지: {page_url}"
                    start = time.time()
                    try:
                        # instantiate a fresh KNGraph using same config from the provided kg
                        thread_kg = KNGraph(model=kg.model, api_key=getattr(kg, 'api_key', None), api_base=getattr(kg, 'api_base', None))
                        result = thread_kg.generate(
                            input_data=page_input,
                            context=page_context,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            deduplication_method=None,
                            no_dspy=no_dspy,
                        )
                    finally:
                        elapsed = time.time() - start
                        log_progress(4, 6, f'페이지별 그래프 생성 완료 ({idx}/{total_pages}) - {elapsed:.1f}s')
                    return idx, result

                futures[executor.submit(_task, index, page_document)] = index

            results = {}
            for fut in as_completed(futures):
                idx, page_graph = fut.result()
                results[idx] = page_graph

        # collect results in order
        for idx in range(1, total_pages + 1):
            if idx in results:
                page_graphs.append(results[idx])

        total_elapsed = time.time() - overall_start
        log_progress(4, 6, f'페이지 그래프 전체 생성 완료 - 총 {total_elapsed:.1f}s')

    if not page_graphs:
        raise RuntimeError('크롤링된 페이지 텍스트가 없어 지식 그래프를 생성할 수 없습니다.')

    if len(page_graphs) == 1:
        log_progress(4, 6, '단일 페이지 그래프 중복 제거 중')
        return kg.deduplicate(
            page_graphs[0],
            method=DeduplicateMethod.SEMHASH,
            context=system_guideline,
        )

    log_progress(4, 6, f'{len(page_graphs)}개 페이지 그래프를 통합 중')
    aggregated_graph = kg.aggregate(page_graphs)
    log_progress(4, 6, '통합 그래프 중복 제거 중')
    return kg.deduplicate(
        aggregated_graph,
        method=DeduplicateMethod.SEMHASH,
        context=system_guideline,
    )


def load_or_generate_graph(args, config: dict):
    target_url = resolve_target_url(args)
    if not target_url:
        raise RuntimeError('대상 URL이 없습니다.')

    output_path = build_output_path(target_url, args.output)
    source_cache_path = build_source_cache_path(output_path)

    log_progress(1, 6, '실행 환경을 준비하는 중')
    ensure_nltk_resources()
    ensure_local_kngraph_path()

    from kngraph import KNGraph

    if output_path.exists() and not args.renew:
        log_progress(2, 6, '기존 graph JSON 캐시를 재사용하는 중')
        graph = KNGraph.from_file(str(output_path))
        source_documents = load_source_documents(source_cache_path)
        if not source_documents or not source_documents_have_crawl_source(source_documents):
            log_progress(2, 6, 'Q&A용 URL 소스 페이지를 수집하는 중')
            _, _, page_documents = get_web_content(
                target_url,
                progress_callback=lambda message: log_progress(2, 6, message),
                verbose_callback=(
                    lambda message, payload=None: log_verbose(args.verbose, message, payload)
                ),
                page_limit=args.page,
            )
            source_documents = save_source_documents(source_cache_path, page_documents)
        print({'loaded_from': str(output_path)})
        return KNGraph, graph, source_documents

    log_progress(2, 6, '웹 페이지를 수집하는 중')
    web_text, web_links, page_documents = get_web_content(
        target_url,
        progress_callback=lambda message: log_progress(2, 6, message),
        verbose_callback=(
            lambda message, payload=None: log_verbose(args.verbose, message, payload)
        ),
        page_limit=args.page,
    )
    log_collected_pages(page_documents)
    source_documents = save_source_documents(source_cache_path, page_documents)
    system_guideline = build_system_guideline(config, web_links)

    model, api_key, api_base = resolve_model_settings(
        getattr(args, 'model', None),
        getattr(args, 'locally_ollama', False),
    )
    use_direct_ollama = bool(
        getattr(args, 'locally_ollama', False)
        or (api_base and ':11434' in api_base and '/' not in model)
    )

    log_progress(3, 6, f'지식 그래프 생성기를 초기화하는 중: {model}')
    if api_key:
        print({'model': model, 'has_api_key': True})
        kg = KNGraph(model=model, api_key=api_key, api_base=api_base)
    else:
        print({'model': model, 'has_api_key': False, 'api_base': api_base, 'use_direct_ollama': use_direct_ollama})
        kg = KNGraph(model=model, api_base=api_base)

    try:
        if len(page_documents) > 1:
            log_progress(4, 6, f'멀티페이지 지식 그래프를 생성하는 중: {len(page_documents)}개 페이지')
            graph = generate_graph_from_documents(
                kg,
                page_documents,
                system_guideline,
                verbose=args.verbose,
                no_dspy=use_direct_ollama,
            )
        else:
            log_progress(4, 6, '단일 페이지 지식 그래프를 생성하는 중')
            log_verbose(
                args.verbose,
                '단일 페이지 그래프 생성 입력',
                {
                    'target_url': target_url,
                    'text_length': len(web_text),
                },
            )
            graph = kg.generate(
                input_data=web_text,
                context=system_guideline,
                chunk_size=DEFAULT_CHUNK_SIZE,
                no_dspy=use_direct_ollama,
            )
    except Exception as exc:
        normalized_error_message = normalize_graph_generation_error(exc, model)
        if normalized_error_message != str(exc):
            raise UserFacingError(normalized_error_message) from None
        raise

    log_progress(5, 6, 'graph JSON을 저장하는 중')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_file(str(output_path))
    print({'saved_to': str(output_path)})

    return KNGraph, graph, source_documents

def analyze(
    url: str | None = None,
    model: str | None = None,
    locally_ollama: bool = False,
    renew: bool = False,
    output: str | None = None,
    html: str | None = None,
    page: int | None = None,
    verbose: bool = False,
    open_browser: bool = False,
    config_path: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and return paths + summary."""
    load_config_files(search_dir=work_dir)

    target_url = url or os.getenv('TARGET_URL')
    if not target_url:
        raise UserFacingError('--url 또는 TARGET_URL 이 필요합니다.')

    if config_path is not None:
        config_file = Path(config_path)
    elif work_dir is not None:
        config_file = Path(work_dir) / 'config.json'
    else:
        config_file = Path('config.json')
    with config_file.open('r', encoding='utf-8') as f:
        config = json.load(f)

    qa_runtime_config = start_qa_server(model=model)

    args = type(
        'Args', (), {
            'url': target_url, 'model': model, 'locally_ollama': locally_ollama,
            'renew': renew, 'output': output, 'page': page, 'verbose': verbose,
        },
    )()
    KNGraph, graph, source_documents = load_or_generate_graph(args, config)
    build_info = allocate_build_version()
    output_path = build_output_path(target_url, output)
    graph_output_metadata = build_output_metadata(
        file_type='json', target_url=target_url, build_info=build_info,
    )
    html_output_metadata = build_output_metadata(
        file_type='html', target_url=target_url, build_info=build_info,
    )
    visualization_path = build_visualization_path(output_path, html)

    save_graph_with_metadata(output_path, graph, graph_output_metadata)
    log_progress(6, 6, 'HTML 시각화를 생성하는 중')
    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    KNGraph.visualize(
        graph,
        str(visualization_path),
        open_in_browser=open_browser,
        source_documents=source_documents,
        qa_runtime_config=qa_runtime_config,
        build_info=build_info,
        output_metadata=html_output_metadata,
    )
    return {
        'output': str(output_path),
        'html': str(visualization_path),
        'build_version': build_info['version'],
        'graph_summary': summarize_graph(graph),
    }
