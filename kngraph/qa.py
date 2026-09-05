"""Q&A generation (Luna 5-step pipeline) and local QA HTTP server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import litellm

from kngraph.config import (
    DEFAULT_QA_MODEL,
    DEFAULT_QA_SERVER_HOST,
    DEFAULT_QA_SERVER_PORT,
    QA_SOURCE_TEXT_LIMIT,
    load_config_files,
    resolve_model_settings,
    resolve_qa_model,
)


def build_qa_runtime_config(endpoint: str | None, model: str | None) -> dict[str, str | bool | None]:
    return {
        'enabled': bool(endpoint and model),
        'endpoint': endpoint,
        'model': model,
    }


def normalize_graph_generation_error(exc: Exception, model: str) -> str:
    error_text = str(exc)
    lowered_error_text = error_text.lower()

    if 'insufficient_quota' in lowered_error_text or 'exceeded your current quota' in lowered_error_text:
        return (
            f"모델 '{model}' 호출이 OpenAI 사용량 한도 초과로 실패했습니다. "
            'OPENAI_API_KEY/KG_API_KEY 의 과금 상태를 확인하거나, 다른 사용 가능한 API 키로 다시 실행하세요.'
        )

    if 'rate limit' in lowered_error_text:
        return (
            f"모델 '{model}' 호출이 요청 한도에 걸렸습니다. 잠시 후 다시 시도하세요."
        )

    if '403' in lowered_error_text or 'permissiondeniederror' in lowered_error_text:
        return (
            f"모델 '{model}' 호출 권한이 없습니다. 접근 가능한 모델이나 API 키를 확인하세요."
        )

    if 'authentication' in lowered_error_text or 'invalid_api_key' in lowered_error_text:
        return (
            '그래프 생성에 사용하는 API 키가 유효하지 않습니다. OPENAI_API_KEY 또는 KG_API_KEY 를 확인하세요.'
        )

    return error_text


def find_free_port(host: str = DEFAULT_QA_SERVER_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_qa_server(endpoint: str, timeout_seconds: float = 5.0) -> bool:
    import urllib.request
    import urllib.error

    health_url = f'{endpoint}/health'
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.15)
    return False


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, 'output_text', None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = getattr(response, 'output', None)
    if isinstance(output_items, list):
        for output_item in reversed(output_items):
            content_items = getattr(output_item, 'content', None)
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                text_value = getattr(content_item, 'text', None)
                if isinstance(text_value, str) and text_value.strip():
                    return text_value
                if isinstance(content_item, dict):
                    dict_text = content_item.get('text')
                    if isinstance(dict_text, str) and dict_text.strip():
                        return dict_text

    if hasattr(response, 'model_dump'):
        response_dict = response.model_dump(mode='python')
        if isinstance(response_dict, dict):
            return _extract_response_text(response_dict)

    if isinstance(response, dict):
        dict_output_text = response.get('output_text')
        if isinstance(dict_output_text, str) and dict_output_text.strip():
            return dict_output_text

        dict_output_items = response.get('output')
        if isinstance(dict_output_items, list):
            for output_item in reversed(dict_output_items):
                if not isinstance(output_item, dict):
                    continue
                content_items = output_item.get('content')
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    text_value = content_item.get('text')
                    if isinstance(text_value, str) and text_value.strip():
                        return text_value

    raise ValueError('모델 응답에서 JSON 텍스트를 찾을 수 없습니다.')




def normalize_runtime_qa_error(exc: Exception, model: str) -> tuple[str, int]:
    error_text = str(exc)
    lowered_error_text = error_text.lower()

    if 'insufficient_quota' in lowered_error_text or 'exceeded your current quota' in lowered_error_text:
        return (
            f"모델 '{model}' Q&A 호출이 OpenAI 사용량 한도 초과로 실패했습니다. "
            'OPENAI_API_KEY/KG_API_KEY 의 과금 상태를 확인하거나, 다른 사용 가능한 API 키로 다시 실행하세요.',
            429,
        )

    if 'rate limit' in lowered_error_text:
        return (
            f"모델 '{model}' Q&A 호출이 요청 한도에 걸렸습니다. 잠시 후 다시 시도하세요.",
            429,
        )

    if '403' in lowered_error_text or 'permissiondeniederror' in lowered_error_text:
        return (
            f"모델 '{model}' Q&A 호출 권한이 없습니다. 접근 가능한 모델이나 API 키를 확인하세요.",
            403,
        )

    if 'authentication' in lowered_error_text or 'invalid_api_key' in lowered_error_text:
        return (
            'Q&A 생성에 사용하는 API 키가 유효하지 않습니다. OPENAI_API_KEY 또는 KG_API_KEY 를 확인하세요.',
            401,
        )

    return (error_text, 500)




QA_CATEGORY_DEFINITIONS = [
    {
        'id': 'direct',
        'label': '직접 질문',
        'label_en': 'Direct',
        'description': '페이지 주제를 직접 묻는 질문 (예: 혜택이 뭐야, 얼마야, 어떻게 신청해)',
        'share': 0.3,
    },
    {
        'id': 'comparison',
        'label': '비교 질문',
        'label_en': 'Comparison',
        'description': '대안·경쟁·이전 방식과 비교하는 질문 (예: A와 B 중 뭐가 나아, 기존 방식과 뭐가 달라)',
        'share': 0.2,
    },
    {
        'id': 'situation',
        'label': '상황·유즈케이스 질문',
        'label_en': 'Situational',
        'description': '특정 상황·조건에서의 적용을 묻는 질문 (예: 이런 경우에도 돼, 누구에게 좋아)',
        'share': 0.2,
    },
    {
        'id': 'howto',
        'label': '방법·절차 질문',
        'label_en': 'How-to',
        'description': '이용 방법·절차·주의사항을 묻는 질문 (예: 어떻게 해, 뭐가 필요해, 주의할 점은)',
        'share': 0.2,
    },
    {
        'id': 'value',
        'label': '가치·의사결정 질문',
        'label_en': 'Value',
        'description': '할 만한지·이득인지를 묻는 질문 (예: 할 만해, 이득이야, 왜 해야 해)',
        'share': 0.1,
    },
]


def allocate_qa_category_counts(qa_count: int) -> list[dict[str, Any]]:
    allocations: list[dict[str, Any]] = []
    if qa_count < len(QA_CATEGORY_DEFINITIONS):
        # Small counts: one question per front bucket, in order.
        for category in QA_CATEGORY_DEFINITIONS[:qa_count]:
            allocations.append({**category, 'count': 1})
        return allocations
    remaining = qa_count
    for index, category in enumerate(QA_CATEGORY_DEFINITIONS):
        if index == len(QA_CATEGORY_DEFINITIONS) - 1:
            count = remaining
        else:
            count = int(round(qa_count * category['share']))
            count = max(0, min(remaining, count))
            remaining -= count
            if index == 0 and qa_count > 0 and count == 0:
                count = 1
                remaining -= 1
        allocations.append({**category, 'count': count})
    allocated_total = sum(item['count'] for item in allocations)
    if allocated_total < qa_count:
        allocations[0]['count'] += qa_count - allocated_total
    elif allocated_total > qa_count:
        overflow = allocated_total - qa_count
        for item in reversed(allocations):
            if overflow <= 0:
                break
            reducible = min(item['count'], overflow)
            item['count'] -= reducible
            overflow -= reducible
    return [item for item in allocations if item['count'] > 0]


def build_qa_category_guidance(category_counts: list[dict[str, Any]]) -> str:
    lines = []
    for position, item in enumerate(category_counts, start=1):
        lines.append(
            f"{position}. category=\"{item['id']}\" {item['count']}개 - {item['label']}: {item['description']}"
        )
    return '\n'.join(lines)


def generate_runtime_qa(payload: dict, model: str, api_key: str, api_base: str | None = None) -> list[dict[str, Any]]:
    """Luna AEO/GEO question generation with category allocation."""
    requested_count = payload.get('qaCount', 10)
    try:
        qa_count = int(requested_count)
    except (TypeError, ValueError):
        qa_count = 10
    qa_count = max(1, min(30, qa_count))
    category_counts = allocate_qa_category_counts(qa_count)
    category_guidance = build_qa_category_guidance(category_counts)
    source_document = payload.get('sourceDocument') or {}
    source_url = str(source_document.get('url', '')).strip()
    source_text = str(source_document.get('text', '')).strip()
    input_url = str(payload.get('inputUrl', source_url)).strip() or source_url
    selected_entity = str(payload.get('selectedEntityLabel', '')).strip()
    focus_entities = payload.get('focusEntities') or []
    related_relations = payload.get('relatedRelations') or []
    qa_focus = str(payload.get('qaFocus', 'page') or 'page').strip()
    if qa_focus not in ('page', 'entity', 'relation', 'indirect'):
        qa_focus = 'page'
    selected_relation = payload.get('selectedRelation') or {}
    if isinstance(selected_relation, dict):
        selected_relation_text = (
            f"{selected_relation.get('source', '')} -> "
            f"{selected_relation.get('predicate', '')} -> "
            f"{selected_relation.get('target', '')}"
        ).strip()
        if selected_relation_text.strip('-> ') == '':
            selected_relation_text = '없음'
    else:
        selected_relation_text = '없음'
    focus_labels = {'page': '페이지 전체', 'entity': '선택 엔티티', 'relation': '선택 관계', 'indirect': '간접 질문'}
    focus_label = focus_labels[qa_focus]

    focus_entities_text = '\n'.join(f'- {entity}' for entity in focus_entities[:10]) or '- 없음'
    relation_lines = '\n'.join(
        f"- {relation.get('source', '')} -> {relation.get('predicate', '')} -> {relation.get('target', '')}"
        for relation in related_relations[:12]
    ) or '- 없음'

    schema = {
        'type': 'object',
        'properties': {
            'qa_pairs': {
                'type': 'array',
                'minItems': qa_count,
                'maxItems': qa_count,
                'items': {
                    'type': 'object',
                    'properties': {
                        'question': {'type': 'string'},
                        'answer': {'type': 'string'},
                        'category': {'type': 'string'},
                    },
                    'required': [
                        'question', 'answer', 'category',
                    ],
                    'additionalProperties': False,
                },
            }
        },
        'required': ['qa_pairs'],
        'additionalProperties': False,
    }

    system_prompt = (
        'You are Luna, an AEO/GEO question strategist. '
        f'Write {qa_count} Korean questions covering both '
        'direct brand queries and indirect queries (comparisons, situations, how-tos, value judgments), '
        'following the category counts below. '
        f'Generation focus is "{focus_label}". When focus is a selected entity, center questions on it. '
        'When focus is a selected relation, center questions on that triple (source, predicate, target) '
        'and its endpoint neighborhoods. '
        'When focus is indirect ("간접 질문"), do NOT name the page brand or product in the question. '
        'Ask need-based questions (e.g. "쇼핑 혜택이 좋은 멤버십 추천해줘") so that the answer naturally '
        'cites this page as the solution. '
        'Use only the provided page content and graph context. Do not invent facts. '
        'Answers must be concise, direct, quotable, and grounded in the source (2-4 sentences). '
        'Tag each pair with exactly one category id: '
        + ', '.join(f"{item['id']} ({item['label']})" for item in category_counts)
        + '.'
    )

    user_prompt = f"""
입력 URL:
{input_url}

매칭된 소스 URL:
{source_url}

생성 기준:
{focus_label}

선택 엔티티:
{selected_entity or '없음'}

선택 관계:
{selected_relation_text}

페이지 핵심 엔티티 후보:
{focus_entities_text}

그래프 관계 참고:
{relation_lines}

페이지 원문:
<source>
{source_text[:QA_SOURCE_TEXT_LIMIT]}
</source>

요구사항:
- 일반 사용자가 실제로 궁금해할 질문으로 구성
- FAQ나 AI 답변에 바로 사용할 수 있는 명확한 질문/답변
- 마케팅 문구보다 정보 전달 우선
- 답변은 원문과 그래프 문맥을 함께 반영
- 생성 기준이 "선택 관계"이면 해당 관계 triple을 중심으로 질문을 구성
- 생성 기준이 "간접 질문"이면 질문에 브랜드·상품명을 직접 언급하지 말고 니즈 중심으로 묻되, 답변에서는 이 페이지 정보가 인용되도록 구성
- AEO/GEO 노출을 위해 직접 질문뿐 아니라 간접 질문(비교·상황·방법·가치)도 포함
- 아래 카테고리별 개수를 정확히 준수 (합계가 {qa_count}개가 되도록)
- 각 쌍의 category 필드에는 아래 id 중 하나를 그대로 사용

카테고리별 개수:
{category_guidance}
"""

    kwargs = {
        'model': model,
        'input': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'text': {
            'format': {
                'type': 'json_schema',
                'name': 'qa_pairs_response',
                'schema': schema,
                'strict': True,
            }
        },
        'api_key': api_key,
    }

    # Some newer models (eg. gpt-5 family) reject the 'temperature' parameter.
    # Only include it when the model appears to accept it.
    if 'gpt-5' not in model:
        kwargs['temperature'] = 0.2

    if api_base:
        kwargs['api_base'] = api_base

    response = cast(Any, litellm.responses(**kwargs))
    raw_json = _extract_response_text(response)
    parsed = json.loads(raw_json)
    qa_pairs = parsed.get('qa_pairs', [])
    valid_category_ids = {item['id'] for item in category_counts}
    fallback_category = category_counts[0]['id'] if category_counts else 'direct'
    results: list[dict[str, Any]] = []
    for item in qa_pairs:
        question = str(item.get('question', '')).strip()
        answer = str(item.get('answer', '')).strip()
        if not question or not answer:
            continue
        category = str(item.get('category', '')).strip()
        results.append(
            {
                'question': question,
                'answer': answer,
                'category': category if category in valid_category_ids else fallback_category,
            }
        )
    return results


def create_qa_http_handler(model: str, api_key: str, api_base: str | None = None):
    class QARequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            response_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(response_bytes)

        def do_OPTIONS(self):
            self._send_json(200, {'ok': True})

        def do_GET(self):
            if self.path == '/health':
                self._send_json(200, {'ok': True, 'model': model})
                return
            self._send_json(404, {'ok': False, 'error': 'Not found'})

        def do_POST(self):
            if self.path != '/qa':
                self._send_json(404, {'ok': False, 'error': 'Not found'})
                return

            try:
                content_length = int(self.headers.get('Content-Length', '0'))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode('utf-8'))
                qa_pairs = generate_runtime_qa(payload, model=model, api_key=api_key, api_base=api_base)
                self._send_json(200, {'ok': True, 'qaPairs': qa_pairs})
            except Exception as exc:  # noqa: BLE001
                error_message, status_code = normalize_runtime_qa_error(exc, model)
                self._send_json(status_code, {'ok': False, 'error': error_message})

        def log_message(self, format: str, *args):
            return

    return QARequestHandler


def run_qa_server(host: str, port: int, model: str, api_key: str, api_base: str | None = None) -> None:
    handler_class = create_qa_http_handler(model=model, api_key=api_key, api_base=api_base)
    server = ThreadingHTTPServer((host, port), handler_class)
    server.serve_forever()


def start_qa_server(
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    host: str = DEFAULT_QA_SERVER_HOST,
    port: int = DEFAULT_QA_SERVER_PORT,
    executable: str | None = None,
) -> dict[str, str | bool | None]:
    """Start the QA server on a fixed port (43870). Reuse it if already running."""
    load_config_files()
    resolved_model = resolve_qa_model(model)
    resolved_key = api_key or os.getenv('KG_API_KEY') or os.getenv('OPENAI_API_KEY')
    resolved_base = api_base or os.getenv('KG_API_BASE') or os.getenv('OPENAI_API_BASE')
    if not resolved_key:
        return build_qa_runtime_config(None, resolved_model)

    endpoint = f'http://{host}:{port}'
    if wait_for_qa_server(endpoint, timeout_seconds=1.0):
        return build_qa_runtime_config(endpoint, resolved_model)

    env = os.environ.copy()
    env['KG_QA_MODEL'] = resolved_model
    if resolved_base:
        env['KG_QA_API_BASE'] = resolved_base

    runner = executable or sys.executable
    subprocess.Popen(
        [
            runner,
            '-m',
            'kngraph.qa',
            '--host',
            host,
            '--port',
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
        close_fds=True,
    )

    if wait_for_qa_server(endpoint):
        return build_qa_runtime_config(endpoint, resolved_model)

    return build_qa_runtime_config(None, resolved_model)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='kngraph QA server')
    parser.add_argument('--host', default=DEFAULT_QA_SERVER_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_QA_SERVER_PORT)
    parser.add_argument('--model', default=None)
    args = parser.parse_args(argv)

    load_config_files()
    model = resolve_qa_model(args.model)
    api_key = os.getenv('KG_API_KEY') or os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('KG_QA_API_BASE') or os.getenv('KG_API_BASE') or os.getenv('OPENAI_API_BASE')
    if not api_key:
        raise RuntimeError('Q&A 서버 실행에 필요한 KG_API_KEY 또는 OPENAI_API_KEY 가 없습니다.')
    run_qa_server(args.host, int(args.port), model=model, api_key=api_key, api_base=api_base)


if __name__ == '__main__':
    main()

