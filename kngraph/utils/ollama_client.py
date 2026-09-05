from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error, request


DEFAULT_OLLAMA_API_BASE = 'http://127.0.0.1:11434'
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 600


class OllamaError(RuntimeError):
    pass


def _should_log_ollama_progress() -> bool:
    return os.getenv('KG_OLLAMA_PROGRESS', '1').lower() not in {'0', 'false', 'no', 'off'}


def _log_ollama_progress(message: str) -> None:
    if _should_log_ollama_progress():
        print(f'[ollama] {message}', flush=True)


def _extract_json_object_text(raw_text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character not in '{[':
            continue
        try:
            _, end_index = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        return raw_text[index:index + end_index]
    return None


def normalize_ollama_api_base(api_base: str | None = None) -> str:
    candidate = (
        api_base
        or os.getenv('KG_OLLAMA_API_BASE')
        or os.getenv('OLLAMA_HOST')
        or os.getenv('OLLAMA_API_BASE')
        or DEFAULT_OLLAMA_API_BASE
    )
    normalized_candidate = candidate.rstrip('/')
    if '://' not in normalized_candidate:
        normalized_candidate = f'http://{normalized_candidate}'
    normalized_candidate = normalized_candidate.replace('://0.0.0.0', '://127.0.0.1', 1)
    return normalized_candidate


def normalize_ollama_model_name(model: str) -> str:
    if model.startswith('ollama/'):
        return model.split('/', 1)[1]
    return model


def is_ollama_model(model: str | None, api_base: str | None = None) -> bool:
    if not model:
        return False
    if model.startswith('ollama/'):
        return True
    if api_base is None:
        return False
    normalized_api_base = normalize_ollama_api_base(api_base)
    return normalized_api_base.startswith('http://127.0.0.1:11434') or normalized_api_base.startswith('http://localhost:11434')


def ollama_chat_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    api_base: str | None = None,
    temperature: float | None = None,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    progress_label: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        'model': normalize_ollama_model_name(model),
        'messages': messages,
        'stream': True,
        'format': schema,
    }

    if temperature is not None:
        payload['options'] = {'temperature': temperature}

    request_url = f"{normalize_ollama_api_base(api_base)}/api/chat"
    resolved_timeout = int(os.getenv('KG_OLLAMA_TIMEOUT_SECONDS', str(timeout)))
    started_at = time.monotonic()
    request_name = progress_label or normalize_ollama_model_name(model)
    _log_ollama_progress(f'{request_name} 요청 시작')
    request_body = json.dumps(payload).encode('utf-8')
    http_request = request.Request(
        request_url,
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with request.urlopen(http_request, timeout=resolved_timeout) as response:
            response_chunks: list[str] = []
            fallback_chunks: list[str] = []
            chunk_count = 0
            received_visible_output = False
            for raw_line in response:
                decoded_line = raw_line.decode('utf-8').strip()
                if not decoded_line:
                    continue
                response_payload = json.loads(decoded_line)
                if not isinstance(response_payload, dict):
                    continue
                message = response_payload.get('message')
                if isinstance(message, dict):
                    content = message.get('content')
                    if isinstance(content, str) and content:
                        response_chunks.append(content)
                        received_visible_output = True
                    thinking = message.get('thinking')
                    if isinstance(thinking, str) and thinking:
                        fallback_chunks.append(thinking)
                inline_response = response_payload.get('response')
                if isinstance(inline_response, str) and inline_response:
                    response_chunks.append(inline_response)
                    received_visible_output = True
                chunk_count += 1
                if chunk_count == 1:
                    _log_ollama_progress(f'{request_name} 응답 스트림 연결 완료')
                elif chunk_count % 50 == 0:
                    elapsed = time.monotonic() - started_at
                    phase = '출력 생성 중' if received_visible_output else 'thinking 중'
                    _log_ollama_progress(f'{request_name} {phase} ({elapsed:.1f}초 경과, 청크 {chunk_count})')
                if response_payload.get('done'):
                    break
    except error.HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace')
        raise OllamaError(f'Ollama HTTP {exc.code}: {error_body}') from exc
    except error.URLError as exc:
        raise OllamaError(f'Ollama 연결 실패: {exc.reason}') from exc
    except TimeoutError as exc:
        raise OllamaError(
            f'Ollama 응답 대기 시간이 {resolved_timeout}초를 초과했습니다. KG_OLLAMA_TIMEOUT_SECONDS 값을 늘려 다시 시도하세요.'
        ) from exc

    joined_content = ''.join(response_chunks).strip()
    json_text = _extract_json_object_text(joined_content)
    if json_text:
        elapsed = time.monotonic() - started_at
        _log_ollama_progress(f'{request_name} 완료 ({elapsed:.1f}초)')
        return json_text

    fallback_text = ''.join(fallback_chunks).strip()
    fallback_json_text = _extract_json_object_text(fallback_text)
    if fallback_json_text:
        elapsed = time.monotonic() - started_at
        _log_ollama_progress(f'{request_name} 완료 ({elapsed:.1f}초, thinking 파싱)')
        return fallback_json_text

    raise OllamaError('Ollama 응답에서 JSON 텍스트를 찾을 수 없습니다.')