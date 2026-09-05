"""Configuration and API key management for kngraph runner.

Resolution order for every setting:
    CLI argument > environment variable > user config file
    (~/.config/kngraph/.env) > local .env (via python-dotenv)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_CHUNK_SIZE = 5000
SOURCE_TEXT_LIMIT = 12000
QA_SOURCE_TEXT_LIMIT = 6000
DEFAULT_QA_SERVER_HOST = '127.0.0.1'
DEFAULT_QA_SERVER_PORT = 43870
DEFAULT_OLLAMA_MODEL = 'gemma4:e2b'
DEFAULT_OLLAMA_API_BASE = 'http://127.0.0.1:11434'
DEFAULT_ANALYZE_MODEL = 'openai/gpt-5-nano'
DEFAULT_QA_MODEL = 'openai/gpt-5.6-luna'
BUILD_VERSION_SERIES = '1.0'
BUILD_NUMBER_START = 1
BUILD_VERSION_FILE_NAME = '.build_version.json'
OUTPUT_FILE_LICENSE = 'AI Archive All Right Reserved. since 2026.'
OUTPUT_GENERATOR_NAME = 'kngraph-runner'

CONFIG_DIR_NAME = '.config'
CONFIG_APP_NAME = 'kngraph'
CONFIG_FILE_NAME = '.env'


def get_user_config_dir() -> Path:
    return Path.home() / CONFIG_DIR_NAME / CONFIG_APP_NAME


def get_user_config_path() -> Path:
    return get_user_config_dir() / CONFIG_FILE_NAME


def load_config_files(search_dir: str | Path | None = None) -> None:
    """Load user-level then local .env files (local takes precedence)."""
    load_dotenv(get_user_config_path(), override=False)
    if search_dir is not None:
        load_dotenv(Path(search_dir) / '.env', override=False)
    else:
        load_dotenv(override=False)


def resolve_value(
    cli_value: Any | None,
    env_names: str | list[str],
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    names = [env_names] if isinstance(env_names, str) else env_names
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def resolve_model_settings(
    model: str | None = None,
    locally_ollama: bool = False,
) -> tuple[str, str | None, str | None]:
    if locally_ollama:
        resolved_model = model or os.getenv('KG_OLLAMA_MODEL') or DEFAULT_OLLAMA_MODEL
        api_base = (
            os.getenv('KG_OLLAMA_API_BASE')
            or os.getenv('OLLAMA_HOST')
            or os.getenv('KG_API_BASE')
            or DEFAULT_OLLAMA_API_BASE
        )
        if '://' not in api_base:
            api_base = f'http://{api_base}'
        api_base = api_base.replace('://0.0.0.0', '://127.0.0.1', 1)
        return resolved_model, None, api_base

    resolved_model = model or os.getenv('KG_MODEL', DEFAULT_ANALYZE_MODEL)
    api_key = os.getenv('KG_API_KEY') or os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('KG_API_BASE') or os.getenv('OPENAI_API_BASE')
    return resolved_model, api_key, api_base


def resolve_qa_model(cli_model: str | None = None) -> str:
    return (
        cli_model
        or os.getenv('KG_QA_MODEL')
        or os.getenv('KG_MODEL', DEFAULT_QA_MODEL)
    )


def read_user_config() -> dict[str, str]:
    path = get_user_config_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, _, value = stripped.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def write_user_config(values: dict[str, str]) -> Path:
    config_dir = get_user_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = get_user_config_path()
    existing = read_user_config()
    existing.update({k: v for k, v in values.items() if v})
    lines = [f'{key}={value}' for key, value in sorted(existing.items())]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return path


def mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return '(미설정)'
    if len(api_key) <= 8:
        return '****'
    return f'{api_key[:4]}****{api_key[-4:]}'
