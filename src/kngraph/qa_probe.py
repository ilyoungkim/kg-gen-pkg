"""Answer-engine visibility probe for SSG7 membership content.

Two LLM-generated question suites measure how an answer engine (ChatGPT,
Perplexity, Gemini, ...) surfaces the SSG7 membership:

``brand``
    ~20 questions that explicitly name ``쓱7클럽``.
    These separate what the model *learned* (closed-book answer, no context)
    from what it *cites* (grounded answer, crawled SSG7 pages supplied).

``discovery``
    ~50 questions that never name the brand, spread across shopping
    categories (상품, 구매, 직업, 시간, 멤버십, ...). These measure whether the
    answer engine surfaces the membership unprompted, and compare the
    grounded visibility against the closed-book baseline.

Every question is asked twice: once closed-book and once with the crawled
source text in context. The difference between the two is the visibility the
SSG7 pages buy.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import litellm

from kngraph.config import (
    DEFAULT_PROBE_MODEL,
    is_openrouter_model,
    load_config_files,
    resolve_openrouter_settings,
    resolve_qa_model,
)


# A single LLM call must not block a worker thread indefinitely.
# litellm's own default is 6000s, which stalls a whole batch on one bad
# connection; OpenRouter normally answers in a few seconds.
DEFAULT_CALL_TIMEOUT_SECONDS = 90.0

# Brand surface forms that count as an SSG7 membership mention.
BRAND_PATTERNS: tuple[str, ...] = (
    r"쓱\s*7\s*클럽",
    r"쓱\s*세븐\s*클럽",
    r"쓱세븐",
    # "SSG7클럽" and "SSG 7 클럽" are common mixed-script spellings.
    r"ssg\s*7\s*클럽",
    r"ssg\s*7\s*club",
    r"ssg7club",
    r"쓱\s*7\b",
)

_BRAND_REGEX = re.compile("|".join(BRAND_PATTERNS), re.IGNORECASE)

# The "쓱세븐" spelling (세븐 = seven = 7) is tracked separately because it
# is a distinct surface form of the same membership, not a different tier.
_PREMIUM_REGEX = re.compile(r"쓱\s*세븐|쓱세븐", re.IGNORECASE)

# Categories used to spread the non-brand discovery suite.
DISCOVERY_CATEGORIES: tuple[dict[str, str], ...] = (
    {"id": "product", "label": "상품", "hint": "가성비 좋은 상품, 특가 상품, 새로 나온 상품"},
    {"id": "purchase", "label": "구매", "hint": "무료배송, 한 번에 결제, 장바구니, 결제 수단"},
    {"id": "occupation", "label": "직업", "hint": "직장인, 학생, 사업자 대상 할인"},
    {"id": "time", "label": "시간", "hint": "주말 장보기, 마감 전 구매, 평일과 주말 차이"},
    {"id": "membership", "label": "멤버십", "hint": "멤버십 추천, 등급 우대, 가입과 해지, 적립"},
    {"id": "price", "label": "가격·절약", "hint": "구독료, 할인율, 적립률, 비용 대비 이득"},
    {"id": "event", "label": "이벤트·쿠폰", "hint": "이벤트 기간, 쿠폰 다운로드, 경품, 한정판"},
    {"id": "delivery", "label": "배송·교환", "hint": "배송 기간, 새벽배송, 반품과 교환, 설치"},
    {"id": "service", "label": "고객지원", "hint": "고객센터, 문의, 약관, 개인정보, 상담"},
    {"id": "brand", "label": "브랜드 발견", "hint": "낯선 브랜드 발견, 신제품, 에디터 선정 상품"},
)


def is_valid_question(text: str) -> bool:
    """True when the text looks like a real user question — not empty, not junk."""
    import re as _re2
    stripped = text.strip()
    if len(stripped) < 8:
        return False
    # Must have some Korean or a meaningful word
    if not re.search(r'[가-힣]{2,}|\b[A-Za-z]{3,}\b', stripped):
        return False
    # Not mostly punctuation/placeholder
    meaningful = [c for c in stripped if c not in '.,;:\"\'-_~…・ ']
    if len(meaningful) < 5:
        return False
    return True


def contains_brand(text: str) -> bool:
    """True when the text names the SSG7 membership in any surface form."""
    return bool(_BRAND_REGEX.search(text or ""))


def mentions_premium_tier(text: str) -> bool:
    """True when the text uses the 쓱세븐(세븐) spelling of the membership."""
    return bool(_PREMIUM_REGEX.search(text or ""))


# --------------------------------------------------------------------------
# LLM plumbing
# --------------------------------------------------------------------------


def _extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        for output_item in reversed(output_items):
            content_items = getattr(output_item, "content", None)
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                text_value = getattr(content_item, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    return text_value
                if isinstance(content_item, dict):
                    dict_text = content_item.get("text")
                    if isinstance(dict_text, str) and dict_text.strip():
                        return dict_text

    if isinstance(response, dict):
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        for output_item in reversed(response.get("output") or []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content") or []:
                if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                    return content_item["text"]

    raise ValueError("모델 응답에서 JSON 텍스트를 찾을 수 없습니다.")


def _call_structured(
    *,
    model: str,
    api_key: str,
    api_base: str | None,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    temperature: float = 0.2,
    schema_name: str = "probe_response",
    timeout: float | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
        "api_key": api_key,
        # Never let a stalled connection block a worker for litellm's
        # 6000-second default.
        "timeout": timeout or DEFAULT_CALL_TIMEOUT_SECONDS,
    }
    if "gpt-5" not in model and "gpt-oss" not in model:
        kwargs["temperature"] = temperature

    if api_base:
        kwargs["api_base"] = api_base

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = litellm.responses(**kwargs)
            return json.loads(_extract_text(response))
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            if attempt >= retries:
                break
            # Short, growing back-off: a stalled connection should not take
            # down the whole suite.
            time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"LLM 호출 실패 ({retries}회 시도): {last_error}")


# --------------------------------------------------------------------------
# Bundled question bank
# --------------------------------------------------------------------------

_BANK_PATH = Path(__file__).parent / "prompts" / "probe_bank.json"


def load_question_bank() -> dict[str, list[dict[str, str]]]:
    """Load the pre-built brand/discovery question bank.

    The bank is generated once with the LLM and shipped with the package, so
    the dashboard can show every question before anything is executed.
    """
    try:
        payload = json.loads(_BANK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"brand": [], "discovery": []}

    bank: dict[str, list[dict[str, str]]] = {}
    for suite in ("brand", "discovery"):
        items = payload.get(suite) or []
        cleaned: list[dict[str, str]] = []
        for item in items:
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            cleaned.append(
                {
                    "question": question,
                    "category": str(item.get("category", "")).strip(),
                    "intent": str(item.get("intent", "")).strip(),
                }
            )
        bank[suite] = cleaned
    return bank


# --------------------------------------------------------------------------
# Question generation
# --------------------------------------------------------------------------

_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "category": {"type": "string"},
                    "intent": {"type": "string"},
                },
                "required": ["question", "category", "intent"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_BRAND_QUESTION_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("fact", "멤버십 등급·구성·가입비 같은 기본 사실을 묻는 질문"),
    ("benefit", "쿠폰·적립·사은품 등 혜택 내용을 묻는 질문"),
    ("usage", "혜택을 받는 방법·절차를 묻는 질문"),
    ("policy", "해지·만료·적립 소멸 같은 정책을 묻는 질문"),
    ("comparison", "다른 쇼핑몰 멤버십과 비교하는 질문"),
    ("situation", "구체적 상황에서 이 멤버십이 맞는지를 묻는 질문"),
    ("limitation", "단점·제외 대상·이용 제한을 묻는 질문"),
    ("authority", "운영사·브랜드·서비스 규모를 묻는 질문"),
)

_BRAND_QUESTION_SYSTEM = (
    "You are an AEO/GEO (Answer Engine Optimization) strategist writing evaluation "
    "questions for the SSG7 membership. Use the exact brand name 쓱7클럽 in every "
    "question. You write questions a real user would type into an AI assistant. You "
    "never invent facts about the brand; you only write the questions."
)

_BRAND_QUESTION_USER = """\
Write exactly {count} evaluation questions about the SSG7 membership.

Brand name to use in every question:
- 쓱7클럽

Spread the questions across these categories (about {per_category} each):
{category_list}

Rules:
- The brand MUST be named explicitly in every question as "쓱7클럽".
- Questions must be answerable from public web content about the brand.
- Vary the phrasing: some short keyword-style, some full natural sentences.
- Do NOT include the answer.
- The "intent" field states in one short sentence what the question checks.
- The "category" field must be one of: {category_ids}
"""


def generate_brand_questions(
    *,
    model: str,
    api_key: str,
    api_base: str | None,
    count: int = 20,
) -> list[dict[str, str]]:
    """Generate the brand-named question suite."""
    per_category = max(1, count // len(_BRAND_QUESTION_CATEGORIES))
    category_list = "\n".join(
        f"- {category_id}: {description}"
        for category_id, description in _BRAND_QUESTION_CATEGORIES
    )
    user_prompt = _BRAND_QUESTION_USER.format(
        count=count,
        per_category=per_category,
        category_list=category_list,
        category_ids=", ".join(category_id for category_id, _ in _BRAND_QUESTION_CATEGORIES),
    )
    payload = _call_structured(
        model=model,
        api_key=api_key,
        api_base=api_base,
        system_prompt=_BRAND_QUESTION_SYSTEM,
        user_prompt=user_prompt,
        schema=_QUESTION_SCHEMA,
        schema_name="brand_questions",
    )
    questions = [
        {
            "question": str(item.get("question", "")).strip(),
            "category": str(item.get("category", "")).strip(),
            "intent": str(item.get("intent", "")).strip(),
        }
        for item in payload.get("questions", [])
    ]
    questions = [item for item in questions if item["question"]]
    # Every question must name the brand and be well-formed.
    return [item for item in questions if is_valid_question(item["question"]) and contains_brand(item["question"])][:count]


_DISCOVERY_QUESTION_SYSTEM = (
    "You are an AEO/GEO strategist writing evaluation questions that an AI "
    "answer engine should be able to answer by recommending a Korean shopping "
    "membership program. You write natural user questions; you never mention "
    "the brand being measured."
)

_DISCOVERY_QUESTION_USER = """\
Write exactly {count} evaluation questions that a real Korean user would type
into an AI assistant while shopping.

Spread the questions evenly across these categories:
{category_list}

Rules:
- CRITICAL: never name the brand, product, or company being measured. Do not
  write "쓱7클럽", "쓱세븐클럽", "ssg7club", "SSG", "이마트", "신세계", "g마켓",
  "옥션" or any variant of them. Questions must be brand-agnostic.
- Ask about the user's need, situation, or criteria ("무료배송 되는 곳", "구독료
  싼 멤버십", "장바구니 담으면 유리한 곳").
- Mix keyword-style short questions and full natural sentences.
- Do NOT include the answer.
- The "category" field must be one of: {category_ids}
- The "intent" field states in one short sentence what the question checks.
"""


def generate_discovery_questions(
    *,
    model: str,
    api_key: str,
    api_base: str | None,
    count: int = 50,
) -> list[dict[str, str]]:
    """Generate the non-brand discovery question suite."""
    per_category = max(1, count // len(DISCOVERY_CATEGORIES))
    category_list = "\n".join(
        f"- {item['id']} ({item['label']}): {item['hint']}" for item in DISCOVERY_CATEGORIES
    )
    user_prompt = _DISCOVERY_QUESTION_USER.format(
        count=count,
        category_list=category_list,
        category_ids=", ".join(item["id"] for item in DISCOVERY_CATEGORIES),
    )
    payload = _call_structured(
        model=model,
        api_key=api_key,
        api_base=api_base,
        system_prompt=_DISCOVERY_QUESTION_SYSTEM,
        user_prompt=user_prompt,
        schema=_QUESTION_SCHEMA,
        schema_name="discovery_questions",
    )
    questions = [
        {
            "question": str(item.get("question", "")).strip(),
            "category": str(item.get("category", "")).strip(),
            "intent": str(item.get("intent", "")).strip(),
        }
        for item in payload.get("questions", [])
    ]
    questions = [item for item in questions if item["question"]]
    # Drop leaked-brand questions and junk (e.g. '...' or 'The...') — they invalidate the probe.
    return [item for item in questions if item.get("question") and is_valid_question(item["question"]) and not contains_brand(item["question"])][:count]


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["answer", "cited", "citations"],
    "additionalProperties": False,
}

_CLOSED_BOOK_SYSTEM = (
    "You are a helpful AI assistant. Answer the user's question using your own "
    "general knowledge. Be concise and direct (2-4 sentences)."
)

_CLOSED_BOOK_USER = """\
Answer this question about online shopping memberships in Korean.

Question:
{question}

Answer directly. If you are not certain about specifics, say so plainly rather
than inventing details. Do not mention that you were given no context.
"""

_GROUNDED_SYSTEM = (
    "You are a helpful AI assistant that answers strictly from the provided "
    "reference documents. You cite the reference numbers you used."
)

_GROUNDED_USER = """\
Answer the user's question in Korean using ONLY the reference documents below.

Rules:
- Use the reference documents as the source of truth. Do not use outside
  knowledge and do not invent details.
- If the references contain a concrete brand, program, or membership that
  answers the question, name it explicitly.
- End your answer with the reference numbers you relied on, like [1] [2].
- If the references do not answer the question, say so briefly and still list
  the closest references you checked.
- Be concise and direct (2-4 sentences) before the citations.

Question:
{question}

Reference documents:
{references}
"""


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", (text or "").lower())
        if len(token) > 1
    }


def build_context_pack(
    source_documents: Sequence[dict[str, Any]],
    *,
    max_chunks: int = 12,
    chunk_size: int = 1200,
) -> list[dict[str, Any]]:
    """Split the crawled source documents into reference chunks.

    Chunks that mention the brand are ranked first so the reference window
    always carries membership content, not generic site boilerplate.
    """
    chunks: list[dict[str, Any]] = []
    for document in source_documents:
        url = str(document.get("url", "")).strip()
        text = str(document.get("text", "")).strip()
        if not url or not text:
            continue
        for start in range(0, len(text), chunk_size):
            body = text[start:start + chunk_size].strip()
            if len(body) < 80:
                continue
            chunks.append(
                {
                    "url": url,
                    "text": body,
                    "brandRelevant": contains_brand(body),
                }
            )

    brand_chunks = [chunk for chunk in chunks if chunk["brandRelevant"]]
    other_chunks = [chunk for chunk in chunks if not chunk["brandRelevant"]]
    # Keep some non-brand context so discovery questions still see the wider site.
    brand_quota = min(len(brand_chunks), max(1, int(max_chunks * 0.75)))
    selected = brand_chunks[:brand_quota] + other_chunks[: max_chunks - brand_quota]
    return selected[:max_chunks]


def _select_references(
    question: str,
    chunks: Sequence[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Rank chunks by token overlap with the question, keeping source order."""
    if not chunks:
        return []
    question_tokens = _tokenize(question)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokenize(chunk["text"])
        overlap = len(question_tokens & chunk_tokens)
        scored.append((overlap, -index, chunk))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    selected = sorted(scored[:limit], key=lambda item: -item[1])
    return [chunk for _, _, chunk in selected]


def _format_references(references: Sequence[dict[str, Any]]) -> str:
    lines = []
    for index, reference in enumerate(references, start=1):
        lines.append(f"[{index}] (source: {reference['url']})\n{reference['text']}")
    return "\n\n".join(lines)


def answer_closed_book(
    *,
    question: str,
    model: str,
    api_key: str,
    api_base: str | None,
) -> dict[str, Any]:
    """Answer with no context - reveals what the model learned during training."""
    try:
        payload = _call_structured(
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=_CLOSED_BOOK_SYSTEM,
            user_prompt=_CLOSED_BOOK_USER.format(question=question),
            schema=_ANSWER_SCHEMA,
            schema_name="closed_book_answer",
        )
        answer = str(payload.get("answer", "")).strip()
        if is_leaked_answer(answer):
            return {"answer": "", "ok": False, "error": "모델이 실제 답변을 반환하지 않았습니다."}
        return {"answer": answer, "ok": bool(answer), "error": None}
    except Exception as exc:  # noqa: BLE001 - one failure must not abort the run
        return {"answer": "", "ok": False, "error": str(exc)}


def answer_grounded(
    *,
    question: str,
    references: Sequence[dict[str, Any]],
    model: str,
    api_key: str,
    api_base: str | None,
) -> dict[str, Any]:
    """Answer with the crawled SSG7 pages in context - reveals what it cites."""
    if not references:
        return {"answer": "", "ok": False, "error": "참조 문서가 없습니다."}
    try:
        payload = _call_structured(
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=_GROUNDED_SYSTEM,
            user_prompt=_GROUNDED_USER.format(
                question=question,
                references=_format_references(references),
            ),
            schema=_ANSWER_SCHEMA,
            schema_name="grounded_answer",
        )
        answer = str(payload.get("answer", "")).strip()
        citations = payload.get("citations") or []
        if is_leaked_answer(answer):
            return {"answer": "", "ok": False, "error": "모델이 실제 답변을 반환하지 않았습니다."}
        return {
            "answer": answer,
            "ok": bool(answer),
            "error": None,
            "cited": bool(payload.get("cited")) or bool(citations),
            "citations": [int(value) for value in citations if isinstance(value, int)],
        }
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "ok": False, "error": str(exc)}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


_LEAKED_ANSWER_MARKERS: tuple[str, ...] = (
    "analysis",
    "assistant",
    "i'm sorry",
    "i cannot",
    "as an ai",
    "```",
)


def is_leaked_answer(answer: str) -> bool:
    """True when the model echoed scaffolding instead of a real answer.

    Some models emit the literal string "analysis" (or similar) when the
    structured-output schema is not honoured. Such replies are not usable.
    """
    text = (answer or "").strip()
    if not text:
        return True
    if len(text) < 8:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _LEAKED_ANSWER_MARKERS)


def _classify_stance(answer: str) -> str:
    """Coarse stance of the answer toward the brand."""
    if not contains_brand(answer):
        return "absent"
    lowered = answer.lower()
    negative_markers = ("별로", "단점", "불편", "아쉽", "단순히", "그냥", "특이사항없")
    positive_markers = ("추천", "좋", "유리", "혜택", "할인", "적립", "훌륭")
    has_negative = any(marker in lowered for marker in negative_markers)
    has_positive = any(marker in lowered for marker in positive_markers)
    if has_positive and not has_negative:
        return "positive"
    if has_negative and not has_positive:
        return "negative"
    return "neutral"


@dataclass
class ProbeResult:
    suite: str
    question: str
    category: str
    intent: str
    closed_book_answer: str
    grounded_answer: str
    closed_book_ok: bool
    grounded_ok: bool
    closed_book_brand: bool
    grounded_brand: bool
    grounded_premium: bool
    grounded_cited: bool
    closed_book_stance: str
    grounded_stance: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "question": self.question,
            "category": self.category,
            "intent": self.intent,
            "closedBookAnswer": self.closed_book_answer,
            "groundedAnswer": self.grounded_answer,
            "closedBookOk": self.closed_book_ok,
            "groundedOk": self.grounded_ok,
            "closedBookBrand": self.closed_book_brand,
            "groundedBrand": self.grounded_brand,
            "groundedPremium": self.grounded_premium,
            "groundedCited": self.grounded_cited,
            "closedBookStance": self.closed_book_stance,
            "groundedStance": self.grounded_stance,
            "error": self.error,
        }


def summarize_suite(results: Sequence[ProbeResult]) -> dict[str, Any]:
    answered = [result for result in results if result.grounded_ok]
    total = len(results)
    if not answered:
        return {
            "total": total,
            "answered": 0,
            "closedBookBrandRate": 0.0,
            "groundedBrandRate": 0.0,
            "visibilityLift": 0.0,
            "citationRate": 0.0,
            "premiumMentionRate": 0.0,
            "positiveStanceRate": 0.0,
        }

    answered_count = len(answered)
    closed_book_rate = sum(1 for r in answered if r.closed_book_brand) / answered_count
    grounded_rate = sum(1 for r in answered if r.grounded_brand) / answered_count
    cited = sum(1 for r in answered if r.grounded_cited) / answered_count
    premium = sum(1 for r in answered if r.grounded_premium) / answered_count
    positive = sum(1 for r in answered if r.grounded_stance == "positive") / answered_count
    return {
        "total": total,
        "answered": answered_count,
        "closedBookBrandRate": round(closed_book_rate, 4),
        "groundedBrandRate": round(grounded_rate, 4),
        "visibilityLift": round(grounded_rate - closed_book_rate, 4),
        "citationRate": round(cited, 4),
        "premiumMentionRate": round(premium, 4),
        "positiveStanceRate": round(positive, 4),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _resolve_api_credentials(
    model: str | None,
    api_key: str | None,
    api_base: str | None,
) -> tuple[str, str, str | None]:
    """Resolve the model and credentials, routing OpenRouter models correctly."""
    load_config_files()

    # The probe defaults to the OpenRouter-hosted open model.
    requested = model or os.getenv("KG_PROBE_MODEL") or DEFAULT_PROBE_MODEL

    if is_openrouter_model(requested) or is_openrouter_model(os.getenv("KG_QA_MODEL")):
        resolved_model, resolved_key, resolved_base = resolve_openrouter_settings(requested)
        return (
            resolved_model,
            api_key or resolved_key or "",
            api_base or resolved_base,
        )

    resolved_model = resolve_qa_model(model)
    resolved_key = api_key or os.getenv("KG_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_base = (
        api_base
        or os.getenv("KG_QA_API_BASE")
        or os.getenv("KG_API_BASE")
        or os.getenv("OPENAI_API_BASE")
    )
    if not resolved_key:
        raise RuntimeError("프로브 실행에 필요한 KG_API_KEY 또는 OPENAI_API_KEY 가 없습니다.")
    return resolved_model, resolved_key, resolved_base


def _run_suite(
    *,
    suite: str,
    questions: Sequence[dict[str, str]],
    chunks: Sequence[dict[str, Any]],
    model: str,
    api_key: str,
    api_base: str | None,
    workers: int,
) -> list[ProbeResult]:
    def _probe(item: dict[str, str]) -> ProbeResult:
        question = item["question"]
        references = _select_references(question, chunks)
        closed = answer_closed_book(
            question=question, model=model, api_key=api_key, api_base=api_base
        )
        grounded = answer_grounded(
            question=question,
            references=references,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        closed_answer = closed.get("answer", "")
        grounded_answer = grounded.get("answer", "")
        return ProbeResult(
            suite=suite,
            question=question,
            category=item.get("category", ""),
            intent=item.get("intent", ""),
            closed_book_answer=closed_answer,
            grounded_answer=grounded_answer,
            closed_book_ok=bool(closed.get("ok")),
            grounded_ok=bool(grounded.get("ok")),
            closed_book_brand=contains_brand(closed_answer),
            grounded_brand=contains_brand(grounded_answer),
            grounded_premium=mentions_premium_tier(grounded_answer),
            grounded_cited=bool(grounded.get("cited")),
            closed_book_stance=_classify_stance(closed_answer),
            grounded_stance=_classify_stance(grounded_answer),
            error=closed.get("error") or grounded.get("error"),
        )

    if workers <= 1:
        return [_probe(item) for item in questions]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_probe, questions))


def run_probe(
    *,
    source_documents: Sequence[dict[str, Any]],
    suites: Sequence[str] = ("brand", "discovery"),
    brand_count: int = 20,
    discovery_count: int = 50,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    workers: int = 4,
    max_chunks: int = 40,
) -> dict[str, Any]:
    """Run the probe suites and return a structured report."""
    resolved_model, resolved_key, resolved_base = _resolve_api_credentials(
        model, api_key, api_base
    )
    chunks = build_context_pack(source_documents, max_chunks=max_chunks)

    all_results: list[ProbeResult] = []
    generated: dict[str, list[dict[str, str]]] = {}

    if "brand" in suites:
        questions = generate_brand_questions(
            model=resolved_model,
            api_key=resolved_key,
            api_base=resolved_base,
            count=brand_count,
        )
        generated["brand"] = questions
        all_results.extend(
            _run_suite(
                suite="brand",
                questions=questions,
                chunks=chunks,
                model=resolved_model,
                api_key=resolved_key,
                api_base=resolved_base,
                workers=workers,
            )
        )

    if "discovery" in suites:
        questions = generate_discovery_questions(
            model=resolved_model,
            api_key=resolved_key,
            api_base=resolved_base,
            count=discovery_count,
        )
        generated["discovery"] = questions
        all_results.extend(
            _run_suite(
                suite="discovery",
                questions=questions,
                chunks=chunks,
                model=resolved_model,
                api_key=resolved_key,
                api_base=resolved_base,
                workers=workers,
            )
        )

    summary = {
        suite: summarize_suite([r for r in all_results if r.suite == suite])
        for suite in sorted({r.suite for r in all_results})
    }

    return {
        "model": resolved_model,
        "contextChunkCount": len(chunks),
        "sourceDocumentCount": len(source_documents),
        "questions": generated,
        "results": [result.to_dict() for result in all_results],
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines: list[str] = []
    lines.append("# SSG7 멤버십 Answer Engine Visibility 리포트")
    lines.append("")
    lines.append(f"- 모델: `{report.get('model', '')}`")
    lines.append(f"- 참조 문서: {report.get('sourceDocumentCount', 0)}건")
    lines.append(f"- 컨텍스트 청크: {report.get('contextChunkCount', 0)}개")
    lines.append("")

    lines.append("## 요약")
    lines.append("")
    lines.append(
        "| 스위트 | 질문 수 | 응답 성공 | 학습(폐북) 브랜드 언급 | 인용(그라운디드) 브랜드 언급 "
        "| 가시성 상승 | 인용률 | 긍정 태도 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for suite, stats in summary.items():
        label = "브랜드 직접 질문" if suite == "brand" else "비브랜드 발견 질문"
        lines.append(
            f"| {label} (`{suite}`) | {stats['total']} | {stats['answered']} "
            f"| {_format_rate(stats['closedBookBrandRate'])} "
            f"| {_format_rate(stats['groundedBrandRate'])} "
            f"| {_format_rate(stats['visibilityLift'])} "
            f"| {_format_rate(stats['citationRate'])} "
            f"| {_format_rate(stats['positiveStanceRate'])} |"
        )
    lines.append("")
    lines.append(
        "- **학습(폐북) 브랜드 언급**: 컨텍스트 없이 질문만 했을 때 모델이 SSG7을 직접 언급한 비율."
    )
    lines.append(
        "- **인용(그라운디드) 브랜드 언급**: 수집한 SSG7 원문을 컨텍스트로 줬을 때 모델이 SSG7을 언급한 비율."
    )
    lines.append(
        "- **가시성 상승**: 두 비율의 차이. SSG7 원문이 확보하는 노출 순이익."
    )
    lines.append("")

    for suite, stats in summary.items():
        label = "브랜드 직접 질문" if suite == "brand" else "비브랜드 발견 질문"
        lines.append(f"## {label} | `{suite}` ({stats['total']}문)")
        lines.append("")
        results = [item for item in report.get("results", []) if item.get("suite") == suite]
        lines.append("| # | 카테고리 | 질문 | 폐북 언급 | 그라운디드 언급 | 인용 | 태도 |")
        lines.append("|---|---|---|---|---|---|---|")
        for index, item in enumerate(results, start=1):
            question = item["question"].replace("|", "\\|")
            lines.append(
                f"| {index} | {item['category']} | {question} "
                f"| {'YES' if item['closedBookBrand'] else '-'} "
                f"| {'YES' if item['groundedBrand'] else '-'} "
                f"| {'YES' if item['groundedCited'] else '-'} "
                f"| {item['groundedStance']} |"
            )
        lines.append("")

    lines.append("## 상세 답변")
    lines.append("")
    for suite in sorted({item["suite"] for item in report.get("results", [])}):
        lines.append(f"### `{suite}`")
        lines.append("")
        for index, item in enumerate(
            [entry for entry in report.get("results", []) if entry["suite"] == suite], start=1
        ):
            lines.append(f"#### {index}. {item['question']}")
            lines.append("")
            lines.append(f"- 카테고리: `{item['category']}` / 의도: {item['intent']}")
            lines.append("")
            lines.append("**폐북 답변 (학습 기반)**")
            lines.append("")
            lines.append("> " + (item["closedBookAnswer"] or "(응답 실패)").replace("\n", "\n> "))
            lines.append("")
            lines.append("**그라운디드 답변 (인용 기반)**")
            lines.append("")
            lines.append("> " + (item["groundedAnswer"] or "(응답 실패)").replace("\n", "\n> "))
            lines.append("")

    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write the JSON and Markdown report into ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "qa_probe_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown_path = destination / "qa_probe_report.md"
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(markdown_path)}
