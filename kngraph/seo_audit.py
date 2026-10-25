"""SEO / structured-data audit for crawled pages and generated output.

Checks the elements that decide how a page is understood and cited:

* ``title`` — presence, length, and whether it names the page subject.
* ``description`` — presence and length.
* ``Open Graph`` — ``og:title``, ``og:description``, ``og:type``, ``og:url``,
  ``og:image``.
* ``Heading hierarchy`` — ``h1`` count, level skips (``h1`` -> ``h3``).
* ``Structured data`` — JSON-LD blocks and their schema.org types.

The audit runs against the raw HTML of each crawled page, and the generated
HTML output is checked with the same rules so the two can be compared.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from bs4 import BeautifulSoup


TITLE_MIN_LENGTH = 10
TITLE_MAX_LENGTH = 60
DESCRIPTION_MIN_LENGTH = 50
DESCRIPTION_MAX_LENGTH = 160

OG_PROPERTIES = (
    "og:title",
    "og:description",
    "og:type",
    "og:url",
    "og:image",
    "og:site_name",
)

_HEADING_PATTERN = re.compile(r"<h([1-6])\b", re.IGNORECASE)
_JSON_LD_PATTERN = re.compile(
    r'<script[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _meta_content(soup: BeautifulSoup, **attrs: str) -> str:
    meta = soup.find("meta", attrs=attrs)
    if meta is None:
        return ""
    return _clean(meta.get("content"))


def _collect_heading_levels(soup: BeautifulSoup) -> list[int]:
    levels: list[int] = []
    for heading in soup.find_all(re.compile(r"^h[1-6]$", re.IGNORECASE)):
        try:
            levels.append(int(heading.name[1]))
        except (ValueError, IndexError):
            continue
    return levels


def _collect_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Parse every JSON-LD block and report its schema.org types."""
    blocks: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            blocks.append({"valid": False, "types": [], "error": "빈 JSON-LD 블록"})
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            blocks.append({"valid": False, "types": [], "error": f"JSON 파싱 실패: {exc.msg}"})
            continue

        types: list[str] = []
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            if isinstance(item_type, str):
                types.append(item_type)
            elif isinstance(item_type, list):
                types.extend(str(value) for value in item_type)
        blocks.append({"valid": True, "types": types, "error": None})
    return blocks


def audit_page_html(
    html: str,
    *,
    url: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Audit one page's SEO / structured-data signals."""
    soup = BeautifulSoup(html or "", "html.parser")

    page_title = title if title is not None else _clean(soup.title.get_text() if soup.title else "")
    description = _meta_content(soup, name="description") or _meta_content(
        soup, property="og:description"
    )
    canonical = ""
    canonical_link = soup.find("link", attrs={"rel": "canonical"})
    if canonical_link is not None:
        canonical = _clean(canonical_link.get("href"))

    og_values = {
        prop: _meta_content(soup, property=prop) for prop in OG_PROPERTIES
    }

    heading_levels = _collect_heading_levels(soup)
    h1_count = heading_levels.count(1)

    level_skips: list[dict[str, Any]] = []
    for index in range(1, len(heading_levels)):
        previous = heading_levels[index - 1]
        current = heading_levels[index]
        if current > previous + 1:
            level_skips.append({"from": previous, "to": current})

    json_ld_blocks = _collect_json_ld(soup)
    json_ld_types = sorted(
        {
            schema_type
            for block in json_ld_blocks
            for schema_type in block.get("types", [])
        }
    )
    invalid_json_ld = [block for block in json_ld_blocks if not block.get("valid")]

    # ---- checks ----
    checks: list[dict[str, Any]] = []

    def add_check(
        key: str,
        label: str,
        passed: bool,
        detail: str,
        severity: str = "warning",
    ) -> None:
        checks.append(
            {
                "key": key,
                "label": label,
                "passed": passed,
                "detail": detail,
                "severity": severity,
            }
        )

    add_check(
        "title",
        "title",
        bool(page_title),
        f"제목: {page_title[:80]}" if page_title else "title 태그가 없습니다.",
        "critical" if not page_title else "warning",
    )
    if page_title:
        if len(page_title) < TITLE_MIN_LENGTH:
            add_check(
                "titleLength",
                "title 길이",
                False,
                f"{len(page_title)}자 — {TITLE_MIN_LENGTH}자 미만입니다.",
            )
        elif len(page_title) > TITLE_MAX_LENGTH:
            add_check(
                "titleLength",
                "title 길이",
                False,
                f"{len(page_title)}자 — {TITLE_MAX_LENGTH}자 초과(검색 결과 잘림 위험).",
            )
        else:
            add_check("titleLength", "title 길이", True, f"{len(page_title)}자 (권장 범위)")

    add_check(
        "description",
        "description",
        bool(description),
        f"설명: {description[:80]}" if description else "meta description이 없습니다.",
        "critical" if not description else "warning",
    )
    if description:
        if len(description) < DESCRIPTION_MIN_LENGTH:
            add_check(
                "descriptionLength",
                "description 길이",
                False,
                f"{len(description)}자 — {DESCRIPTION_MIN_LENGTH}자 미만입니다.",
            )
        elif len(description) > DESCRIPTION_MAX_LENGTH:
            add_check(
                "descriptionLength",
                "description 길이",
                False,
                f"{len(description)}자 — {DESCRIPTION_MAX_LENGTH}자 초과입니다.",
            )
        else:
            add_check(
                "descriptionLength", "description 길이", True, f"{len(description)}자 (권장 범위)"
            )

    for prop in OG_PROPERTIES:
        value = og_values[prop]
        required = prop in {"og:title", "og:description", "og:type", "og:url"}
        add_check(
            f"og:{prop}",
            prop,
            bool(value),
            value[:80] if value else f"{prop} 메타 태그가 없습니다.",
            "critical" if required and not value else "warning",
        )

    add_check(
        "h1",
        "h1",
        h1_count == 1,
        f"h1 {h1_count}개" + ("" if h1_count == 1 else " — 정확히 1개여야 합니다."),
        "critical" if h1_count == 0 else "warning",
    )

    add_check(
        "headingHierarchy",
        "제목 계층",
        not level_skips,
        "단계 건너뛰기 없음" if not level_skips else ", ".join(
            f"h{skip['from']}→h{skip['to']}" for skip in level_skips
        ),
    )

    add_check(
        "jsonLd",
        "구조화 데이터",
        bool(json_ld_types) and not invalid_json_ld,
        (
            f"JSON-LD {len(json_ld_blocks)}개 — {', '.join(json_ld_types)}"
            if json_ld_types
            else "JSON-LD 블록이 없습니다."
        ),
        "critical" if not json_ld_types else "warning",
    )

    passed = sum(1 for check in checks if check["passed"])
    critical_failures = [
        check for check in checks if not check["passed"] and check["severity"] == "critical"
    ]

    return {
        "url": url,
        "title": page_title,
        "titleLength": len(page_title),
        "description": description,
        "descriptionLength": len(description),
        "canonical": canonical,
        "og": og_values,
        "headingLevels": heading_levels,
        "h1Count": h1_count,
        "levelSkips": level_skips,
        "jsonLdBlocks": len(json_ld_blocks),
        "jsonLdTypes": json_ld_types,
        "invalidJsonLd": len(invalid_json_ld),
        "checks": checks,
        "score": round(passed / len(checks) * 100) if checks else 0,
        "passedCount": passed,
        "totalChecks": len(checks),
        "criticalFailures": len(critical_failures),
        "status": "good" if passed == len(checks) else ("warn" if not critical_failures else "poor"),
    }


def summarize_audits(audits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-page audits into summary statistics."""
    if not audits:
        return {
            "pageCount": 0,
            "averageScore": 0,
            "goodCount": 0,
            "warnCount": 0,
            "poorCount": 0,
            "missingCounts": {},
        }

    missing_counts: dict[str, int] = {}
    for audit in audits:
        for check in audit.get("checks", []):
            if not check.get("passed"):
                missing_counts[check["label"]] = missing_counts.get(check["label"], 0) + 1

    return {
        "pageCount": len(audits),
        "averageScore": round(sum(audit["score"] for audit in audits) / len(audits)),
        "goodCount": sum(1 for audit in audits if audit["status"] == "good"),
        "warnCount": sum(1 for audit in audits if audit["status"] == "warn"),
        "poorCount": sum(1 for audit in audits if audit["status"] == "poor"),
        "missingCounts": dict(
            sorted(missing_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
