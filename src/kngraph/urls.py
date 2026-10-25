"""Load collected URL cases from a YAML configuration file.

The `urls.yaml` layout is shared across the ssg7 project family
(Bot-Access-ssg7, Browser-use-ssg7), so the loader validates the same
`cases:` schema those tools expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml


@dataclass(frozen=True)
class Case:
    case_id: int
    source: str
    query: str
    in_sitemap: bool
    url: str

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""


def load_cases(path: Path = Path("urls.yaml")) -> tuple[Case, ...]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError(f"{path}: cases must be a non-empty list")

    cases: list[Case] = []
    ids: set[int] = set()
    fields = {"case_id", "source", "query", "in_sitemap", "url"}
    for index, row in enumerate(data["cases"], start=1):
        prefix = f"{path}: cases[{index}]"
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"{prefix}: expected fields {', '.join(sorted(fields))}")
        case_id = row["case_id"]
        if type(case_id) is not int or case_id < 1 or case_id in ids:
            raise ValueError(f"{prefix}: case_id must be a unique positive integer")
        if type(row["in_sitemap"]) is not bool:
            raise ValueError(f"{prefix}: in_sitemap must be a boolean")
        for name in ("source", "query", "url"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"{prefix}: {name} must be a non-empty string")
        parts = urlsplit(row["url"])
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"{prefix}: url must be an absolute HTTP(S) URL")
        ids.add(case_id)
        cases.append(Case(**row))
    return tuple(cases)


def unique_urls(cases: tuple[Case, ...]) -> list[str]:
    """Deduplicate URLs, keeping first-seen order."""
    seen: dict[str, None] = {}
    for case in cases:
        seen.setdefault(case.url, None)
    return list(seen)


def cases_by_url(cases: tuple[Case, ...]) -> dict[str, list[Case]]:
    """Group cases by URL so a single crawl can serve every case sharing it."""
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(case.url, []).append(case)
    return grouped
