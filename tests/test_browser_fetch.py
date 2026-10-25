"""Offline unit tests for the browser-based fetch helpers.

No browser is launched here: only the pure helpers are exercised.
"""

from pathlib import Path

import pytest

from src.kngraph.utils.browser_fetch import (
    BLOCK_MARKERS,
    BLOCK_STATUS_CODES,
    BrowserUnavailableError,
    cdp_ready,
    find_chrome,
    free_port,
    is_blocked,
)
from src.kngraph.utils.scraping import _extract_links_from_html


def test_find_chrome_rejects_missing_explicit_path():
    with pytest.raises(BrowserUnavailableError):
        find_chrome("/definitely/not/here/chrome")


def test_free_port_returns_a_usable_port():
    port = free_port()

    assert 1024 < port < 65536


def test_cdp_ready_is_false_for_a_closed_port():
    assert cdp_ready(free_port()) is False


def test_is_blocked_detects_status_codes():
    for status in BLOCK_STATUS_CODES:
        assert is_blocked(status, "") is True


def test_is_blocked_detects_waf_copy():
    assert is_blocked(200, "Please solve the CAPTCHA to continue") is True
    assert is_blocked(200, "비정상적인 접근이 감지되었습니다") is True


def test_is_blocked_allows_normal_pages():
    assert is_blocked(200, "쓱7클럽 멤버십 안내 페이지입니다") is False
    assert is_blocked(None, "") is False


def test_block_markers_are_lowercase():
    for marker in BLOCK_MARKERS:
        assert marker == marker.lower()


def test_extract_links_from_html_keeps_document_links():
    html = """
    <html><body>
      <a href="/page/pc/membershiponly.ssg">멤버십</a>
      <a href="https://www.ssg.com/service/emart/dvstore.ssg">이마트</a>
      <a href="/image/logo.png">로고</a>
      <a href="javascript:void(0)">script</a>
    </body></html>
    """

    links = _extract_links_from_html(html, "https://www.ssg.com/page/pc/membershiponly.ssg")

    urls = [link["url"] for link in links]
    assert "https://www.ssg.com/page/pc/membershiponly.ssg" in urls
    assert "https://www.ssg.com/service/emart/dvstore.ssg" in urls
    # Image and javascript links are not documents.
    assert not any(url.endswith(".png") for url in urls)
    assert not any(url.startswith("javascript") for url in urls)


def test_extract_links_from_html_handles_empty_input():
    assert _extract_links_from_html("", "https://example.com") == []
