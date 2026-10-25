import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree
from urllib.parse import urljoin, urlparse, urldefrag

from curl_cffi import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


REQUEST_TIMEOUT = 15
BROWSER_IMPERSONATION = 'chrome110'
GOOGLEBOT_USER_AGENT = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
NON_DOCUMENT_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.webp',
    '.svg',
    '.ico',
    '.bmp',
    '.tiff',
    '.avif',
    '.mp4',
    '.webm',
    '.mov',
    '.avi',
    '.mp3',
    '.wav',
    '.pdf',
    '.zip',
    '.gz',
    '.rar',
    '.7z',
    '.css',
    '.js',
    '.json',
    '.xml',
}
DEFAULT_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'User-Agent': GOOGLEBOT_USER_AGENT,
}


def _fetch_url(url: str) -> requests.Response | None:
    try:
        with requests.Session() as session:
            response = session.get(
                url,
                impersonate=BROWSER_IMPERSONATION,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response
    except requests.RequestsError:
        return None


def _normalize_url(base_url: str, candidate_url: str) -> str | None:
    if not candidate_url:
        return None

    absolute_url = urljoin(base_url, candidate_url)
    cleaned_url, _ = urldefrag(absolute_url)
    parsed = urlparse(cleaned_url)
    if parsed.scheme not in {'http', 'https'}:
        return None
    return cleaned_url


def _is_same_domain(url: str, target_netloc: str) -> bool:
    return urlparse(url).netloc == target_netloc


def _looks_like_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if not path or path.endswith('/'):
        return True
    return not any(path.endswith(extension) for extension in NON_DOCUMENT_EXTENSIONS)


def _has_html_content_type(response: requests.Response) -> bool:
    content_type = response.headers.get('Content-Type', '').lower()
    if not content_type:
        return True
    return 'text/html' in content_type or 'application/xhtml+xml' in content_type


def _extract_page_data(page_url: str) -> tuple[str, list[dict[str, str]], list[str], str]:
    """Return (text, links, discovered_urls, raw_html)."""
    response = _fetch_url(page_url)
    if response is None:
        return '', [], [], ''
    if not _has_html_content_type(response):
        return '', [], [], ''

    raw_html = response.text
    soup = BeautifulSoup(raw_html, 'html.parser')
    page_text = soup.get_text(separator=' ', strip=True)

    links: list[dict[str, str]] = []
    discovered_urls: list[str] = []
    for anchor in soup.find_all('a', href=True):
        href = anchor.get('href')
        if not isinstance(href, str):
            continue

        normalized_url = _normalize_url(page_url, href)
        if not normalized_url:
            continue
        if not _looks_like_document_url(normalized_url):
            continue
        links.append({
            'text': anchor.text.strip(),
            'url': normalized_url,
        })
        discovered_urls.append(normalized_url)

    return page_text, links, discovered_urls, raw_html


def _collect_sitemap_urls(target_url: str, max_urls: int = 30) -> list[str]:
    parsed_target = urlparse(target_url)
    if not parsed_target.scheme or not parsed_target.netloc:
        return []

    root_url = f'{parsed_target.scheme}://{parsed_target.netloc}'
    sitemap_url = urljoin(root_url, '/sitemap.xml')
    visited_sitemaps: set[str] = set()
    queued_sitemaps: deque[str] = deque([sitemap_url])
    collected_urls: list[str] = []

    while queued_sitemaps and len(collected_urls) < max_urls:
        current_sitemap = queued_sitemaps.popleft()
        if current_sitemap in visited_sitemaps:
            continue
        visited_sitemaps.add(current_sitemap)

        response = _fetch_url(current_sitemap)
        if response is None:
            continue

        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError:
            continue

        root_tag = root.tag.split('}')[-1]
        loc_values = [element.text.strip() for element in root.iter() if element.tag.split('}')[-1] == 'loc' and element.text]

        if root_tag == 'sitemapindex':
            for sitemap_loc in loc_values:
                normalized_url = _normalize_url(root_url, sitemap_loc)
                if normalized_url and normalized_url not in visited_sitemaps:
                    queued_sitemaps.append(normalized_url)
            continue

        for page_url in loc_values:
            normalized_url = _normalize_url(root_url, page_url)
            if not normalized_url:
                continue
            if not _is_same_domain(normalized_url, parsed_target.netloc):
                continue
            if not _looks_like_document_url(normalized_url):
                continue
            if normalized_url not in collected_urls:
                collected_urls.append(normalized_url)
            if len(collected_urls) >= max_urls:
                break

    return collected_urls


def _crawl_same_domain(
    start_url: str,
    max_pages: int = 50,
    progress_callback: Callable[[str], None] | None = None,
    verbose_callback: Callable[[str, dict | None], None] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parsed_start = urlparse(start_url)
    target_netloc = parsed_start.netloc
    queue: deque[str] = deque([start_url])
    visited: set[str] = set()
    page_documents: list[dict[str, str]] = []
    collected_links: list[dict[str, str]] = []
    seen_link_pairs: set[tuple[str, str]] = set()

    while queue and len(visited) < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        if progress_callback is not None:
            progress_callback(f'동일 도메인 크롤링 중 ({len(visited) + 1}/{max_pages})')
        if verbose_callback is not None:
            verbose_callback('크롤링 대상 페이지', {'url': current_url})
        visited.add(current_url)

        page_text, links, discovered_urls, page_html = _extract_page_data(current_url)
        if page_text:
            page_documents.append({
                'url': current_url,
                'text': page_text,
                'crawl_source': 'same_domain',
                'html': page_html,
            })
            if verbose_callback is not None:
                verbose_callback(
                    '페이지 수집 완료',
                    {
                        'url': current_url,
                        'text_length': len(page_text),
                        'discovered_url_count': len(discovered_urls),
                    },
                )
        elif verbose_callback is not None:
            verbose_callback('페이지 수집 실패 또는 빈 텍스트', {'url': current_url})

        for link in links:
            link_key = (link['text'], link['url'])
            if link_key not in seen_link_pairs:
                collected_links.append(link)
                seen_link_pairs.add(link_key)

        for discovered_url in discovered_urls:
            if not _is_same_domain(discovered_url, target_netloc):
                continue
            if discovered_url not in visited and discovered_url not in queue:
                queue.append(discovered_url)

    return page_documents, collected_links


class _BrowserPageFetcher:
    """Synchronous facade over :class:`BrowserFetcher` for the crawl loop.

    The browser runs in its own thread so the existing synchronous crawler
    code can stay synchronous.
    """

    def __init__(self, **options: object):
        self.options = options
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._fetcher: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stopped = False

    def __enter__(self) -> "_BrowserPageFetcher":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        """Own a permanently running event loop.

        browser-use keeps background tasks alive to hold the CDP client
        open, so the loop must never go idle between requests.
        """
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.create_task(self._start())
            self._loop.run_forever()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._error = exc
        finally:
            self._ready.set()

    async def _start(self) -> None:
        try:
            from kngraph.utils.browser_fetch import BrowserFetcher

            self._fetcher = BrowserFetcher(**self.options)  # type: ignore[arg-type]
            await self._fetcher.start()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            self._error = exc
        finally:
            self._ready.set()

    def _submit(self, coro: Any, timeout: float) -> Any:
        if self._loop is None or self._fetcher is None:
            coro.close()
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._loop is None:
            return
        if self._fetcher is not None and not self._stopped:
            try:
                self._submit(self._fetcher.close(), timeout=30.0)
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("브라우저 종료 오류: %s", exc)
            self._stopped = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        try:
            self._loop.close()
        except Exception:  # noqa: BLE001 - best effort
            pass
        self._loop = None

    def fetch(self, url: str) -> dict[str, Any] | None:
        """Fetch one page; returns None when the page is blocked or fails."""
        if self._loop is None or self._fetcher is None:
            return None
        timeout = float(self.options.get("timeout", 30.0)) + 20.0
        try:
            return self._submit(self._fetcher.fetch(url), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - a failed page must not stop the crawl
            logger.debug("브라우저 수집 실패 %s: %s", url, exc)
            return None


def _extract_links_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Parse anchor tags out of a browser-rendered document."""
    if not html:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    links: list[dict[str, str]] = []
    for anchor in soup.find_all('a', href=True):
        href = anchor.get('href')
        if not isinstance(href, str):
            continue
        normalized_url = _normalize_url(base_url, href)
        if not normalized_url:
            continue
        if not _looks_like_document_url(normalized_url):
            continue
        links.append({'text': anchor.text.strip(), 'url': normalized_url})
    return links


def _fetch_page_via_browser(
    page_url: str,
    fetcher: _BrowserPageFetcher,
) -> tuple[str, list[dict[str, str]], str] | None:
    """Fetch one page in the browser and return (text, links, html)."""
    page = fetcher.fetch(page_url)
    if not page:
        return None
    text = str(page.get('text') or '').strip()
    html = str(page.get('html') or '')
    if not text:
        return None
    links = _extract_links_from_html(html, page_url)
    return text, links, html


def _crawl_with_browser(
    url: str,
    *,
    progress_callback: Callable[[str], None] | None = None,
    verbose_callback: Callable[[str, dict | None], None] | None = None,
    page_limit: int | None = None,
    browser_options: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Crawl with a real browser instead of plain HTTP requests.

    The browser solves the site's bot challenge and returns the rendered
    document, so pages that answer 403 to a plain HTTP client still load.
    """
    crawl_max = page_limit if isinstance(page_limit, int) and page_limit > 0 else 10

    with _BrowserPageFetcher(**(browser_options or {})) as fetcher:
        if progress_callback is not None:
            progress_callback('브라우저를 실행하는 중 (Chrome + CDP)')
        if verbose_callback is not None:
            verbose_callback('브라우저 수집 시작', {'url': url, 'max_pages': crawl_max})

        # The requested page always comes first, then same-domain links.
        queue: deque[str] = deque([url])
        visited: set[str] = set()
        page_documents: list[dict[str, str]] = []
        collected_links: list[dict[str, str]] = []
        seen_link_pairs: set[tuple[str, str]] = set()
        target_netloc = urlparse(url).netloc

        while queue and len(visited) < crawl_max:
            current_url = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            if progress_callback is not None:
                progress_callback(
                    f'브라우저 페이지 수집 중 ({len(visited)}/{crawl_max})'
                )
            if verbose_callback is not None:
                verbose_callback('브라우저 수집 대상', {'url': current_url})

            result = _fetch_page_via_browser(current_url, fetcher)
            if result is None:
                if verbose_callback is not None:
                    verbose_callback('브라우저 수집 실패 또는 차단', {'url': current_url})
                continue

            page_text, links, page_html = result
            page_documents.append(
                {
                    'url': current_url,
                    'text': page_text,
                    'crawl_source': 'browser',
                    'html': page_html,
                }
            )
            if verbose_callback is not None:
                verbose_callback(
                    '브라우저 페이지 수집 완료',
                    {
                        'url': current_url,
                        'text_length': len(page_text),
                        'link_count': len(links),
                    },
                )

            for link in links:
                link_key = (link['text'], link['url'])
                if link_key not in seen_link_pairs:
                    seen_link_pairs.add(link_key)
                    collected_links.append(link)

            for link in links:
                discovered_url = link['url']
                if not _is_same_domain(discovered_url, target_netloc):
                    continue
                if discovered_url not in visited and discovered_url not in queue:
                    queue.append(discovered_url)

    aggregated_text = '\n\n'.join(
        f"[SOURCE] {page_document['url']}\n{page_document['text']}"
        for page_document in page_documents
    )
    return aggregated_text, collected_links, page_documents


def get_web_content(
    url: str,
    progress_callback: Callable[[str], None] | None = None,
    verbose_callback: Callable[[str, dict | None], None] | None = None,
    page_limit: int | None = None,
    browser_options: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    if browser_options:
        return _crawl_with_browser(
            url,
            progress_callback=progress_callback,
            verbose_callback=verbose_callback,
            page_limit=page_limit,
            browser_options=browser_options,
        )

    if progress_callback is not None:
        progress_callback('sitemap.xml을 확인하는 중')
    sitemap_max = page_limit if isinstance(page_limit, int) and page_limit > 0 else 30
    sitemap_urls = _collect_sitemap_urls(url, max_urls=sitemap_max)

    if sitemap_urls:
        # The requested target URL must always be crawled; a sitemap can list
        # unrelated pages and none of the target itself.
        if url not in sitemap_urls:
            sitemap_urls = [url, *sitemap_urls]

        if progress_callback is not None:
            progress_callback(f'sitemap 기반 수집을 시작하는 중: {len(sitemap_urls)}개 페이지')
        page_documents: list[dict[str, str]] = []
        collected_links: list[dict[str, str]] = []
        seen_link_pairs: set[tuple[str, str]] = set()

        for index, sitemap_page_url in enumerate(sitemap_urls, start=1):
            if progress_callback is not None:
                progress_callback(f'sitemap 페이지 수집 중 ({index}/{len(sitemap_urls)})')
            if verbose_callback is not None:
                verbose_callback('sitemap 대상 페이지', {'url': sitemap_page_url})
            page_text, links, _, page_html = _extract_page_data(sitemap_page_url)
            if page_text:
                page_documents.append({
                    'url': sitemap_page_url,
                    'text': page_text,
                    'crawl_source': 'sitemap',
                    'html': page_html,
                })
                if verbose_callback is not None:
                    verbose_callback(
                        'sitemap 페이지 수집 완료',
                        {
                            'url': sitemap_page_url,
                            'text_length': len(page_text),
                            'link_count': len(links),
                        },
                    )
            elif verbose_callback is not None:
                verbose_callback('sitemap 페이지 수집 실패 또는 빈 텍스트', {'url': sitemap_page_url})

            for link in links:
                link_key = (link['text'], link['url'])
                if link_key not in seen_link_pairs:
                    collected_links.append(link)
                    seen_link_pairs.add(link_key)

        aggregated_text = '\n\n'.join(
            f"[SOURCE] {page_document['url']}\n{page_document['text']}"
            for page_document in page_documents
        )
        return aggregated_text, collected_links, page_documents

    if progress_callback is not None:
        progress_callback('sitemap.xml이 없어 동일 도메인 크롤링으로 전환하는 중')
    crawl_max = page_limit if isinstance(page_limit, int) and page_limit > 0 else 10
    page_documents, collected_links = _crawl_same_domain(
        url,
        max_pages=crawl_max,
        progress_callback=progress_callback,
        verbose_callback=verbose_callback,
    )
    aggregated_text = '\n\n'.join(
        f"[SOURCE] {page_document['url']}\n{page_document['text']}"
        for page_document in page_documents
    )
    return aggregated_text, collected_links, page_documents