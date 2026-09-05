from collections import deque
from collections.abc import Callable
from xml.etree import ElementTree
from urllib.parse import urljoin, urlparse, urldefrag

from curl_cffi import requests
from bs4 import BeautifulSoup


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


def _extract_page_data(page_url: str) -> tuple[str, list[dict[str, str]], list[str]]:
    response = _fetch_url(page_url)
    if response is None:
        return '', [], []
    if not _has_html_content_type(response):
        return '', [], []

    soup = BeautifulSoup(response.text, 'html.parser')
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

    return page_text, links, discovered_urls


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

        page_text, links, discovered_urls = _extract_page_data(current_url)
        if page_text:
            page_documents.append({
                'url': current_url,
                'text': page_text,
                'crawl_source': 'same_domain',
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


def get_web_content(
    url: str,
    progress_callback: Callable[[str], None] | None = None,
    verbose_callback: Callable[[str, dict | None], None] | None = None,
    page_limit: int | None = None,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    if progress_callback is not None:
        progress_callback('sitemap.xml을 확인하는 중')
    sitemap_max = page_limit if isinstance(page_limit, int) and page_limit > 0 else 30
    sitemap_urls = _collect_sitemap_urls(url, max_urls=sitemap_max)

    if sitemap_urls:
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
            page_text, links, _ = _extract_page_data(sitemap_page_url)
            if page_text:
                page_documents.append({
                    'url': sitemap_page_url,
                    'text': page_text,
                    'crawl_source': 'sitemap',
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