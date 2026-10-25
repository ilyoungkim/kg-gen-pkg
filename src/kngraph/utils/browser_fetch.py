"""Browser-based page fetching via real Chrome (CDP) + browser-use.

Plain HTTP clients are blocked by the WAF of shopping sites such as
``ssg.com`` (HTTP 403 on every request). Driving a real Chrome instance
instead lets the site serve a normal page: the browser solves the bot
challenge, executes the client-side scripts, and hands back the rendered
document.

The approach is the same one proven in the sibling ``Browser-use-ssg7``
project:

* launch the locally installed Chrome (not headless by default) with remote
  debugging enabled,
* reuse a persistent profile directory so cookies and WAF tokens survive
  between runs,
* attach ``browser-use`` to that Chrome over CDP,
* inject a small stealth script to reduce automation fingerprints.

The module is import-safe: if ``browser-use`` is not installed the caller
gets a clear error instead of an ``ImportError`` at import time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# HTTP status codes that mean the site refused the request outright.
BLOCK_STATUS_CODES: frozenset[int] = frozenset({401, 403, 407, 429, 503})

# Page copy that identifies a WAF / captcha interstitial.
BLOCK_MARKERS: tuple[str, ...] = (
    "captcha",
    "access denied",
    "비정상적인 접근",
    "자동화된 접근",
    "요청이 차단",
    "잠시 후 다시 시도",
    "차단되었습니다",
    "cloudflare",
)

# Locations of a locally installed Chrome, in preference order.
CHROME_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

# Minimal script that hides the most obvious automation fingerprints.
STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => false});
Object.defineProperty(navigator, 'languages', {get: () => ['ko-KR', 'ko', 'en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
"""

# The single page script that pulls everything we need out of a document.
_EXTRACT_JS = """
() => {
    const nav = performance.getEntriesByType('navigation')[0];
    const links = Array.from(document.querySelectorAll('a[href]')).map(el => ({
        text: (el.textContent || '').trim(),
        href: el.href || ''
    }));
    return JSON.stringify({
        url: location.href,
        status: nav && typeof nav.responseStatus === 'number' ? nav.responseStatus : null,
        contentType: document.contentType || '',
        title: document.title || '',
        text: document.body ? document.body.innerText : '',
        html: document.documentElement ? document.documentElement.outerHTML : '',
        links: links
    });
}
"""


class BrowserUnavailableError(RuntimeError):
    """Raised when a browser-backed crawl is requested but cannot run."""


class BrowserPageBlockedError(RuntimeError):
    """Raised when the site answers with a WAF / captcha page."""


def find_chrome(explicit: str | None = None) -> str:
    """Locate the Chrome executable to drive."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        raise BrowserUnavailableError(f"Chrome 실행 파일을 찾을 수 없습니다: {explicit}")

    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    raise BrowserUnavailableError(
        "Chrome을 찾을 수 없습니다. --chrome-path로 Chrome 실행 파일을 지정하세요."
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def cdp_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/version", timeout=1
        ):
            return True
    except (urllib.error.URLError, OSError):
        return False


def is_blocked(status: int | None, text: str) -> bool:
    """True when the response looks like a WAF / captcha refusal."""
    if status is not None and status in BLOCK_STATUS_CODES:
        return True
    lowered = (text or "").lower()
    return any(marker in lowered for marker in BLOCK_MARKERS)


class ChromeProcess:
    """A Chrome instance started with remote debugging enabled."""

    def __init__(
        self,
        executable: str,
        profile_dir: Path,
        port: int,
        headless: bool,
        size: tuple[int, int] = (1440, 900),
    ):
        self.executable = executable
        self.profile_dir = profile_dir
        self.port = port
        self.headless = headless
        self.size = size
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def cdp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if cdp_ready(self.port):
            logger.info("이미 실행 중인 Chrome CDP에 연결합니다: %s", self.cdp_url)
            return

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        width, height = self.size
        args = [
            self.executable,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile_dir.resolve()}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--lang=ko-KR",
            f"--window-size={width},{height}",
        ]
        if self.headless:
            args.append("--headless=new")
        args.append("about:blank")

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        for _ in range(80):
            if cdp_ready(self.port):
                logger.info("Chrome 실행 완료 (port=%s)", self.port)
                return
            if self._process.poll() is not None:
                break
            time.sleep(0.25)

        raise BrowserUnavailableError("Chrome 원격 디버깅 포트가 열리지 않았습니다.")

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        logger.info("Chrome 종료 (pid=%s)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()


class BrowserFetcher:
    """Fetches pages through a real Chrome driven by browser-use."""

    def __init__(
        self,
        *,
        chrome_path: str | None = None,
        profile_dir: str | Path = ".browser-profile",
        cdp_url: str | None = None,
        headless: bool = False,
        timeout: float = 30.0,
        nav_settle: float = 1.5,
        window_size: tuple[int, int] = (1440, 900),
    ):
        self.chrome_path = chrome_path
        self.profile_dir = Path(profile_dir)
        self.cdp_url = cdp_url
        self.headless = headless
        self.timeout = timeout
        self.nav_settle = nav_settle
        self.window_size = window_size

        self._session: Any = None
        self._page: Any = None
        self._chrome: ChromeProcess | None = None
        self._owns_chrome = False

    # -- lifecycle ---------------------------------------------------
    async def __aenter__(self) -> "BrowserFetcher":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        try:
            from browser_use import BrowserSession
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise BrowserUnavailableError(
                "브라우저 모드를 쓰려면 browser-use 패키지가 필요합니다. "
                "`pip install 'browser-use>=0.13.11'` 로 설치하세요."
            ) from exc

        if self.cdp_url:
            cdp_url = self.cdp_url
            logger.info("지정된 CDP URL에 연결합니다: %s", cdp_url)
        else:
            executable = find_chrome(self.chrome_path)
            port = free_port()
            self._chrome = ChromeProcess(
                executable, self.profile_dir, port, self.headless, self.window_size
            )
            self._chrome.start()
            self._owns_chrome = True
            cdp_url = self._chrome.cdp_url

        self._session = BrowserSession(cdp_url=cdp_url)
        await self._session.start()
        await self._install_stealth()

    async def _install_stealth(self) -> None:
        """Reduce automation fingerprints on every new document."""
        try:
            page = await self._require_page()
            session_id = await page.session_id
            cdp = getattr(self._session, "cdp_client", None)
            if cdp is None:
                return
            await cdp.send.Page.enable(session_id=session_id)
            await cdp.send.Page.addScriptToEvaluateOnNewDocument(
                {"source": STEALTH_JS}, session_id=session_id
            )
        except Exception as exc:  # noqa: BLE001 - stealth is best-effort
            logger.debug("스텔스 스크립트 주입 실패: %s", exc)

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.stop()
            finally:
                self._session = None
        if self._chrome is not None and self._owns_chrome:
            self._chrome.stop()
            self._owns_chrome = False

    # -- helpers -----------------------------------------------------
    async def _require_page(self) -> Any:
        if self._session is None:
            raise BrowserUnavailableError("브라우저 세션이 시작되지 않았습니다.")
        if self._page is None:
            self._page = await self._session.must_get_current_page()
        return self._page

    async def _evaluate(self, js: str, *args: Any) -> Any:
        page = await self._require_page()
        raw = await page.evaluate(js, *args)
        if raw == "" or raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    # -- public API --------------------------------------------------
    async def fetch(self, url: str) -> dict[str, Any]:
        """Load a URL in the browser and return its text, HTML and links."""
        if self._session is None:
            raise BrowserUnavailableError("브라우저 세션이 시작되지 않았습니다.")

        await asyncio.wait_for(self._session.navigate_to(url), timeout=self.timeout)
        await asyncio.sleep(self.nav_settle)

        payload = await self._evaluate(_EXTRACT_JS)
        if not isinstance(payload, dict):
            raise BrowserPageBlockedError(f"브라우저 응답을 해석하지 못했습니다: {url}")

        status = payload.get("status")
        text = str(payload.get("text") or "")
        html = str(payload.get("html") or "")
        final_url = str(payload.get("url") or url)

        if is_blocked(status, text):
            raise BrowserPageBlockedError(
                f"브라우저 요청이 차단되었습니다 (status={status}): {url}"
            )

        links = [
            {"text": str(link.get("text") or ""), "url": str(link.get("href") or "")}
            for link in payload.get("links") or []
            if link.get("href")
        ]

        return {
            "url": final_url,
            "requested_url": url,
            "status": status,
            "content_type": str(payload.get("contentType") or ""),
            "title": str(payload.get("title") or ""),
            "text": text,
            "html": html,
            "links": links,
        }

    async def fetch_many(
        self,
        urls: list[str],
        *,
        delay: float = 1.0,
        on_result: Any = None,
    ) -> list[dict[str, Any]]:
        """Fetch several URLs in order, keeping the ones that succeed."""
        results: list[dict[str, Any]] = []
        for index, url in enumerate(urls, start=1):
            if index > 1 and delay > 0:
                await asyncio.sleep(delay)
            try:
                page = await self.fetch(url)
            except BrowserPageBlockedError as exc:
                logger.warning("차단됨: %s (%s)", url, exc)
                if on_result is not None:
                    on_result(index, len(urls), None, url, str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - one page must not stop the run
                logger.warning("브라우저 수집 실패: %s (%s)", url, exc)
                if on_result is not None:
                    on_result(index, len(urls), None, url, str(exc))
                continue

            results.append(page)
            if on_result is not None:
                on_result(index, len(urls), page, url, None)
        return results
