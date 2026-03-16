"""Stealth web scraper with anti-bot evasion.

Uses Playwright when available; falls back to httpx + BeautifulSoup.
Implements human-like behavior: random delays, User-Agent rotation,
and page interaction simulation.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User-Agent pool (realistic, recent browsers)
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
]


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _human_delay(min_s: float = 1.5, max_s: float = 5.0):
    """Sleep for a random duration to mimic human behavior."""
    time.sleep(random.uniform(min_s, max_s))


# ---------------------------------------------------------------------------
# Scrape result
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    """Raw HTML + metadata returned by the scraper."""
    url: str = ""
    html: str = ""
    status_code: int = 0
    success: bool = False
    error: str = ""
    method: str = ""  # "playwright" or "httpx"


# ---------------------------------------------------------------------------
# Playwright scraper
# ---------------------------------------------------------------------------

def _check_playwright_available() -> bool:
    """Return True if Playwright and chromium are installed."""
    try:
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        return False


def _scrape_with_playwright(url: str) -> ScrapeResult:
    """Scrape a URL using Playwright in stealth mode."""
    from playwright.sync_api import sync_playwright

    result = ScrapeResult(url=url, method="playwright")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = browser.new_context(
                user_agent=_random_ua(),
                viewport={"width": random.randint(1280, 1920), "height": random.randint(800, 1080)},
                locale="tr-TR",
                timezone_id="Europe/Istanbul",
                java_script_enabled=True,
            )

            # Anti-detection: override navigator.webdriver
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'languages', { get: () => ['tr-TR', 'tr', 'en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                window.chrome = { runtime: {} };
            """)

            page = context.new_page()

            # Random delay before navigation
            _human_delay(0.5, 1.5)

            # Amazon has too many trackers for networkidle, use domcontentloaded + specific wait
            response = page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # If it's Amazon, explicitly wait for price elements to render
            if "amazon." in url:
                try:
                    # Wait for either standard price class or new core price divs
                    page.wait_for_selector(".a-price, #corePrice_feature_div, #corePriceDisplay_desktop_feature_div", timeout=5000)
                except Exception:
                    # Don't fail if price is genuinely missing (e.g. out of stock)
                    logger.debug("Amazon price selectors not found, continuing anyway.")

            # Simulate human scrolling
            for _ in range(random.randint(1, 2)):
                page.evaluate(f"window.scrollBy(0, {random.randint(200, 500)})")
                _human_delay(0.3, 1.0)

            result.html = page.content()
            result.status_code = response.status if response else 0
            result.success = True

            browser.close()

    except Exception as e:
        result.error = str(e)
        logger.debug("Playwright scrape failed for %s: %s", url, e)

    return result


# ---------------------------------------------------------------------------
# httpx fallback scraper
# ---------------------------------------------------------------------------

def _scrape_with_httpx(url: str) -> ScrapeResult:
    """Scrape a URL using httpx with stealth headers (no JS rendering)."""
    import httpx

    result = ScrapeResult(url=url, method="httpx")

    headers = {
        "User-Agent": _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    _human_delay(1.0, 3.0)

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers=headers,
        ) as client:
            resp = client.get(url)
            result.html = resp.text
            result.status_code = resp.status_code
            result.success = resp.status_code == 200

    except Exception as e:
        result.error = str(e)
        logger.debug("httpx scrape failed for %s: %s", url, e)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StealthScraper:
    """Scrapes web pages with anti-bot evasion.

    Prefers Playwright for JS-heavy sites; falls back to httpx.
    """

    def __init__(self, prefer_playwright: bool = True):
        self._use_playwright = prefer_playwright and _check_playwright_available()
        if self._use_playwright:
            logger.info("StealthScraper: Playwright mode active")
        else:
            logger.info("StealthScraper: httpx fallback mode")

    @property
    def method(self) -> str:
        return "playwright" if self._use_playwright else "httpx"

    def scrape(self, url: str) -> ScrapeResult:
        """Scrape a URL and return the raw HTML."""
        if self._use_playwright:
            result = _scrape_with_playwright(url)
            # Fall back to httpx if Playwright fails
            if not result.success:
                logger.info("Playwright failed, falling back to httpx for %s", url)
                result = _scrape_with_httpx(url)
        else:
            result = _scrape_with_httpx(url)

        return result
