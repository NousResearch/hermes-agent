"""Bounded no-key public search worker used by the first pivot slice."""
from __future__ import annotations

import ipaddress
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse


@dataclass(frozen=True)
class SearchLimits:
    timeout_seconds: float = 8.0
    max_bytes: int = 256 * 1024
    max_results: int = 10


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            href = values.get("href", "")
            parsed = urllib.parse.urlparse(href)
            if parsed.query:
                redirected = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
                href = urllib.parse.unquote(redirected) or href
            self._current = {"url": href, "title": "", "snippet": ""}
            self._in_title = True
        elif tag in {"a", "div"} and "result__snippet" in classes and self._current is not None:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current and self._current["url"]:
                self.results.append(self._current)
                self._current = None
        if tag in {"a", "div"}:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet:
            self._current["snippet"] += data


def _safe_result_url(value: str) -> str | None:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        if ipaddress.ip_address(parsed.hostname).is_private:
            return None
    except ValueError:
        pass
    return parsed._replace(fragment="").geturl()


def _fetch_search(query: str, limits: SearchLimits = SearchLimits()) -> tuple[str, bytes]:
    endpoint = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(endpoint, headers={
        "User-Agent": "Hermes-companyintel-public-search/1.0",
        "Accept": "text/html",
    })
    with urllib.request.urlopen(request, timeout=limits.timeout_seconds) as response:
        body = response.read(limits.max_bytes + 1)
    return "text/html", body[:limits.max_bytes]


def build_query(node_type: str, value: str, mode: str = "exact") -> str:
    normalized = " ".join(str(value).split())[:300]
    if mode == "maps":
        return f'"{normalized}" map'
    if mode == "marketplace":
        return f'("{normalized}") (site:prom.ua OR site:rozetka.com.ua OR site:olx.ua)'
    if mode == "document":
        return f'"{normalized}" (filetype:pdf OR filetype:xls OR filetype:doc OR filetype:xml)'
    return f'"{normalized}"'


def execute_public_search(node_type: str, value: str, *, fetcher=None, limits: SearchLimits = SearchLimits(), mode: str = "exact") -> dict:
    query = build_query(node_type, value, mode=mode)
    fetch = fetcher or _fetch_search
    try:
        content_type, body = fetch(query, limits)
    except Exception as exc:
        return {
            "outcome": "RETRYABLE_ERROR",
            "query": query,
            "results": [],
            "error": " ".join(str(exc).split())[:300] or "public search failed",
        }
    if not content_type.startswith("text/html"):
        return {"outcome": "UNAVAILABLE", "query": query, "results": [], "error": "search response is not HTML"}
    parser = _DuckDuckGoParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    results = []
    seen = set()
    for item in parser.results:
        url = _safe_result_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "url": url,
            "title": " ".join(item.get("title", "").split())[:500],
            "snippet": " ".join(item.get("snippet", "").split())[:1000],
        })
        if len(results) >= limits.max_results:
            break
    return {
        "outcome": "COMPLETED_WITH_RESULTS" if results else "COMPLETED_ZERO_RESULTS",
        "query": query,
        "results": results,
        "error": None,
    }
