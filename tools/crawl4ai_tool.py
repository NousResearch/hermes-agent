#!/usr/bin/env python3
"""Crawl4AI deep crawling tool for Hermes.

The Crawl4AI dependency is intentionally isolated. If the current Hermes Python
can import ``crawl4ai`` we run in-process; otherwise we fall back to a configured
or conventional Crawl4AI virtualenv (``CRAWL4AI_PYTHON`` or
``~/tasklines/browser/.venv/bin/python``). This keeps the normal Hermes runtime
light while still exposing Crawl4AI as a first-class tool.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from tools.registry import registry, tool_error
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access


DEFAULT_CRAWL4AI_PYTHON = Path.home() / "tasklines" / "browser" / ".venv" / "bin" / "python"
MAX_DEPTH_LIMIT = 5
MAX_PAGES_LIMIT = 100
MAX_CONTENT_CHARS_LIMIT = 100_000
DEFAULT_MAX_CONTENT_CHARS_PER_PAGE = 20_000
DEFAULT_PAGE_TIMEOUT_MS = 60_000
SUBPROCESS_JSON_PREFIX = "__HERMES_CRAWL4AI_JSON__:"


class Crawl4AIUnavailable(RuntimeError):
    """Raised when no usable Crawl4AI runtime is available."""


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _string_list(value: Any, max_items: int = 50) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return []
    result: List[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _normalize_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        raise ValueError("url is required")
    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    return candidate


def _metadata_to_dict(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "model_dump"):
        dumped = metadata.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    return {}


def _extract_markdown(markdown: Any) -> str:
    if markdown is None:
        return ""
    if isinstance(markdown, str):
        return markdown
    for attr in ("raw_markdown", "fit_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value:
            return value
    if hasattr(markdown, "model_dump"):
        dumped = markdown.model_dump()
        if isinstance(dumped, dict):
            for key in ("raw_markdown", "fit_markdown", "markdown"):
                value = dumped.get(key)
                if isinstance(value, str) and value:
                    return value
    return str(markdown)


def _extract_links(links: Any, max_links_per_page: int) -> Dict[str, List[str]]:
    def _hrefs(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        hrefs: List[str] = []
        for item in items:
            href = ""
            if isinstance(item, dict):
                href = str(item.get("href") or item.get("url") or "").strip()
            else:
                href = str(getattr(item, "href", None) or getattr(item, "url", None) or "").strip()
            if href and href not in hrefs:
                hrefs.append(href)
            if len(hrefs) >= max_links_per_page:
                break
        return hrefs

    if not isinstance(links, dict):
        return {"internal": [], "external": []}
    return {
        "internal": _hrefs(links.get("internal")),
        "external": _hrefs(links.get("external")),
    }


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... truncated {omitted} characters ...]"


def _result_sequence(raw_results: Any) -> List[Any]:
    if raw_results is None:
        return []
    if isinstance(raw_results, list):
        return raw_results
    if isinstance(raw_results, tuple):
        return list(raw_results)
    # Crawl4AI single-page runs return one CrawlResult object, while deep crawl
    # strategies may return a list. Normalize both shapes.
    return [raw_results]


def _normalize_crawl_result(result: Any, max_content_chars_per_page: int, max_links_per_page: int) -> Dict[str, Any]:
    metadata = _metadata_to_dict(getattr(result, "metadata", None))
    page_url = (
        getattr(result, "url", None)
        or metadata.get("sourceURL")
        or metadata.get("url")
        or ""
    )
    title = metadata.get("title") or getattr(result, "title", "") or ""
    markdown = _extract_markdown(getattr(result, "markdown", None))
    content = markdown or getattr(result, "cleaned_html", None) or getattr(result, "html", None) or ""
    content = _trim_text(str(content), max_content_chars_per_page)
    success = bool(getattr(result, "success", True))
    error = None if success else (getattr(result, "error_message", None) or "Crawl failed")

    return {
        "url": str(page_url),
        "title": str(title),
        "content": content,
        "links": _extract_links(getattr(result, "links", None), max_links_per_page),
        "error": error,
    }


def _build_filter_chain(
    allowed_domains: Sequence[str],
    blocked_domains: Sequence[str],
    url_patterns: Sequence[str],
    exclude_url_patterns: Sequence[str],
) -> Optional[Any]:
    filters = []
    try:
        from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain, URLPatternFilter
    except Exception:
        return None

    if allowed_domains or blocked_domains:
        filters.append(DomainFilter(
            allowed_domains=list(allowed_domains) or None,
            blocked_domains=list(blocked_domains) or None,
        ))
    if url_patterns:
        filters.append(URLPatternFilter(list(url_patterns), use_glob=True))
    if exclude_url_patterns:
        filters.append(URLPatternFilter(list(exclude_url_patterns), use_glob=True, reverse=True))
    return FilterChain(filters) if filters else None


def _load_crawl4ai_classes():
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    return AsyncWebCrawler, CrawlerRunConfig, BFSDeepCrawlStrategy


def _current_python_has_crawl4ai() -> bool:
    return importlib.util.find_spec("crawl4ai") is not None


def _configured_crawl4ai_python() -> Optional[Path]:
    configured = os.getenv("CRAWL4AI_PYTHON", "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CRAWL4AI_PYTHON if DEFAULT_CRAWL4AI_PYTHON.exists() else None


def _probe_python_for_crawl4ai(python_path: Path, timeout_seconds: int = 5) -> bool:
    if not python_path or not python_path.exists():
        return False
    try:
        completed = subprocess.run(
            [str(python_path), "-c", "import crawl4ai"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def check_crawl4ai_available() -> bool:
    """Return True when Hermes can execute Crawl4AI locally."""
    if _current_python_has_crawl4ai():
        return True
    python_path = _configured_crawl4ai_python()
    return bool(python_path and _probe_python_for_crawl4ai(python_path))


def _runtime_description() -> str:
    if _current_python_has_crawl4ai():
        return "current_python"
    python_path = _configured_crawl4ai_python()
    return str(python_path) if python_path else "unavailable"


async def _run_crawl4ai_in_process(payload: Dict[str, Any]) -> str:
    AsyncWebCrawler, CrawlerRunConfig, BFSDeepCrawlStrategy = _load_crawl4ai_classes()

    filter_chain = _build_filter_chain(
        payload["allowed_domains"],
        payload["blocked_domains"],
        payload["url_patterns"],
        payload["exclude_url_patterns"],
    )
    strategy_kwargs: Dict[str, Any] = {
        "max_depth": payload["max_depth"],
        "include_external": payload["include_external"],
        "max_pages": payload["max_pages"],
    }
    if filter_chain is not None:
        strategy_kwargs["filter_chain"] = filter_chain

    deep_strategy = BFSDeepCrawlStrategy(**strategy_kwargs)
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_strategy,
        css_selector=payload["css_selector"] or None,
        excluded_tags=payload["excluded_tags"] or None,
        excluded_selector=payload["excluded_selector"] or None,
        only_text=payload["only_text"],
        remove_forms=payload["remove_forms"],
        remove_overlay_elements=payload["remove_overlay_elements"],
        remove_consent_popups=payload["remove_consent_popups"],
        wait_for=payload["wait_for"] or None,
        page_timeout=payload["page_timeout_ms"],
        wait_until=payload["wait_until"],
        semaphore_count=payload["semaphore_count"],
        magic=payload["magic"],
        simulate_user=payload["simulate_user"],
        scan_full_page=payload["scan_full_page"],
        check_robots_txt=payload["check_robots_txt"],
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        raw_results = await crawler.arun(url=payload["url"], config=run_config)

    normalized = [
        _normalize_crawl_result(result, payload["max_content_chars_per_page"], payload["max_links_per_page"])
        for result in _result_sequence(raw_results)
    ]

    # Crawl4AI itself is allowed to discover links dynamically; re-apply Hermes
    # policy and SSRF checks to every returned URL before exposing content.
    safe_results: List[Dict[str, Any]] = []
    blocked_count = 0
    for page in normalized:
        page_url = page.get("url") or ""
        blocked = check_website_access(page_url) if page_url else None
        if page_url and (not is_safe_url(page_url) or blocked):
            blocked_count += 1
            safe_results.append({
                "url": page_url,
                "title": page.get("title", ""),
                "content": "",
                "links": {"internal": [], "external": []},
                "error": blocked["message"] if blocked else "Blocked: URL targets a private or internal network address",
            })
            continue
        safe_results.append(page)

    total_chars = sum(len(page.get("content", "")) for page in safe_results)
    return json.dumps({
        "success": True,
        "tool": "crawl4ai_deep_crawl",
        "runtime": payload.get("runtime", "current_python"),
        "crawl": {
            "start_url": payload["url"],
            "max_depth": payload["max_depth"],
            "max_pages": payload["max_pages"],
            "include_external": payload["include_external"],
            "pages_returned": len(safe_results),
            "pages_blocked_after_crawl": blocked_count,
            "total_content_chars": total_chars,
        },
        "results": safe_results,
    }, ensure_ascii=False)


def _subprocess_script() -> str:
    """Standalone script run inside the Crawl4AI virtualenv.

    It avoids importing Hermes modules in the external venv; parent-side policy
    checks happen before launch, and returned URLs are re-checked by the parent
    after the subprocess exits.
    """
    return r'''
import asyncio
import json
import sys


def metadata_to_dict(metadata):
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "model_dump"):
        dumped = metadata.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    return {}


def extract_markdown(markdown):
    if markdown is None:
        return ""
    if isinstance(markdown, str):
        return markdown
    for attr in ("raw_markdown", "fit_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value:
            return value
    if hasattr(markdown, "model_dump"):
        dumped = markdown.model_dump()
        if isinstance(dumped, dict):
            for key in ("raw_markdown", "fit_markdown", "markdown"):
                value = dumped.get(key)
                if isinstance(value, str) and value:
                    return value
    return str(markdown)


def trim_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... truncated {omitted} characters ...]"


def hrefs(items, max_links):
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict):
            href = str(item.get("href") or item.get("url") or "").strip()
        else:
            href = str(getattr(item, "href", None) or getattr(item, "url", None) or "").strip()
        if href and href not in out:
            out.append(href)
        if len(out) >= max_links:
            break
    return out


def normalize(result, payload):
    metadata = metadata_to_dict(getattr(result, "metadata", None))
    page_url = getattr(result, "url", None) or metadata.get("sourceURL") or metadata.get("url") or ""
    title = metadata.get("title") or getattr(result, "title", "") or ""
    content = extract_markdown(getattr(result, "markdown", None)) or getattr(result, "cleaned_html", None) or getattr(result, "html", None) or ""
    links = getattr(result, "links", None)
    links_out = {"internal": [], "external": []}
    if isinstance(links, dict):
        links_out = {
            "internal": hrefs(links.get("internal"), payload["max_links_per_page"]),
            "external": hrefs(links.get("external"), payload["max_links_per_page"]),
        }
    success = bool(getattr(result, "success", True))
    return {
        "url": str(page_url),
        "title": str(title),
        "content": trim_text(str(content), payload["max_content_chars_per_page"]),
        "links": links_out,
        "error": None if success else (getattr(result, "error_message", None) or "Crawl failed"),
    }


def result_sequence(raw_results):
    if raw_results is None:
        return []
    if isinstance(raw_results, list):
        return raw_results
    if isinstance(raw_results, tuple):
        return list(raw_results)
    return [raw_results]


def build_filter_chain(payload):
    filters = []
    try:
        from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain, URLPatternFilter
    except Exception:
        return None
    if payload["allowed_domains"] or payload["blocked_domains"]:
        filters.append(DomainFilter(
            allowed_domains=payload["allowed_domains"] or None,
            blocked_domains=payload["blocked_domains"] or None,
        ))
    if payload["url_patterns"]:
        filters.append(URLPatternFilter(payload["url_patterns"], use_glob=True))
    if payload["exclude_url_patterns"]:
        filters.append(URLPatternFilter(payload["exclude_url_patterns"], use_glob=True, reverse=True))
    return FilterChain(filters) if filters else None


async def main():
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    payload = json.load(sys.stdin)
    filter_chain = build_filter_chain(payload)
    strategy_kwargs = {
        "max_depth": payload["max_depth"],
        "include_external": payload["include_external"],
        "max_pages": payload["max_pages"],
    }
    if filter_chain is not None:
        strategy_kwargs["filter_chain"] = filter_chain
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(**strategy_kwargs),
        css_selector=payload["css_selector"] or None,
        excluded_tags=payload["excluded_tags"] or None,
        excluded_selector=payload["excluded_selector"] or None,
        only_text=payload["only_text"],
        remove_forms=payload["remove_forms"],
        remove_overlay_elements=payload["remove_overlay_elements"],
        remove_consent_popups=payload["remove_consent_popups"],
        wait_for=payload["wait_for"] or None,
        page_timeout=payload["page_timeout_ms"],
        wait_until=payload["wait_until"],
        semaphore_count=payload["semaphore_count"],
        magic=payload["magic"],
        simulate_user=payload["simulate_user"],
        scan_full_page=payload["scan_full_page"],
        check_robots_txt=payload["check_robots_txt"],
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        raw = await crawler.arun(url=payload["url"], config=run_config)
    results = [normalize(item, payload) for item in result_sequence(raw)]
    print("__HERMES_CRAWL4AI_JSON__:" + json.dumps({"success": True, "results": results}, ensure_ascii=False))

asyncio.run(main())
'''


def _parse_subprocess_stdout(stdout: str) -> Dict[str, Any]:
    """Extract the JSON payload from Crawl4AI subprocess stdout.

    Crawl4AI can emit progress logs to stdout even when ``verbose=False``.
    The child script therefore prefixes the machine-readable payload with a
    sentinel and the parent ignores everything before it.
    """
    marker_index = (stdout or "").rfind(SUBPROCESS_JSON_PREFIX)
    if marker_index < 0:
        preview = (stdout or "").strip()[-1000:]
        raise RuntimeError(f"Crawl4AI subprocess did not emit a JSON payload. stdout tail: {preview}")
    payload_text = stdout[marker_index + len(SUBPROCESS_JSON_PREFIX):].strip().splitlines()[0]
    loaded = json.loads(payload_text)
    if not isinstance(loaded, dict):
        raise RuntimeError("Crawl4AI subprocess emitted a non-object JSON payload")
    return loaded


def _run_crawl4ai_subprocess(payload: Dict[str, Any]) -> str:
    python_path = _configured_crawl4ai_python()
    if not python_path or not _probe_python_for_crawl4ai(python_path):
        raise Crawl4AIUnavailable(
            "Crawl4AI is not importable in the current Hermes Python, and no usable "
            "CRAWL4AI_PYTHON or ~/tasklines/browser/.venv/bin/python runtime was found."
        )

    timeout_seconds = max(30, int(payload["page_timeout_ms"] / 1000) * max(1, payload["max_pages"]) + 30)
    completed = subprocess.run(
        [str(python_path), "-c", _subprocess_script()],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"Crawl4AI subprocess failed with exit code {completed.returncode}: {stderr[-1000:]}")

    child_payload = _parse_subprocess_stdout(completed.stdout or "")
    safe_results: List[Dict[str, Any]] = []
    blocked_count = 0
    for page in child_payload.get("results", []):
        page_url = page.get("url") or ""
        blocked = check_website_access(page_url) if page_url else None
        if page_url and (not is_safe_url(page_url) or blocked):
            blocked_count += 1
            safe_results.append({
                "url": page_url,
                "title": page.get("title", ""),
                "content": "",
                "links": {"internal": [], "external": []},
                "error": blocked["message"] if blocked else "Blocked: URL targets a private or internal network address",
            })
            continue
        safe_results.append(page)

    total_chars = sum(len(page.get("content", "")) for page in safe_results)
    return json.dumps({
        "success": True,
        "tool": "crawl4ai_deep_crawl",
        "runtime": str(python_path),
        "crawl": {
            "start_url": payload["url"],
            "max_depth": payload["max_depth"],
            "max_pages": payload["max_pages"],
            "include_external": payload["include_external"],
            "pages_returned": len(safe_results),
            "pages_blocked_after_crawl": blocked_count,
            "total_content_chars": total_chars,
        },
        "results": safe_results,
    }, ensure_ascii=False)


def _build_payload(
    url: str,
    *,
    max_depth: int = 2,
    max_pages: int = 20,
    include_external: bool = False,
    allowed_domains: Optional[Sequence[str]] = None,
    blocked_domains: Optional[Sequence[str]] = None,
    url_patterns: Optional[Sequence[str]] = None,
    exclude_url_patterns: Optional[Sequence[str]] = None,
    css_selector: Optional[str] = None,
    excluded_selector: Optional[str] = None,
    excluded_tags: Optional[Sequence[str]] = None,
    only_text: bool = False,
    remove_forms: bool = False,
    remove_overlay_elements: bool = True,
    remove_consent_popups: bool = True,
    wait_for: Optional[str] = None,
    wait_until: str = "domcontentloaded",
    page_timeout_ms: int = DEFAULT_PAGE_TIMEOUT_MS,
    semaphore_count: int = 5,
    magic: bool = False,
    simulate_user: bool = False,
    scan_full_page: bool = False,
    check_robots_txt: bool = False,
    max_content_chars_per_page: int = DEFAULT_MAX_CONTENT_CHARS_PER_PAGE,
    max_links_per_page: int = 20,
) -> Dict[str, Any]:
    normalized_url = _normalize_url(url)
    return {
        "url": normalized_url,
        "max_depth": _clamp_int(max_depth, 2, 1, MAX_DEPTH_LIMIT),
        "max_pages": _clamp_int(max_pages, 20, 1, MAX_PAGES_LIMIT),
        "include_external": _as_bool(include_external),
        "allowed_domains": _string_list(allowed_domains),
        "blocked_domains": _string_list(blocked_domains),
        "url_patterns": _string_list(url_patterns),
        "exclude_url_patterns": _string_list(exclude_url_patterns),
        "css_selector": (css_selector or "").strip(),
        "excluded_selector": (excluded_selector or "").strip(),
        "excluded_tags": _string_list(excluded_tags),
        "only_text": _as_bool(only_text),
        "remove_forms": _as_bool(remove_forms),
        "remove_overlay_elements": _as_bool(remove_overlay_elements, True),
        "remove_consent_popups": _as_bool(remove_consent_popups, True),
        "wait_for": (wait_for or "").strip(),
        "wait_until": (wait_until or "domcontentloaded").strip() or "domcontentloaded",
        "page_timeout_ms": _clamp_int(page_timeout_ms, DEFAULT_PAGE_TIMEOUT_MS, 5_000, 300_000),
        "semaphore_count": _clamp_int(semaphore_count, 5, 1, 20),
        "magic": _as_bool(magic),
        "simulate_user": _as_bool(simulate_user),
        "scan_full_page": _as_bool(scan_full_page),
        "check_robots_txt": _as_bool(check_robots_txt),
        "max_content_chars_per_page": _clamp_int(
            max_content_chars_per_page,
            DEFAULT_MAX_CONTENT_CHARS_PER_PAGE,
            1_000,
            MAX_CONTENT_CHARS_LIMIT,
        ),
        "max_links_per_page": _clamp_int(max_links_per_page, 20, 0, 100),
    }


async def crawl4ai_deep_crawl_tool(
    url: str,
    max_depth: int = 2,
    max_pages: int = 20,
    include_external: bool = False,
    allowed_domains: Optional[Sequence[str]] = None,
    blocked_domains: Optional[Sequence[str]] = None,
    url_patterns: Optional[Sequence[str]] = None,
    exclude_url_patterns: Optional[Sequence[str]] = None,
    css_selector: Optional[str] = None,
    excluded_selector: Optional[str] = None,
    excluded_tags: Optional[Sequence[str]] = None,
    only_text: bool = False,
    remove_forms: bool = False,
    remove_overlay_elements: bool = True,
    remove_consent_popups: bool = True,
    wait_for: Optional[str] = None,
    wait_until: str = "domcontentloaded",
    page_timeout_ms: int = DEFAULT_PAGE_TIMEOUT_MS,
    semaphore_count: int = 5,
    magic: bool = False,
    simulate_user: bool = False,
    scan_full_page: bool = False,
    check_robots_txt: bool = False,
    max_content_chars_per_page: int = DEFAULT_MAX_CONTENT_CHARS_PER_PAGE,
    max_links_per_page: int = 20,
) -> str:
    """Deep crawl a website locally with Crawl4AI and return compact Markdown pages."""
    try:
        payload = _build_payload(
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            url_patterns=url_patterns,
            exclude_url_patterns=exclude_url_patterns,
            css_selector=css_selector,
            excluded_selector=excluded_selector,
            excluded_tags=excluded_tags,
            only_text=only_text,
            remove_forms=remove_forms,
            remove_overlay_elements=remove_overlay_elements,
            remove_consent_popups=remove_consent_popups,
            wait_for=wait_for,
            wait_until=wait_until,
            page_timeout_ms=page_timeout_ms,
            semaphore_count=semaphore_count,
            magic=magic,
            simulate_user=simulate_user,
            scan_full_page=scan_full_page,
            check_robots_txt=check_robots_txt,
            max_content_chars_per_page=max_content_chars_per_page,
            max_links_per_page=max_links_per_page,
        )
    except ValueError as exc:
        return tool_error(str(exc), success=False)

    if not is_safe_url(payload["url"]):
        return tool_error("Blocked: URL targets a private or internal network address", success=False)

    blocked = check_website_access(payload["url"])
    if blocked:
        return json.dumps({
            "success": False,
            "error": blocked["message"],
            "blocked_by_policy": {
                "host": blocked["host"],
                "rule": blocked["rule"],
                "source": blocked["source"],
            },
        }, ensure_ascii=False)

    try:
        payload["runtime"] = _runtime_description()
        if _current_python_has_crawl4ai():
            return await _run_crawl4ai_in_process(payload)
        return await asyncio.to_thread(_run_crawl4ai_subprocess, payload)
    except Crawl4AIUnavailable as exc:
        return tool_error(str(exc), success=False)
    except subprocess.TimeoutExpired:
        return tool_error("Crawl4AI crawl timed out", success=False)
    except Exception as exc:
        return tool_error(f"Error running Crawl4AI deep crawl: {type(exc).__name__}: {exc}", success=False)


CRAWL4AI_DEEP_CRAWL_SCHEMA = {
    "name": "crawl4ai_deep_crawl",
    "description": (
        "Deep crawl a website locally with Crawl4AI. Use this for recursive or structured site crawling, "
        "multi-page documentation ingestion, link-following, and collecting Markdown from many related pages. "
        "For a single URL, prefer web_extract; for interactive login/click flows, use browser tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Start URL to crawl. https:// is added if omitted."},
            "max_depth": {"type": "integer", "description": "Maximum link depth to follow. Defaults to 2; capped at 5.", "minimum": 1, "maximum": MAX_DEPTH_LIMIT, "default": 2},
            "max_pages": {"type": "integer", "description": "Maximum pages to return. Defaults to 20; capped at 100.", "minimum": 1, "maximum": MAX_PAGES_LIMIT, "default": 20},
            "include_external": {"type": "boolean", "description": "Whether to follow external-domain links. Default false.", "default": False},
            "allowed_domains": {"type": "array", "items": {"type": "string"}, "description": "Optional domains to allow during crawl filtering."},
            "blocked_domains": {"type": "array", "items": {"type": "string"}, "description": "Optional domains to block during crawl filtering."},
            "url_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional glob-style URL patterns to include, e.g. */docs/*."},
            "exclude_url_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional glob-style URL patterns to exclude, e.g. */login* or */tag/*."},
            "css_selector": {"type": "string", "description": "Optional CSS selector limiting extracted page content."},
            "excluded_selector": {"type": "string", "description": "Optional CSS selector to remove from extracted pages."},
            "excluded_tags": {"type": "array", "items": {"type": "string"}, "description": "HTML tags to exclude, such as script, style, nav, footer."},
            "only_text": {"type": "boolean", "description": "Extract only text content. Default false.", "default": False},
            "remove_forms": {"type": "boolean", "description": "Remove forms from extracted content. Default false.", "default": False},
            "remove_overlay_elements": {"type": "boolean", "description": "Remove overlays/popups. Default true.", "default": True},
            "remove_consent_popups": {"type": "boolean", "description": "Remove cookie consent popups. Default true.", "default": True},
            "wait_for": {"type": "string", "description": "Optional Crawl4AI wait_for condition/selector before extraction."},
            "wait_until": {"type": "string", "description": "Browser load state to wait for. Defaults to domcontentloaded.", "default": "domcontentloaded"},
            "page_timeout_ms": {"type": "integer", "description": "Per-page browser timeout in milliseconds. Defaults to 60000.", "minimum": 5000, "maximum": 300000, "default": DEFAULT_PAGE_TIMEOUT_MS},
            "semaphore_count": {"type": "integer", "description": "Crawl4AI concurrency. Defaults to 5; capped at 20.", "minimum": 1, "maximum": 20, "default": 5},
            "magic": {"type": "boolean", "description": "Enable Crawl4AI magic mode for harder pages. Default false.", "default": False},
            "simulate_user": {"type": "boolean", "description": "Simulate user-like browser behavior. Default false.", "default": False},
            "scan_full_page": {"type": "boolean", "description": "Scroll/scan full pages before extraction. Default false.", "default": False},
            "check_robots_txt": {"type": "boolean", "description": "Ask Crawl4AI to respect robots.txt. Default false.", "default": False},
            "max_content_chars_per_page": {"type": "integer", "description": "Maximum content characters returned per page. Defaults to 20000; capped at 100000.", "minimum": 1000, "maximum": MAX_CONTENT_CHARS_LIMIT, "default": DEFAULT_MAX_CONTENT_CHARS_PER_PAGE},
            "max_links_per_page": {"type": "integer", "description": "Maximum internal/external discovered links returned per page. Defaults to 20.", "minimum": 0, "maximum": 100, "default": 20},
        },
        "required": ["url"],
    },
}


registry.register(
    name="crawl4ai_deep_crawl",
    toolset="web",
    schema=CRAWL4AI_DEEP_CRAWL_SCHEMA,
    handler=lambda args, **kw: crawl4ai_deep_crawl_tool(
        args.get("url", ""),
        max_depth=args.get("max_depth", 2),
        max_pages=args.get("max_pages", 20),
        include_external=args.get("include_external", False),
        allowed_domains=args.get("allowed_domains"),
        blocked_domains=args.get("blocked_domains"),
        url_patterns=args.get("url_patterns"),
        exclude_url_patterns=args.get("exclude_url_patterns"),
        css_selector=args.get("css_selector"),
        excluded_selector=args.get("excluded_selector"),
        excluded_tags=args.get("excluded_tags"),
        only_text=args.get("only_text", False),
        remove_forms=args.get("remove_forms", False),
        remove_overlay_elements=args.get("remove_overlay_elements", True),
        remove_consent_popups=args.get("remove_consent_popups", True),
        wait_for=args.get("wait_for"),
        wait_until=args.get("wait_until", "domcontentloaded"),
        page_timeout_ms=args.get("page_timeout_ms", DEFAULT_PAGE_TIMEOUT_MS),
        semaphore_count=args.get("semaphore_count", 5),
        magic=args.get("magic", False),
        simulate_user=args.get("simulate_user", False),
        scan_full_page=args.get("scan_full_page", False),
        check_robots_txt=args.get("check_robots_txt", False),
        max_content_chars_per_page=args.get("max_content_chars_per_page", DEFAULT_MAX_CONTENT_CHARS_PER_PAGE),
        max_links_per_page=args.get("max_links_per_page", 20),
    ),
    check_fn=check_crawl4ai_available,
    requires_env=[],
    is_async=True,
    emoji="🕸️",
    max_result_size_chars=200_000,
)
