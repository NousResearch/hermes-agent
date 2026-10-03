"""Direct, safety-preserving routes for public GitHub file URLs."""

from __future__ import annotations

import html
import json
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx

from tools import url_safety, website_policy

_DIRECT_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)
_DIRECT_TEXT_MAX_BYTES = 2_000_000
_GITHUB_HTML_HOSTS = {"github.com", "www.github.com"}
_GITHUB_RAW_HOST = "raw.githubusercontent.com"
_X_STATUS_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
_X_STATUS_PATH_RE = re.compile(r"/([^/]+)/status/(\d+)(?:/|$)")
_X_ARTICLE_URL_RE = re.compile(r"https?://(?:x\.com|twitter\.com)/i/article/\d+")


class SpecialRouteBlocked(Exception):
    """A direct route encountered a URL rejected by the website policy."""

    def __init__(self, url: str, message: str) -> None:
        super().__init__(message)
        self.url = url
        self.message = message


class SpecialRouteUnavailable(Exception):
    """A direct route failed transiently and should use ordinary provider dispatch instead."""


def _check_website_policy(url: str) -> None:
    if blocked := website_policy.check_website_access(url):
        raise SpecialRouteBlocked(url, blocked["message"])


def _direct_fetch_text(url: str, timeout: int = 20) -> tuple[str, str]:
    """Fetch bounded text using the SSRF-safe client and guard every redirect by policy."""
    headers = {
        "User-Agent": _DIRECT_FETCH_USER_AGENT,
        "Accept": "text/plain,text/markdown,text/*;q=0.9,*/*;q=0.1",
    }
    current_url = url
    with url_safety.create_ssrf_safe_client(timeout=timeout) as client:
        for _ in range(5):
            _check_website_policy(current_url)
            try:
                with client.stream(
                    "GET", current_url, headers=headers, follow_redirects=False
                ) as response:
                    if response.is_redirect:
                        redirect_url = url_safety.redirect_target_from_response(
                            response
                        )
                        if not redirect_url:
                            raise ValueError(
                                "Redirect response is missing a Location header"
                            )
                        _check_website_policy(redirect_url)
                        current_url = redirect_url
                        continue
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > _DIRECT_TEXT_MAX_BYTES:
                            raise ValueError(
                                "Direct-routed response is larger than 2MB"
                            )
                    return body.decode(
                        response.encoding or "utf-8", errors="replace"
                    ), str(response.url)
            except url_safety.SSRFConnectionBlocked as exc:
                raise SpecialRouteBlocked(
                    current_url,
                    "Blocked: URL targets a private or internal network address",
                ) from exc
    raise ValueError("Too many redirects while fetching direct-routed URL")


def _is_github_file_url(url: str) -> bool:
    """Whether *url* is a public GitHub file endpoint handled by the direct router."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if host == _GITHUB_RAW_HOST:
        return len(parts) >= 4
    return (
        host in _GITHUB_HTML_HOSTS and len(parts) >= 5 and parts[2] in {"blob", "raw"}
    )


def _github_raw_url_from_html(document: str, base_url: str) -> str:
    """Resolve GitHub's raw link so refs containing slashes are not guessed incorrectly."""
    for match in re.finditer(
        r'href=["\']([^"\']*/raw/[^"\']+)["\']', document, re.IGNORECASE
    ):
        candidate = urljoin(base_url, html.unescape(match.group(1)))
        parsed = urlparse(candidate)
        parts = [part for part in parsed.path.split("/") if part]
        if (
            (parsed.hostname or "").lower() in _GITHUB_HTML_HOSTS
            and len(parts) >= 4
            and parts[2] == "raw"
        ):
            owner, repo, _, *raw_path = parts
            return f"https://{_GITHUB_RAW_HOST}/{owner}/{repo}/{'/'.join(raw_path)}"
    return ""


def _extract_github_file(url: str) -> Optional[dict[str, Any]]:
    """Resolve GitHub's own raw link before fetching file content.

    Git refs may contain slashes, so constructing a raw URL from ``/blob/<ref>/<path>`` is
    ambiguous. Fetching the public file page first keeps GitHub authoritative for that split.
    """
    if not _is_github_file_url(url):
        return None
    # Check the caller's original URL before any direct fetch. A website-policy rule for
    # github.com must apply even when raw.githubusercontent.com is allowed.
    _check_website_policy(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    try:
        if host == _GITHUB_RAW_HOST or parts[2] == "raw":
            content, final_url = _direct_fetch_text(url)
        else:
            page, page_url = _direct_fetch_text(url)
            resolved_raw_url = _github_raw_url_from_html(page, page_url)
            if not resolved_raw_url:
                raise ValueError("GitHub file page did not expose a raw-content link")
            content, final_url = _direct_fetch_text(resolved_raw_url)
    except SpecialRouteBlocked:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise SpecialRouteUnavailable from exc
    return {
        "url": final_url,
        "title": urlparse(final_url).path.rsplit("/", 1)[-1] or final_url,
        "content": content,
        "raw_content": content,
        "error": None,
    }


def _extract_meta_content(document: str, attr: str, value: str) -> str:
    """Return a meta content value regardless of attribute order."""
    escaped_attr = re.escape(attr)
    escaped_value = re.escape(value)
    for pattern in (
        rf'<meta[^>]+{escaped_attr}=["\']{escaped_value}["\'][^>]+content=["\']([^"\']*)["\']',
        rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]+{escaped_attr}=["\']{escaped_value}["\']',
    ):
        if match := re.search(pattern, document, re.IGNORECASE):
            return html.unescape(match.group(1)).strip()
    return ""


def _extract_x_status(url: str) -> Optional[dict[str, Any]]:
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() not in _X_STATUS_HOSTS:
        return None
    if not (match := _X_STATUS_PATH_RE.fullmatch(parsed.path)):
        return None

    username, status_id = match.groups()
    try:
        document, final_url = _direct_fetch_text(url)
    except SpecialRouteBlocked:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise SpecialRouteUnavailable from exc
    title = (
        _extract_meta_content(document, "property", "og:title") or f"@{username} on X"
    )
    normalized_document = document.replace(r"\/", "/")
    article_match = _X_ARTICLE_URL_RE.search(normalized_document)
    article_url = (
        article_match.group(0).replace("twitter.com", "x.com") if article_match else ""
    )

    text = ""
    if text_match := re.search(r'"full_text"\s*:\s*("(?:\\.|[^"\\])*")', document):
        try:
            text = json.loads(text_match.group(1))
        except json.JSONDecodeError:
            text = text_match.group(1).strip('"')
    description = _extract_meta_content(document, "property", "og:description")
    lines = [f"Author: @{username}", f"Tweet ID: {status_id}"]
    if text:
        lines.append(f"Text: {text}")
    if article_url:
        lines.append(f"Linked article: {article_url}")
    if description and description != text:
        lines.append(f"Description: {description}")
    lines.append("Note: extracted via public X HTML fallback.")
    content = "# X Post\n\n" + "\n".join(f"- {line}" for line in lines)
    return {
        "url": final_url,
        "title": title,
        "content": content,
        "raw_content": content,
        "error": None,
    }


def extract_special_url(
    url: str, format: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Extract known public routes directly; return ``None`` for ordinary provider dispatch."""
    del format
    return _extract_x_status(url) or _extract_github_file(url)
