"""Native katana web extraction + crawling tool.

Service-gated: only registered when the `katana` binary (ProjectDiscovery)
is available on disk. Provides a self-hosted, API-key-free extract/crawl
capability backed by a local Go binary — no external provider required.

Tools:
  - katana_extract: fetch a single URL, return clean text (web_extract analog)
  - katana_crawl:   spider a site (depth-limited), return discovered URLs

All handlers return JSON strings, per Hermes tool contract.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import shutil
import subprocess

from tools.registry import registry

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------
_CANDIDATE_PATHS = [
    shutil.which("katana"),
    os.path.expanduser("~/go/bin/katana"),
    "/usr/local/bin/katana",
    "/usr/bin/katana",
]


def _katana_bin() -> str | None:
    for p in _CANDIDATE_PATHS:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def check_requirements() -> bool:
    """Tool is only exposed when the katana binary is present."""
    return _katana_bin() is not None


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(raw: str) -> str:
    """Best-effort HTML -> readable text. Prefers BeautifulSoup if installed,
    falls back to a stdlib regex stripper (no new hard dependency)."""
    if not raw:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        cleaned = _SCRIPT_STYLE_RE.sub(" ", raw)
        cleaned = re.sub(r"</(p|div|li|tr|h[1-6]|br|section|article)>",
                         "\n", cleaned, flags=re.IGNORECASE)
        cleaned = _TAG_RE.sub("", cleaned)
        text = _html.unescape(cleaned)

    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return _MULTI_NL_RE.sub("\n\n", text).strip()


def _run_katana(args: list[str], timeout: int) -> tuple[int, str, str]:
    binp = _katana_bin()
    if not binp:
        return 1, "", "katana binary not found"
    try:
        proc = subprocess.run(
            [binp, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"katana timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"katana execution error: {e}"


# ---------------------------------------------------------------------------
# katana_extract — single page -> clean text
# ---------------------------------------------------------------------------
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _extract_title(body: str) -> "str | None":
    m = re.search(r"<title[^>]*>(.*?)</title>", body,
                  re.IGNORECASE | re.DOTALL)
    return _html.unescape(m.group(1)).strip() if m else None


def _direct_fetch(url: str, timeout: int = 15):
    """Fast single-page GET via httpx (preferred) or requests.
    Returns (status_code, body_text) or (None, '') on failure."""
    try:
        import httpx  # type: ignore

        with httpx.Client(follow_redirects=True, timeout=timeout,
                          headers={"User-Agent": _UA}) as c:
            r = c.get(url)
            return r.status_code, r.text
    except Exception:
        pass
    try:
        import requests  # type: ignore

        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": _UA}, allow_redirects=True)
        return r.status_code, r.text
    except Exception:
        return None, ""


def _katana_fetch_body(url: str, headless: bool, timeout: int):
    """Fall back to katana (optionally headless) to capture a page body.
    Returns (status_code, body) or (None, '')."""
    args = ["-u", url, "-depth", "1", "-jsonl", "-silent", "-no-color",
            "-timeout", "15"]
    if headless:
        args.append("-headless")
    rc, out, _err = _run_katana(args, timeout=timeout)
    body, status = "", None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = rec.get("response") or {}
        endpoint = (rec.get("request") or {}).get("endpoint") or rec.get("url")
        b = resp.get("body")
        if b and (body == "" or endpoint == url):
            body, status = b, resp.get("status_code")
    return status, body


def katana_extract(url: str, max_chars: int = 8000,
                   headless: bool = False, task_id: "str | None" = None) -> str:
    if not url or not url.strip():
        return json.dumps({"success": False, "error": "url is required"})
    url = url.strip()

    status, body, used = None, "", None

    # 1) Fast path: direct HTTP GET (sub-second for static pages).
    if not headless:
        status, body = _direct_fetch(url)
        if status and 200 <= status < 300 and body.strip():
            used = "direct"
        else:
            body = ""  # blocked / non-2xx / empty -> fall through to katana

    # 2) Fallback / forced: katana (headless handles JS-walled pages).
    if not body:
        status, body = _katana_fetch_body(url, headless=headless, timeout=45)
        used = "katana-headless" if headless else "katana"

    if not body:
        return json.dumps({"success": False, "url": url, "status_code": status,
                           "error": "no content captured (try headless=true "
                                    "for JS-rendered or bot-protected pages)"})

    title = _extract_title(body)
    text = _html_to_text(body)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return json.dumps({
        "success": True, "url": url, "title": title,
        "status_code": status, "chars": len(text),
        "truncated": truncated, "content": text, "source": used,
    })


# ---------------------------------------------------------------------------
# katana_crawl — spider -> discovered URLs
# ---------------------------------------------------------------------------
def katana_crawl(url: str, depth: int = 2, max_pages: int = 100,
                 same_domain: bool = True, headless: bool = False,
                 task_id: "str | None" = None) -> str:
    if not url or not url.strip():
        return json.dumps({"success": False, "error": "url is required"})
    url = url.strip()
    depth = max(1, min(int(depth), 5))
    max_pages = max(1, min(int(max_pages), 1000))

    args = ["-u", url, "-depth", str(depth), "-silent", "-no-color",
            "-timeout", "20"]
    if same_domain:
        args += ["-field-scope", "rdn"]  # root-domain scope
    if headless:
        args.append("-headless")

    # cap runtime proportional to crawl size
    timeout = min(300, 30 + depth * 45)
    rc, out, err = _run_katana(args, timeout=timeout)

    urls = []
    seen = set()
    for line in out.splitlines():
        u = line.strip()
        if u and u not in seen and u.startswith("http"):
            seen.add(u)
            urls.append(u)
        if len(urls) >= max_pages:
            break

    if not urls and rc != 0:
        return json.dumps({"success": False, "url": url,
                           "error": err.strip() or "crawl failed"})

    return json.dumps({
        "success": True, "url": url, "depth": depth,
        "count": len(urls), "capped": len(urls) >= max_pages,
        "urls": urls, "source": "katana",
    })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
registry.register(
    name="katana_extract",
    toolset="katana",
    description="Extract clean text content from a single URL using the local "
                "katana crawler (self-hosted, no API key). web_extract analog.",
    emoji="🗡️",
    schema={
        "name": "katana_extract",
        "description": "Fetch a single URL and return its readable text "
                       "content, extracted via the local katana binary "
                       "(ProjectDiscovery). No external API or key required. "
                       "Use for reading articles, docs, and pages when the "
                       "configured web_extract backend is unavailable.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "The URL to extract content from."},
                "max_chars": {"type": "integer",
                              "description": "Max characters of text to return "
                                             "(default 8000)."},
                "headless": {"type": "boolean",
                             "description": "Use headless Chrome for JS-heavy "
                                            "pages (slower). Default false."},
            },
            "required": ["url"],
        },
    },
    handler=lambda args, **kw: katana_extract(
        url=args.get("url", ""),
        max_chars=int(args.get("max_chars", 8000) or 8000),
        headless=bool(args.get("headless", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)

registry.register(
    name="katana_crawl",
    toolset="katana",
    description="Spider a site with the local katana crawler and return "
                "discovered URLs (depth-limited, domain-scoped).",
    emoji="🕷️",
    schema={
        "name": "katana_crawl",
        "description": "Crawl a website starting from a URL using the local "
                       "katana binary and return the list of discovered URLs. "
                       "Depth- and page-limited. Use to map a site or gather "
                       "links before extracting specific pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "Seed URL to start crawling from."},
                "depth": {"type": "integer",
                          "description": "Max crawl depth 1-5 (default 2)."},
                "max_pages": {"type": "integer",
                              "description": "Max URLs to return (default 100)."},
                "same_domain": {"type": "boolean",
                                "description": "Restrict to the seed's root "
                                               "domain (default true)."},
                "headless": {"type": "boolean",
                             "description": "Use headless Chrome (default false)."},
            },
            "required": ["url"],
        },
    },
    handler=lambda args, **kw: katana_crawl(
        url=args.get("url", ""),
        depth=int(args.get("depth", 2) or 2),
        max_pages=int(args.get("max_pages", 100) or 100),
        same_domain=bool(args.get("same_domain", True)),
        headless=bool(args.get("headless", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
