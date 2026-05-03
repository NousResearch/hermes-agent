"""
local_web_tools.py — free-tier web search and extraction for Hermes Agent.

Provides `local_web_search_tool` and `local_web_extract_tool` with the same
JSON contracts as `web_search_tool` and `web_extract_tool` in `web_tools.py`,
but using free local-first backends instead of paid APIs.

Search backend chain (auto-detected, falls through):
    1. SearXNG at $SEARXNG_URL (default http://localhost:8888) — self-hosted aggregator
    2. Brave Search API (if $BRAVE_SEARCH_API_KEY set) — free tier 2K/mo
    3. Tavily API (if $TAVILY_API_KEY set) — free tier 1K/mo
    4. ddgr CLI (DuckDuckGo, no key) — if installed
    5. duckduckgo_search Python package (no key) — if installed

Extraction backend:
    - lynx -dump (must be installed: `apt install lynx` or `brew install lynx`)
    - Optional local-LLM summarization via any OpenAI-compatible endpoint at
      $LLM_BASE_URL (Ollama default; works with llama.cpp's llama-server, vLLM,
      LM Studio — anywhere that exposes /v1/chat/completions). Zero API cost.

Designed to be a drop-in alternative to web_tools.py for users who don't have
Firecrawl / Parallel / Exa keys. The tools dispatch to the same JSON schema, so
skills calling them behave identically regardless of backend.

Usage from Hermes Agent: register these tools in toolsets.py alongside the
paid `web_search` / `web_extract` tools. Users can opt-in via config:

    tools:
      web_search: local        # use local_web_search_tool
      web_extract: local       # use local_web_extract_tool

License: MIT (matches Hermes Agent project)
"""
import json
import logging
import os
import re
import subprocess
import urllib.parse
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# ============================================================
# Config (env-overridable, sane defaults)
# ============================================================
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Local LLM endpoint (OpenAI-compatible chat-completions API).
# Supports any backend that exposes /v1/chat/completions:
#   - Ollama:        http://localhost:11434       (default)
#   - llama.cpp:     http://localhost:8080        (set LLM_BASE_URL=http://localhost:8080)
#   - vLLM:          http://localhost:8000        (set LLM_BASE_URL=http://localhost:8000)
#   - LM Studio:     http://localhost:1234        (set LLM_BASE_URL=http://localhost:1234)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "qwen3:8b")

LOCAL_WEB_USER_AGENT = os.getenv("LOCAL_WEB_USER_AGENT", "hermes-agent/local-web-tools")

LYNX_TIMEOUT = 25
SEARCH_TIMEOUT = 15

DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000


# ============================================================
# Search backends
# ============================================================
def _search_searxng(query: str, limit: int) -> Optional[List[dict]]:
    """Self-hosted SearXNG instance — aggregates Google/Bing/DDG/Brave/Startpage."""
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "language": "en"})
        r = requests.get(
            f"{SEARXNG_URL}/search?{params}",
            headers={"User-Agent": LOCAL_WEB_USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return [
            {
                "title": (x.get("title") or "")[:200],
                "url": x.get("url") or "",
                "description": (x.get("content") or "")[:300],
                "position": i + 1,
            }
            for i, x in enumerate((data.get("results") or [])[:limit])
        ]
    except Exception as e:
        logger.debug("SearXNG search failed: %s", e)
        return None


def _search_brave(query: str, limit: int) -> Optional[List[dict]]:
    """Brave Search API — free tier 2K queries/month."""
    if not BRAVE_SEARCH_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": BRAVE_SEARCH_API_KEY,
                     "Accept": "application/json"},
            params={"q": query, "count": min(limit, 20)},
            timeout=SEARCH_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        web_results = (data.get("web") or {}).get("results") or []
        return [
            {
                "title": (x.get("title") or "")[:200],
                "url": x.get("url") or "",
                "description": (x.get("description") or "")[:300],
                "position": i + 1,
            }
            for i, x in enumerate(web_results[:limit])
        ]
    except Exception as e:
        logger.debug("Brave search failed: %s", e)
        return None


def _search_tavily(query: str, limit: int) -> Optional[List[dict]]:
    """Tavily API — free tier 1K queries/month."""
    if not TAVILY_API_KEY:
        return None
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query,
                  "max_results": limit, "search_depth": "basic"},
            timeout=SEARCH_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return [
            {
                "title": (x.get("title") or "")[:200],
                "url": x.get("url") or "",
                "description": (x.get("content") or "")[:300],
                "position": i + 1,
            }
            for i, x in enumerate((data.get("results") or [])[:limit])
        ]
    except Exception as e:
        logger.debug("Tavily search failed: %s", e)
        return None


def _search_ddgr(query: str, limit: int) -> Optional[List[dict]]:
    """ddgr CLI (DuckDuckGo) — no API key, install via `pip install ddgr`."""
    try:
        r = subprocess.run(
            ["ddgr", "--json", "-n", str(limit), query],
            capture_output=True, text=True, timeout=SEARCH_TIMEOUT,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout)
        return [
            {
                "title": x.get("title") or "",
                "url": x.get("url") or "",
                "description": (x.get("abstract") or "")[:300],
                "position": i + 1,
            }
            for i, x in enumerate(data[:limit])
        ]
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.debug("ddgr search failed: %s", e)
        return None


def _search_ddg_python(query: str, limit: int) -> Optional[List[dict]]:
    """DuckDuckGo via Python package. Tries `ddgs` (new) then `duckduckgo_search` (legacy)."""
    DDGS = None
    try:
        from ddgs import DDGS as _DDGS
        DDGS = _DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS  # legacy package
            DDGS = _DDGS
        except ImportError:
            return None
    try:
        results = list(DDGS().text(query, max_results=limit))
        return [
            {
                "title": x.get("title") or "",
                "url": x.get("href") or "",
                "description": (x.get("body") or "")[:300],
                "position": i + 1,
            }
            for i, x in enumerate(results[:limit])
        ]
    except Exception as e:
        logger.debug("duckduckgo Python lib failed: %s", e)
        return None


# Order matters: prefer self-hosted, then free APIs, then CLI/package fallbacks.
SEARCH_BACKENDS = [
    ("searxng", _search_searxng),
    ("brave", _search_brave),
    ("tavily", _search_tavily),
    ("ddgr", _search_ddgr),
    ("ddg-python", _search_ddg_python),
]


def _try_search_chain(query: str, limit: int) -> tuple[List[dict], str]:
    """Try search backends in order; return (results, backend_name).

    Returns ([], 'none') if all backends fail.
    """
    for name, fn in SEARCH_BACKENDS:
        results = fn(query, limit)
        if results:
            return results, name
    return [], "none"


# ============================================================
# Content extraction (lynx) + cleaning
# ============================================================
NAV_PATTERNS = [
    r"(?im)^[ \t]*\*\s+(home|about|services|contact|reports|press releases|verticals|search|sign in|menu|toggle navigation|skip to content)[\s.]*$",
    r"(?im)^[ \t]*\(button\)[^\n]*$",
    r"(?im)^[ \t]*iframe:[^\n]*$",
    r"(?im)^[ \t]*\[[^\]]*\.(ico|png|svg|gif|jpe?g)[^\]]*\][^\n]*$",
    r"(?im)^[ \t]*captcha validation[^\n]*$",
    r"(?im)^[ \t]*skip to (main )?content[^\n]*$",
    r"(?im)^[ \t]*\(accessibility\)[^\n]*$",
    r"(?im)^[ \t]*cookie[s]?\s+(notice|policy|preferences)[^\n]*$",
    r"(?im)^[ \t]*subscribe to[^\n]*$",
    r"(?m)^[ \t]*_+[ \t]*$",  # underscore form fields
]


def _clean_lynx_dump(text: str) -> str:
    """Strip nav menus, button labels, iframe markers, cookie notices, form-field underscores."""
    for pat in NAV_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_lynx(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch a URL via lynx -dump. Returns (cleaned_text, error_or_None)."""
    try:
        r = subprocess.run(
            ["lynx", "-dump", "-nolist", "-width=120", url],
            capture_output=True, text=True, timeout=LYNX_TIMEOUT,
        )
        if r.returncode != 0:
            return None, f"lynx exit {r.returncode}"
        if len(r.stdout) < 200:
            return None, f"body too short ({len(r.stdout)} chars)"
        return _clean_lynx_dump(r.stdout), None
    except FileNotFoundError:
        return None, "lynx not installed (apt install lynx OR brew install lynx)"
    except subprocess.TimeoutExpired:
        return None, f"fetch timeout ({LYNX_TIMEOUT}s)"
    except Exception as e:
        return None, str(e)


# ============================================================
# Optional Ollama-based summarization (zero-cost LLM processing)
# ============================================================
def _is_qwen35_or_36(model_name: str) -> bool:
    """Detect Qwen3.5/3.6 models by tag. Used to set chat_template_kwargs.enable_thinking=false
    automatically — Qwen3.5/3.6 default to thinking mode and do NOT honor /no_think the way
    Qwen3 did."""
    n = (model_name or "").lower().replace("_", ".")
    return "qwen3.5" in n or "qwen3.6" in n


def _summarize_via_local_llm(content: str, model: Optional[str] = None,
                              target_chars: int = 2000) -> Optional[str]:
    """Summarize content via a local OpenAI-compatible LLM endpoint
    (Ollama, llama.cpp's llama-server, vLLM, LM Studio — auto via LLM_BASE_URL).
    Returns None if endpoint unreachable or fails.

    For Qwen3.5/3.6 models, automatically passes chat_template_kwargs.enable_thinking=false
    so the summary doesn't include <think> reasoning blocks."""
    model = model or LLM_DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": ("Summarize the user's web page text in concise plain prose. "
                         "Preserve specific names, numbers, dates, and URLs verbatim. "
                         f"Target length: ~{target_chars} characters. "
                         "Do not add commentary or reasoning preamble.")},
            {"role": "user", "content": content[:20000]},  # cap input
        ],
        "temperature": 0.2,
        "max_tokens": int(target_chars / 3),  # approx tokens
    }
    if _is_qwen35_or_36(model):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = requests.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        # Strip Qwen3-style think blocks if present (defensive)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
        out = re.sub(r"<think>.*$", "", out, flags=re.DOTALL).strip()
        return out
    except Exception as e:
        logger.debug("Ollama summarization failed: %s", e)
        return None


# ============================================================
# Public tool: local_web_search_tool
# ============================================================
def local_web_search_tool(query: str, limit: int = 5) -> str:
    """
    Free-tier web search. Drop-in alternative to `web_search_tool` from web_tools.py.

    Backend chain (auto-fallback): SearXNG → Brave → Tavily → ddgr → duckduckgo_search.
    No paid API key required if SearXNG is self-hosted (recommended) or ddgr is installed.

    Args:
        query (str): The search query
        limit (int): Maximum number of results (1-100, default 5)

    Returns:
        str: JSON string with the same shape as web_search_tool:
             {
                 "success": bool,
                 "data": {"web": [{"title", "url", "description", "position"}, ...]},
                 "backend": str,        # which backend produced these results
                 "error": str | null,
             }
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 100)

    results, backend = _try_search_chain(query, limit)

    if not results:
        return json.dumps({
            "success": False,
            "data": {"web": []},
            "backend": "none",
            "error": (
                "No search backend available. Install one of: "
                "(a) self-host SearXNG at $SEARXNG_URL (recommended, free), "
                "(b) set $BRAVE_SEARCH_API_KEY (free tier 2K/mo), "
                "(c) set $TAVILY_API_KEY (free tier 1K/mo), "
                "(d) install ddgr (`pip install ddgr`), "
                "(e) install duckduckgo_search (`pip install duckduckgo_search`)."
            ),
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "data": {"web": results},
        "backend": backend,
        "error": None,
    }, ensure_ascii=False)


# Tool schema for Hermes registry
local_web_search_tool_schema = {
    "type": "function",
    "function": {
        "name": "local_web_search",
        "description": (
            "Free-tier web search via local SearXNG, Brave/Tavily free tier, ddgr, or "
            "duckduckgo_search. Same return shape as `web_search`. No paid API required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Max results (1-100)", "default": 5},
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# Public tool: local_web_extract_tool
# ============================================================
def local_web_extract_tool(
    urls: List[str],
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION,
) -> str:
    """
    Free-tier web content extraction. Drop-in alternative to `web_extract_tool`.

    Fetches via lynx (text-mode browser; install: apt install lynx / brew install lynx).
    Strips boilerplate (nav menus, button labels, cookie notices, etc.) before returning.

    With use_llm_processing=True and a local OpenAI-compat LLM endpoint reachable at
    $LLM_BASE_URL (Ollama / llama.cpp / vLLM / LM Studio), summarizes pages longer
    than `min_length` chars. Otherwise returns full cleaned text. Default model: qwen3:8b
    (override via $LLM_DEFAULT_MODEL or the `model` argument).

    Args:
        urls (List[str]): URLs to extract
        use_llm_processing (bool): If True and Ollama is reachable, summarize long pages
        model (Optional[str]): Ollama model tag (default: qwen3:8b)
        min_length (int): Char threshold above which LLM summarization is applied

    Returns:
        str: JSON with the same shape as web_extract_tool:
             {
                 "success": bool,
                 "data": [{"url", "domain", "content", "char_count", "summarized", ...}, ...],
                 "error": str | null,
             }
    """
    if not isinstance(urls, list):
        urls = [urls]
    if not urls:
        return json.dumps({"success": False, "data": [],
                           "error": "no URLs provided"}, ensure_ascii=False)

    summarizer_model = model or "qwen3:8b"
    pages = []
    errors = []

    for url in urls:
        if not url or not url.startswith(("http://", "https://")):
            errors.append({"url": url, "error": "invalid URL"})
            continue

        text, err = _fetch_lynx(url)
        if err:
            errors.append({"url": url, "error": err})
            continue

        domain = re.sub(r"^https?://", "", url).split("/")[0].replace("www.", "")
        content = text
        summarized = False

        if use_llm_processing and len(text) >= min_length:
            summary = _summarize_via_local_llm(text, model=summarizer_model,
                                         target_chars=2000)
            if summary:
                content = summary
                summarized = True

        pages.append({
            "url": url,
            "domain": domain,
            "content": content,
            "char_count": len(content),
            "raw_char_count": len(text),
            "summarized": summarized,
        })

    return json.dumps({
        "success": bool(pages),
        "data": pages,
        "errors": errors if errors else None,
        "error": None if pages else (
            "All extractions failed. Check that lynx is installed: "
            "`apt install lynx` (Debian/Ubuntu) / `brew install lynx` (macOS)."
        ),
    }, ensure_ascii=False)


local_web_extract_tool_schema = {
    "type": "function",
    "function": {
        "name": "local_web_extract",
        "description": (
            "Free-tier web extraction. Fetches URLs via lynx, strips nav/footer "
            "boilerplate, optionally summarizes via local Ollama. Same return "
            "shape as `web_extract`. Requires lynx to be installed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"},
                         "description": "List of URLs to extract"},
                "use_llm_processing": {"type": "boolean",
                                        "description": "Summarize long pages via local Ollama",
                                        "default": True},
                "model": {"type": "string",
                          "description": "Ollama model tag for summarization",
                          "default": "qwen3:8b"},
                "min_length": {"type": "integer",
                               "description": "Char threshold above which to summarize",
                               "default": 5000},
            },
            "required": ["urls"],
        },
    },
}


# ============================================================
# Optional: smoke-test entry point (for `python -m tools.local_web_tools`)
# ============================================================
def _smoke_test():
    """Quick sanity check — searches a known query, extracts the first result."""
    print("# local_web_tools smoke test\n")
    print("## search backend availability")
    for name, fn in SEARCH_BACKENDS:
        results = fn("hermes agent nous research", 1)
        print(f"  {name:14s} {'OK' if results else 'unavailable'}")

    print("\n## sample search via fallback chain")
    out = local_web_search_tool("nous research hermes agent", 3)
    data = json.loads(out)
    print(f"  backend used: {data.get('backend')}")
    print(f"  results: {len(data.get('data', {}).get('web', []))}")
    if data.get("data", {}).get("web"):
        first = data["data"]["web"][0]
        print(f"  first result: {first['title'][:80]}\n              {first['url']}")

        print("\n## sample extract")
        ex = local_web_extract_tool([first["url"]], use_llm_processing=False)
        ex_data = json.loads(ex)
        if ex_data.get("data"):
            page = ex_data["data"][0]
            print(f"  url: {page['url']}")
            print(f"  chars: {page['char_count']}  summarized: {page['summarized']}")
            print(f"  preview: {page['content'][:200]!r}")
        else:
            print(f"  extraction failed: {ex_data.get('errors')}")


if __name__ == "__main__":
    _smoke_test()
