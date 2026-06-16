# ABOUTME: Steel scrape tool for server-side content extraction as markdown/HTML.
# ABOUTME: Uses Steel's standalone scrape API — no browser session required.
"""Steel Scrape Tool — extract page content via Steel's server-side scrape API.

Unlike browser_navigate + browser_snapshot which use a full browser session and
accessibility tree, this tool extracts content server-side as clean markdown or
HTML.  It does not require an active browser session and is useful for quick
content extraction without interactive browsing.

Environment Variables:
- STEEL_API_KEY: Required — Steel API key
- STEEL_USE_PROXY: Optional — enable residential proxy for scraping
"""

import json
import logging
import os
from typing import Optional

import requests
from tools.url_safety import is_safe_url as _is_safe_url

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.steel.dev"


def check_steel_scrape_requirements() -> bool:
    """Return True when STEEL_API_KEY is available."""
    return bool(os.environ.get("STEEL_API_KEY"))


def steel_scrape(
    url: str,
    format: str = "markdown",
    use_proxy: Optional[bool] = None,
) -> str:
    """Scrape a URL via Steel's server-side API and return extracted content.

    Args:
        url: The URL to scrape.
        format: Output format — "markdown", "html", "readability", or "cleaned_html".
        use_proxy: Override proxy setting (defaults to STEEL_USE_PROXY env var).

    Returns:
        JSON string with extracted content, metadata, and links.
    """
    if not _is_safe_url(url):
        return json.dumps({
            "error": "Blocked: URL targets a private or internal address",
        })

    api_key = os.environ.get("STEEL_API_KEY")
    if not api_key:
        return json.dumps({"error": "STEEL_API_KEY not configured"})

    base = os.environ.get("STEEL_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

    body = {
        "url": url,
        "format": [format],
    }

    if use_proxy is None:
        use_proxy = os.environ.get("STEEL_USE_PROXY", "").lower() in ("1", "true", "yes")
    if use_proxy:
        body["use_proxy"] = True

    try:
        response = requests.post(
            f"{base}/v1/scrape",
            headers={
                "Content-Type": "application/json",
                "Steel-Api-Key": api_key,
            },
            json=body,
            timeout=60,
        )

        if not response.ok:
            return json.dumps({
                "error": f"Steel scrape failed: {response.status_code} {response.text[:500]}"
            })

        data = response.json()

        result = {
            "url": url,
            "format": format,
        }

        # Extract content — try requested format first, then fall back
        content = data.get("content", {})
        if content.get(format):
            result["content"] = content[format]
        else:
            for key in ("markdown", "readability", "cleaned_html", "html"):
                if content.get(key):
                    result["content"] = content[key]
                    result["format"] = key
                    break

        metadata = data.get("metadata", {})
        if metadata.get("title"):
            result["title"] = metadata["title"]
        if metadata.get("description"):
            result["description"] = metadata["description"]
        if metadata.get("statusCode"):
            result["status_code"] = metadata["statusCode"]

        links = data.get("links", [])
        if links:
            result["links_count"] = len(links)
            result["links"] = links[:20]

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Schema ---

STEEL_SCRAPE_SCHEMA = {
    "name": "steel_scrape",
    "description": (
        "Extract content from a webpage using Steel's server-side scrape API. "
        "Returns clean markdown or HTML without needing a browser session. "
        "Best for quick content extraction; use browser_navigate for interactive browsing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the webpage to scrape",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "html", "readability", "cleaned_html"],
                "description": "Output format (default: markdown)",
                "default": "markdown",
            },
        },
        "required": ["url"],
    },
}


# --- Registration ---

from tools.registry import registry

registry.register(
    name="steel_scrape",
    toolset="browser",
    schema=STEEL_SCRAPE_SCHEMA,
    handler=lambda args, **kw: steel_scrape(
        url=args.get("url", ""),
        format=args.get("format", "markdown"),
    ),
    check_fn=check_steel_scrape_requirements,
    requires_env=["STEEL_API_KEY"],
)
