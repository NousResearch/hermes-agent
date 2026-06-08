"""DuckDuckGo web search provider via the ``ddgs`` Python package.

DuckDuckGo does not provide an official programmatic search API.  The
community-maintained `ddgs <https://pypi.org/project/ddgs/>`_ package (the
renamed successor of ``duckduckgo-search``) scrapes DuckDuckGo's HTML results
page and normalizes them.  It implements ``WebSearchProvider`` only — there is
no extract capability.

Configuration::

    # No API key required. Enable by installing the package and pointing the
    # web backend at ddgs:
    pip install ddgs

    # ~/.hermes/config.yaml
    web:
      search_backend: "ddgs"
      extract_backend: "firecrawl"    # pair with an extract provider if needed

Rate limits are enforced server-side by DuckDuckGo.  Expect intermittent
``DuckDuckGoSearchException`` / 202 responses under heavy use; this provider
surfaces them as ``{"success": False, "error": ...}`` rather than crashing
the tool call.

See https://duckduckgo.com/?q=duckduckgo+tos for terms of use.
"""

from __future__ import annotations

import functools
import logging
from importlib import metadata, util
from pathlib import Path
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


def _path_within(child: Path, parent: Path) -> bool:
    """Return True when ``child`` resolves inside ``parent``."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@functools.lru_cache(maxsize=1)
def ddgs_package_available() -> bool:
    """Return True when the installed ``ddgs`` distribution owns the import target.

    Availability checks must not import ``ddgs``: imports execute top-level code
    and can be shadowed by attacker-written files in the current working
    directory.  Distribution metadata and ``find_spec`` locate the package
    without executing it, then verify Python would import from that installed
    distribution rather than a local ``ddgs.py``.
    """
    try:
        distribution = metadata.distribution("ddgs")
    except metadata.PackageNotFoundError:
        return False

    try:
        spec = util.find_spec("ddgs")
    except (ImportError, ValueError):
        return False
    if spec is None or spec.origin is None:
        return False

    try:
        distribution_root = Path(distribution.locate_file("")).resolve()
        module_origin = Path(spec.origin).resolve()
    except OSError:
        return False

    return module_origin == distribution_root or _path_within(
        module_origin, distribution_root
    )


class DDGSSearchProvider(WebSearchProvider):
    """Search via the ``ddgs`` package (DuckDuckGo HTML scrape).

    No API key required.  The provider is considered "configured" when the
    installed ``ddgs`` distribution owns the module Python would import —
    there is nothing else to set up.
    """

    def provider_name(self) -> str:
        return "ddgs"

    def is_configured(self) -> bool:
        """Return True when the installed ``ddgs`` package is safe to import.

        Called at tool-registration time; must not perform network I/O or
        execute package imports.
        """
        return ddgs_package_available()

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized results.

        Returns ``{"success": True, "data": {"web": [...]}}`` on success or
        ``{"success": False, "error": str}`` on failure (missing package,
        rate-limited, network error, etc.).
        """
        if not ddgs_package_available():
            return {
                "success": False,
                "error": "ddgs package is not installed or is shadowed — run `pip install ddgs`",
            }

        # DDGS().text yields at most `max_results` items; we cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        try:
            from ddgs import DDGS  # type: ignore
            web_results = []
            with DDGS() as client:
                for i, hit in enumerate(client.text(query, max_results=safe_limit)):
                    if i >= safe_limit:
                        break
                    url = str(hit.get("href") or hit.get("url") or "")
                    web_results.append({
                        "title": str(hit.get("title", "")),
                        "url": url,
                        "description": str(hit.get("body", "")),
                        "position": i + 1,
                    })
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        logger.info(
            "DDGS search '%s': %d results (limit %d)", query, len(web_results), limit
        )
        return {"success": True, "data": {"web": web_results}}
