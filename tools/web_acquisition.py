"""Native acquisition for callers which need structured outcomes, without truncation."""

from agent.web_acquisition import (
    WebExtractRequest,
    WebExtractResult,
    WebSearchRequest,
    WebSearchResult,
)


def web_search(request: WebSearchRequest) -> WebSearchResult:
    """Use the same routing, cache and fallback pipeline as the Hermes search tool.

    Results contain safe failure facts and actual attempt provenance. Upstream
    diagnostics and arbitrary vendor metadata are excluded from this API.
    """
    from tools.web_tools_search import acquire_search

    return acquire_search(request).outcome


async def web_extract(request: WebExtractRequest) -> WebExtractResult:
    """Use native URL controls, routing, extraction, rescue and coverage-aware caching."""
    from tools.web_tools_extract import acquire_extract

    return (await acquire_extract(request)).outcome
