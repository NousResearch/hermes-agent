#!/usr/bin/env python3
"""
Extract Provider Router — Layer B, web_extract side (R1 §4 / §11).

Minimal public-URL extraction routing for web_extract_tool. Runs only when
``web.router.enabled`` is true; when false, web_extract_tool never imports
this module.

B1 boundaries (R1 §11 / §15.2):
  - privacy/interaction checks happen BEFORE any external extractor;
  - browser-only / login / interaction / sensitive-parameter URLs produce a
    structured Browser escalation recommendation (NEVER an automatic Browser
    call from inside web_extract_tool);
  - normal public URLs select Firecrawl;
  - at most ONE extractor per call, NO secondary extract provider in B1;
  - no URL is sent to Exa + Tavily + Parallel + Firecrawl in sequence.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from tools.web_router import (
    DEFAULT_EXTRACT_PROVIDER,
    ExtractRouterDecision,
    browser_escalation_reason,
    normalize_browser_only_domains,
    provider_runtime_ready,
    url_requires_browser,
)


def select_extract_provider(
    urls: Iterable[str],
    browser_only_domains: Optional[Sequence[str]] = None,
    registry_getter: Optional[Callable[[str], Any]] = None,
    env_has: Optional[Callable[[str], bool]] = None,
) -> ExtractRouterDecision:
    """Construct ONE ExtractRouterDecision for a web_extract call.

    Order (R1 §11):
      1. URL boundary classification — any URL that requires Browser yields an
         escalation recommendation (no external call is made for that set);
      2. otherwise select Firecrawl (default public-page extractor) if it is
         registered + credential-present + extract-capable;
      3. if no extractor is usable, return an escalation recommendation
         (B1 has NO secondary extract provider).

    Never extracts anything. Pure decision construction.
    """
    decision = ExtractRouterDecision()
    domains = normalize_browser_only_domains(browser_only_domains)

    url_list = list(urls or [])
    if not url_list:
        decision.selected_provider = DEFAULT_EXTRACT_PROVIDER
        decision.selection_reason = "no_urls_default_extractor"
        return decision

    # 1) Privacy + interaction boundary BEFORE any external extractor.
    for url in url_list:
        if url_requires_browser(url, domains):
            decision.escalation_recommended = True
            decision.escalation_tool = "browser_navigate"
            decision.escalation_reason = browser_escalation_reason(url, domains)
            decision.selection_reason = "browser_required"
            decision.selected_provider = None
            return decision

    # 2) Normal public URL(s) -> Firecrawl, if usable.
    if _extractor_usable(DEFAULT_EXTRACT_PROVIDER, registry_getter, env_has):
        decision.selected_provider = DEFAULT_EXTRACT_PROVIDER
        decision.selection_reason = "default_public_extractor"
        return decision

    # 3) No usable extractor in B1 — recommend Browser rather than a
    #    secondary extract chain.
    decision.escalation_recommended = True
    decision.escalation_tool = "browser_navigate"
    decision.escalation_reason = "no_extract_provider_available"
    decision.selection_reason = "no_extract_provider_available"
    return decision


def _extractor_usable(
    name: str,
    registry_getter: Optional[Callable[[str], Any]],
    env_has: Optional[Callable[[str], bool]],
) -> bool:
    return provider_runtime_ready(name, "extract", registry_getter, env_has)
