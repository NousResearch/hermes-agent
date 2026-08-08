#!/usr/bin/env python3
"""
Web Capability Router — Stage B1 foundation.

Canonical decision schemas, deterministic intent classification, and the
five-provider routing registry for the Hermes Web Capability Router V0.1.

LAYER BOUNDARY (R1)
===================
This module is the *provider-router* layer (Layer B). It operates only after
the Agent has already selected a capability (SEARCH or EXTRACT) via the
Capability Policy (Layer A — the ``web-capability-policy`` candidate skill).
It must never claim to enforce NO_WEB or BROWSER; when Browser is required it
returns a structured *escalation recommendation* for the Agent to act on.

Stage B1 scope:
  - canonical decision schemas (serializable, stateless);
  - deterministic intent classification (rule-based, no ML / no LLM);
  - a complete five-provider registry (ddgs, parallel, exa, tavily, firecrawl);
  - deterministic single-provider selection;
  - a feature flag that defaults to false (``web.router.enabled``).

Explicitly NOT in B1 (see R1 §15.2 / §16):
  - runtime fallback execution;
  - verification execution;
  - quota monitoring;
  - live provider tests;
  - automatic Browser invocation;
  - query-content telemetry (decision objects are the only telemetry).

No API keys, page content, cookies, credentials, or billing data are ever
stored in these objects. Quota/pricing figures are deliberately NOT hardcoded
anywhere in this module.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Capability(str, enum.Enum):
    """Mutually-exclusive web capability (Layer A decision output)."""

    NO_WEB = "NO_WEB"
    BROWSER = "BROWSER"
    SEARCH = "SEARCH"
    EXTRACT = "EXTRACT"


class VerificationMode(str, enum.Enum):
    """Orthogonal verification mode (NOT a capability — R1 §6)."""

    NONE = "NONE"
    SECOND_PROVIDER = "SECOND_PROVIDER"
    PRIMARY_SOURCE_REQUIRED = "PRIMARY_SOURCE_REQUIRED"


class SearchIntent(str, enum.Enum):
    """Deterministic search-intent classes (R1 §7 / §8)."""

    SIMPLE_DISCOVERY = "SIMPLE_DISCOVERY"
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    TECHNICAL_RESEARCH = "TECHNICAL_RESEARCH"
    CURRENT_INFORMATION = "CURRENT_INFORMATION"


# ---------------------------------------------------------------------------
# Intent classification — deterministic, rule-based, bilingual (zh/en)
# ---------------------------------------------------------------------------

# Precedence (highest wins): CURRENT_INFORMATION > TECHNICAL_RESEARCH
# > SIMPLE_DISCOVERY > GENERAL_RESEARCH. Refined from R1 §8 and validated
# against the local integration. Signals are deliberately small keyword sets;
# no ML, no LLM, no external calls.

_INTENT_SIGNALS: Dict[SearchIntent, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    SearchIntent.SIMPLE_DISCOVERY: (
        # Chinese signals
        ("官网", "主页", "官方网站", "首页", "网站"),
        # English signals. The contiguous "official site"/"official website"
        # strings are intentionally NOT listed here: they are covered by
        # _has_official_site_discovery_signal (word-boundary token match),
        # which also handles entity words between "official" and the site
        # word and correctly excludes plurals like "official websites".
        ("homepage", "home page", "website for", "website of"),
    ),
    SearchIntent.TECHNICAL_RESEARCH: (
        ("api 文档", "api文档", "技术文档", "文档", "论文", "规范", "架构",
         "语义搜索", "相关页面", "接口文档", "开发文档", "sdk"),
        ("documentation", "docs", "paper", "papers", "specification",
         "architecture", "semantic search", "related pages", "api reference",
         "sdk", "technical", "arxiv"),
    ),
    SearchIntent.CURRENT_INFORMATION: (
        ("最新", "当前", "目前", "最近", "新闻", "价格", "政策", "今日",
         "发布", "更新"),
        ("current", "latest", "recent", "news", "price", "prices", "policy",
         "release", "released", "today", "breaking"),
    ),
    SearchIntent.GENERAL_RESEARCH: ((), ()),  # default — always matches
}

# Order matters: highest precedence first (R1 §8).
_INTENT_PRECEDENCE: Tuple[SearchIntent, ...] = (
    SearchIntent.CURRENT_INFORMATION,
    SearchIntent.TECHNICAL_RESEARCH,
    SearchIntent.SIMPLE_DISCOVERY,
    SearchIntent.GENERAL_RESEARCH,
)


def _has_official_site_discovery_signal(text: str) -> bool:
    """True when an official-site discovery pattern appears in *text*.

    Matches the token combination ``official`` followed (within a small
    window) by ``website`` / ``site`` / ``homepage``, even when one or more
    entity words occur between them — e.g. "Find the official OpenAI
    website". Plural forms ("websites", "sites") deliberately do NOT match,
    so comparative queries like "Compare the official websites of several
    search providers" stay GENERAL_RESEARCH. Higher-precedence intents
    (CURRENT_INFORMATION, TECHNICAL_RESEARCH) are already matched before
    this signal is consulted by :func:`classify_search_intent`.
    """
    tokens = re.findall(r"[a-z]+", text)
    targets = {"website", "site", "homepage"}
    for i, token in enumerate(tokens):
        if token != "official":
            continue
        for j in range(i + 1, min(i + 4, len(tokens))):
            if tokens[j] in targets:
                return True
    return False


def classify_search_intent(query: str) -> SearchIntent:
    """Return the deterministic intent class for *query*.

    Rule-based keyword matching over lowercased text (English signals) and
    raw text (Chinese signals — case is irrelevant for CJK). Precedence:
    CURRENT_INFORMATION > TECHNICAL_RESEARCH > SIMPLE_DISCOVERY
    > GENERAL_RESEARCH. GENERAL_RESEARCH is the always-matching default, so
    this function never returns None for non-empty input.

    Empty/whitespace input maps to GENERAL_RESEARCH so the provider selector
    still has a deterministic answer.
    """
    if not query:
        return SearchIntent.GENERAL_RESEARCH
    lowered = query.lower()
    for intent in _INTENT_PRECEDENCE:
        zh_signals, en_signals = _INTENT_SIGNALS[intent]
        if any(sig in query for sig in zh_signals):
            return intent
        if any(sig in lowered for sig in en_signals):
            return intent
        if (
            intent is SearchIntent.SIMPLE_DISCOVERY
            and _has_official_site_discovery_signal(lowered)
        ):
            return intent
    return SearchIntent.GENERAL_RESEARCH


def normalize_intent_hint(intent_hint: Optional[str]) -> Optional[SearchIntent]:
    """Coerce a caller-supplied intent hint to a valid SearchIntent.

    Returns None for unknown/empty hints so callers fall back to local
    classification (the classifier never depends on the Agent emitting a
    hint — R1 §7).
    """
    if not intent_hint:
        return None
    try:
        return SearchIntent(str(intent_hint).strip().upper())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Five-provider routing registry
# ---------------------------------------------------------------------------
#
# Static *routing preferences* only. Runtime facts (registered, plugin
# enabled, credential present, capability supported, Firecrawl deployment)
# are resolved from the existing provider registry + environment at decision
# time — config.yaml is NOT a second source of truth for provider capability.
#
# Parallel MCP is deliberately excluded (R1 §9). Quota/pricing numbers are
# intentionally absent.

#: intent -> preferred provider (R1 §9.2, §10)
INTENT_TO_PROVIDER: Dict[SearchIntent, str] = {
    SearchIntent.SIMPLE_DISCOVERY: "ddgs",
    SearchIntent.GENERAL_RESEARCH: "parallel",
    SearchIntent.TECHNICAL_RESEARCH: "exa",
    SearchIntent.CURRENT_INFORMATION: "tavily",
}

#: default public-page extraction provider (R1 §11)
DEFAULT_EXTRACT_PROVIDER = "firecrawl"

#: deterministic substitute order when the intent-preferred provider is
#: unavailable (used ONLY during decision construction — never executed as
#: a runtime fallback chain in B1). Same normalized order for every intent.
PROVIDER_SUBSTITUTE_ORDER: Tuple[str, ...] = (
    "parallel",
    "exa",
    "tavily",
    "firecrawl",
    "ddgs",
)

#: stable routing preferences for the five in-scope providers.
#: ``capabilities`` mirrors what the provider plugins expose via
#: supports_search()/supports_extract(); ``deployment`` is a hint only —
#: the authoritative Firecrawl deployment is resolved from the environment.
PROVIDER_PREFERENCES: Dict[str, Dict[str, Any]] = {
    "ddgs": {
        "capabilities": ("search",),
        "deployment": "local",
        "intent_affinities": (SearchIntent.SIMPLE_DISCOVERY,),
        "enabled_for_router": True,
        "notes": "simple keyword and official-homepage discovery (free, local)",
    },
    "parallel": {
        "capabilities": ("search", "extract"),
        "deployment": "cloud",
        "intent_affinities": (SearchIntent.GENERAL_RESEARCH,),
        "enabled_for_router": True,
        "notes": "general and broad research",
    },
    "exa": {
        "capabilities": ("search", "extract"),
        "deployment": "cloud",
        "intent_affinities": (SearchIntent.TECHNICAL_RESEARCH,),
        "enabled_for_router": True,
        "notes": "technical, semantic, paper, niche, related-page, API-doc discovery",
    },
    "tavily": {
        "capabilities": ("search", "extract"),
        "deployment": "cloud",
        "intent_affinities": (SearchIntent.CURRENT_INFORMATION,),
        "enabled_for_router": True,
        "notes": "current prices, policies, news, recent releases, rapidly changing info",
    },
    "firecrawl": {
        "capabilities": ("search", "extract"),
        "deployment": "cloud",  # resolved at runtime; self-hosted via FIRECRAWL_API_URL
        "intent_affinities": (SearchIntent.GENERAL_RESEARCH,),
        "enabled_for_router": True,
        "notes": "public-page extraction and optional search where locally supported",
    },
}

#: default browser-only domain suffixes (safe suffix matching).
#: Shipped empty: operators opt in with their own interactive-only domains;
#: no vendor domain is hard-coded into the runtime default.
DEFAULT_BROWSER_ONLY_DOMAINS: Tuple[str, ...] = ()


def normalize_browser_only_domains(raw: Any) -> Tuple[str, ...]:
    """Coerce config-sourced ``browser_only_domains`` to a tuple of suffixes.

    ``hermes config set`` writes YAML scalars as strings, so a user who ran
    ``hermes config set web.router.browser_only_domains '["taobao.com"]'``
    ends up with the *string* ``'["taobao.com"]'`` in config.yaml rather than
    a list. Accept both shapes defensively: a real list/tuple, a JSON array
    string, or a comma/whitespace separated string. Never raises.
    """
    if raw is None:
        return DEFAULT_BROWSER_ONLY_DOMAINS
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(str(x).strip() for x in parsed if str(x).strip())
            except Exception:  # noqa: BLE001 — fall through to split parsing
                pass
        return tuple(
            part.strip()
            for part in re.split(r"[\s,;]+", text)
            if part.strip()
        )
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    return DEFAULT_BROWSER_ONLY_DOMAINS


# ---------------------------------------------------------------------------
# Decision objects
# ---------------------------------------------------------------------------


@dataclass
class CapabilityPolicyDecision:
    """Layer A output — which capability the Agent should use.

    Produced by the Capability Policy (candidate skill ``web-capability-policy``
    in B1) or by future code; defined here as the canonical serializable
    schema. B1 does NOT implement Layer-A execution.
    """

    policy_version: str = "v0.1"
    web_needed: bool = True
    capability: str = Capability.SEARCH.value
    reason_code: str = "intent_classified"
    recommended_tool: str = "web_search"
    intent_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SearchRouterDecision:
    """Layer B output for a web_search call (R1 §12.2)."""

    decision_version: str = "v0.1"
    selected_provider: Optional[str] = None
    selection_reason: str = ""
    provider_override: Optional[str] = None
    fallback_provider_advisory: Optional[str] = None  # advisory ONLY in B1

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ExtractRouterDecision:
    """Layer B output for a web_extract call (R1 §12.3)."""

    decision_version: str = "v0.1"
    selected_provider: Optional[str] = None
    selection_reason: str = ""
    escalation_recommended: bool = False
    escalation_tool: str = "browser_navigate"
    escalation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Provider availability (runtime facts)
# ---------------------------------------------------------------------------


def resolve_firecrawl_deployment() -> str:
    """Return ``cloud`` or ``self_hosted`` for Firecrawl (R1 §8).

    Cloud is authoritative when FIRECRAWL_API_KEY is set (the managed gateway
    path is also cloud). Self-hosted requires FIRECRAWL_API_URL. Never prints
    or returns secret values.
    """
    try:
        from hermes_cli.config import get_env_value
    except Exception:  # noqa: BLE001 — env resolution is best-effort
        get_env_value = None

    def _env(name: str) -> str:
        if get_env_value is not None:
            try:
                val = get_env_value(name)
                if val:
                    return str(val).strip()
            except Exception:  # noqa: BLE001
                pass
        import os

        return (os.getenv(name) or "").strip()

    api_key = _env("FIRECRAWL_API_KEY")
    api_url = _env("FIRECRAWL_API_URL")
    if api_url and not api_key:
        return "self_hosted"
    return "cloud"


def provider_supports(provider_name: str, capability: str,
                      registry_getter: Optional[Callable[[str], Any]] = None) -> bool:
    """Return True when *provider_name* is registered and supports *capability*.

    ``registry_getter`` mirrors ``agent.web_search_registry.get_provider``;
    injected for tests. Defaults to the real registry. This is a pure
    availability *filter* — it never executes a search or extraction.
    """
    if registry_getter is None:
        try:
            from agent.web_search_registry import get_provider as _get
        except Exception:  # noqa: BLE001 — registry unavailable => provider unknown
            return False
        registry_getter = _get
    try:
        provider = registry_getter(provider_name)
    except Exception:  # noqa: BLE001
        return False
    if provider is None:
        return False
    try:
        if capability == "search":
            return bool(provider.supports_search())
        if capability == "extract":
            return bool(provider.supports_extract())
    except Exception:  # noqa: BLE001 — broken provider is treated as unsupported
        return False
    return False


def provider_dependency_ready(provider_name: str,
                              registry_getter: Optional[Callable[[str], Any]] = None) -> bool:
    """Return True when the provider's local runtime dependency is importable.

    Delegates to the provider's ``is_available()``, which per the
    WebSearchProvider ABC contract covers "optional Python dep importable"
    (parallel → ``parallel`` SDK, exa → ``exa_py``, ddgs → ``ddgs`` package;
    tavily → httpx direct, no extra SDK). Cheap and offline by contract.

    This is the *dependency* half of runtime readiness; the credential half
    is :func:`provider_credential_present`. The post-B1 readiness audit
    proved key-only checks can report "available" for providers whose SDK is
    missing, so Router selection must require both.
    """
    if registry_getter is None:
        try:
            from agent.web_search_registry import get_provider as _get
        except Exception:  # noqa: BLE001
            return False
        registry_getter = _get
    try:
        provider = registry_getter(provider_name)
    except Exception:  # noqa: BLE001
        return False
    if provider is None:
        return False
    try:
        return bool(provider.is_available())
    except Exception:  # noqa: BLE001 — a broken availability probe means "not ready"
        return False


def provider_runtime_ready(
    provider_name: str,
    capability: str,
    registry_getter: Optional[Callable[[str], Any]] = None,
    env_has: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Return True when *provider_name* is fully runtime-ready for Router use.

    Requires ALL of (R1 §10 / post-B1 correction §5):
      - registered + requested capability supported
        (:func:`provider_supports`);
      - required credential present, or provider is credential-free
        (:func:`provider_credential_present`);
      - required local dependency importable
        (:func:`provider_dependency_ready` → provider ``is_available()``).

    Plugin enablement is already reflected by registration (disabled bundled
    plugins never register). This is the single readiness gate the Router
    uses for selection and substitute decisions.
    """
    if not provider_supports(provider_name, capability, registry_getter):
        return False
    if not provider_credential_present(provider_name, env_has):
        return False
    return provider_dependency_ready(provider_name, registry_getter)


def provider_credential_present(provider_name: str,
                                env_has: Optional[Callable[[str], bool]] = None) -> bool:
    """Return True when the provider's credential is present (or it needs none).

    Mirrors the cheap hardcoded probes in ``tools.web_tools._is_backend_available``.
    ddgs needs no credential (local package). env_has injected for tests.
    """
    if provider_name == "ddgs":
        return True  # credential-free
    if env_has is None:
        try:
            from hermes_cli.config import get_env_value as _gev

            def _has(name: str) -> bool:
                try:
                    return bool((_gev(name) or "").strip())
                except Exception:  # noqa: BLE001
                    import os

                    return bool((os.getenv(name) or "").strip())

        except Exception:  # noqa: BLE001
            import os

            def _has(name: str) -> bool:
                return bool((os.getenv(name) or "").strip())

        env_has = _has
    key_map = {
        "parallel": "PARALLEL_API_KEY",
        "exa": "EXA_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "firecrawl": "FIRECRAWL_API_KEY",
    }
    key = key_map.get(provider_name)
    if not key:
        return False
    if env_has(key):
        return True
    # Firecrawl may also be served by the managed tool gateway (no key).
    if provider_name == "firecrawl":
        try:
            from tools.web_tools import _is_tool_gateway_ready

            return bool(_is_tool_gateway_ready())
        except Exception:  # noqa: BLE001
            return False
    return False


def select_substitute_provider(
    capability: str,
    exclude: Optional[str] = None,
    registry_getter: Optional[Callable[[str], Any]] = None,
    env_has: Optional[Callable[[str], bool]] = None,
    enabled_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Return ONE deterministic compatible provider as a substitute.

    Used only when the intent-preferred provider is unavailable, and only
    during decision construction (R1 §10). B1 never executes a second
    provider after a failure. ``enabled_names`` lets tests restrict the
    candidate pool without touching the real registry.
    """
    candidates = PROVIDER_SUBSTITUTE_ORDER
    if enabled_names is not None:
        candidates = tuple(n for n in candidates if n in set(enabled_names))
    for name in candidates:
        if exclude and name == exclude:
            continue
        if not provider_runtime_ready(name, capability, registry_getter, env_has):
            continue
        return name
    return None


# ---------------------------------------------------------------------------
# URL boundary classification (R1 §11 — extract side)
# ---------------------------------------------------------------------------

_LOGIN_PATH_RE = re.compile(
    r"/(?:login|signin|sign-in|auth|authenticate|oauth|sso)(?:/|$)",
    re.IGNORECASE,
)

#: path segments indicating cart / favorites / purchase / checkout / forms
_INTERACTION_PATH_RE = re.compile(
    r"/(?:cart|favorite|favorites|checkout|purchase|buy|order|pay|payment"
    r"|add-to-cart|wishlist|submit|form)(?:/|$)",
    re.IGNORECASE,
)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_browser_only_host(url: str, browser_only_domains: Sequence[str]) -> bool:
    """Safe domain-suffix match: ``taobao.com`` matches ``taobao.com`` and
    ``item.taobao.com`` but NOT ``evil-taobao.com`` (R1 §11)."""
    host = _host_of(url)
    if not host:
        return False
    for domain in browser_only_domains or ():
        domain = (domain or "").strip().lower().lstrip(".")
        if not domain:
            continue
        if host == domain or host.endswith("." + domain):
            return True
    return False


def url_requires_browser(url: str, browser_only_domains: Sequence[str]) -> bool:
    """Return True when *url* must go to Browser, never an external extractor.

    Checks, in order: browser-only host suffixes, login/auth paths,
    interaction paths (cart/favorites/purchase/checkout/forms). Sensitive
    query-parameter URLs are already blocked earlier in web_extract_tool by
    the existing ``sensitive_query_param_name`` guard; the router also
    consults it here so the decision object is self-contained (R1 §11).
    """
    if is_browser_only_host(url, browser_only_domains):
        return True
    try:
        path = urlparse(url).path
    except ValueError:
        path = ""
    if _LOGIN_PATH_RE.search(path):
        return True
    if _INTERACTION_PATH_RE.search(path):
        return True
    try:
        from tools.url_safety import sensitive_query_param_name

        if sensitive_query_param_name(url):
            return True
    except Exception:  # noqa: BLE001 — guard failure must not leak the URL out
        return True
    return False


def browser_escalation_reason(url: str, browser_only_domains: Sequence[str]) -> str:
    """Human/agent-readable reason for a Browser escalation recommendation."""
    if is_browser_only_host(url, browser_only_domains):
        return "browser_only_domain"
    try:
        path = urlparse(url).path
    except ValueError:
        path = ""
    if _LOGIN_PATH_RE.search(path):
        return "login_required"
    if _INTERACTION_PATH_RE.search(path):
        return "interaction_required"
    try:
        from tools.url_safety import sensitive_query_param_name

        if sensitive_query_param_name(url):
            return "sensitive_query_parameter"
    except Exception:  # noqa: BLE001
        pass
    return "browser_required"
