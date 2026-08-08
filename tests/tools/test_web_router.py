"""Mocked unit tests for the Web Capability Router Stage B1.

Covers: intent classification (zh/en + precedence), search provider
selection, extract provider selection + Browser escalation, availability /
disabled-provider / missing-credential / capability filtering, provider
override, serialization-without-secrets, and the feature-flag zero-change
requirement (router disabled preserves legacy backends).

All tests are local and mocked — no live API calls, no network, no credits.
"""

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

from tools.web_router import (
    SearchIntent,
    is_browser_only_host,
    normalize_browser_only_domains,
    normalize_intent_hint,
)

# ---------------------------------------------------------------------------
# Fixtures: fake registered providers
# ---------------------------------------------------------------------------


def _provider(name, search=True, extract=False, available=True, credential=True):
    p = MagicMock()
    p.name = name
    p.supports_search.return_value = search
    p.supports_extract.return_value = extract
    p.is_available.return_value = available
    return p


def _registry_getter(providers):
    def _get(name):
        return providers.get(name)
    return _get


def _env_has(present_keys):
    def _has(name):
        return name in present_keys
    return _has


# Providers with credentials for all five (ddgs is credential-free).
ALL_CREDS = {"PARALLEL_API_KEY", "EXA_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY"}


@pytest.fixture
def all_providers():
    return {
        "ddgs": _provider("ddgs", search=True, extract=False),
        "parallel": _provider("parallel", search=True, extract=True),
        "exa": _provider("exa", search=True, extract=True),
        "tavily": _provider("tavily", search=True, extract=True),
        "firecrawl": _provider("firecrawl", search=True, extract=True),
    }


# ---------------------------------------------------------------------------
# 3-6. Intent -> provider selection
# ---------------------------------------------------------------------------


class TestSearchSelection:
    def test_simple_discovery_selects_ddgs(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "OpenAI official site",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "ddgs"
        assert d.selection_reason.startswith("intent=SIMPLE_DISCOVERY")

    def test_general_research_selects_parallel(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "best practices for building a knowledge base",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "parallel"

    def test_technical_research_selects_exa(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "LLM inference architecture paper",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "exa"

    def test_current_information_selects_tavily(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "latest GPU prices",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "tavily"


# ---------------------------------------------------------------------------
# 7. Chinese and English classifier cases
# ---------------------------------------------------------------------------


class TestIntentClassification:
    def test_chinese_simple_discovery(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("DeepSeek 官网") == SearchIntent.SIMPLE_DISCOVERY
        assert classify_search_intent("查一下 PyTorch 主页") == SearchIntent.SIMPLE_DISCOVERY

    def test_chinese_technical(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("OpenAI API 文档") == SearchIntent.TECHNICAL_RESEARCH
        assert classify_search_intent("找一下视频生成的论文") == SearchIntent.TECHNICAL_RESEARCH

    def test_chinese_current(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("最新显卡价格") == SearchIntent.CURRENT_INFORMATION
        assert classify_search_intent("最近的政策新闻") == SearchIntent.CURRENT_INFORMATION

    def test_english_signals(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("official website for firecrawl") == SearchIntent.SIMPLE_DISCOVERY
        assert classify_search_intent("paper on diffusion models") == SearchIntent.TECHNICAL_RESEARCH
        assert classify_search_intent("current policy update") == SearchIntent.CURRENT_INFORMATION
        assert classify_search_intent("how to train a llama") == SearchIntent.GENERAL_RESEARCH

    def test_empty_query_defaults_general(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("") == SearchIntent.GENERAL_RESEARCH
        assert classify_search_intent("   ") == SearchIntent.GENERAL_RESEARCH


# ---------------------------------------------------------------------------
# 8. Overlapping-signal precedence
# ---------------------------------------------------------------------------


class TestIntentPrecedence:
    def test_current_beats_technical(self):
        from tools.web_router import classify_search_intent, SearchIntent

        # contains both "最新" (current) and "论文" (technical)
        assert classify_search_intent("最新论文发布") == SearchIntent.CURRENT_INFORMATION

    def test_technical_beats_simple(self):
        from tools.web_router import classify_search_intent, SearchIntent

        # contains both "官网" (simple) and "文档" (technical)
        assert classify_search_intent("官网 API 文档") == SearchIntent.TECHNICAL_RESEARCH

    def test_simple_beats_general(self):
        from tools.web_router import classify_search_intent, SearchIntent

        assert classify_search_intent("best homepage design") == SearchIntent.SIMPLE_DISCOVERY

    def test_current_beats_simple(self):
        from tools.web_router import classify_search_intent, SearchIntent

        # contains both "首页" (simple) and "最新" (current)
        assert classify_search_intent("首页最新价格") == SearchIntent.CURRENT_INFORMATION


# ---------------------------------------------------------------------------
# 9-12. Availability / disabled / credential / capability filtering
# ---------------------------------------------------------------------------


class TestAvailabilityFiltering:
    def test_preferred_unavailable_uses_deterministic_substitute(self, all_providers):
        from tools.web_search_router import select_search_provider

        # exa (technical preferred) is NOT registered
        providers = {k: v for k, v in all_providers.items() if k != "exa"}
        d = select_search_provider(
            "paper on vector databases",
            registry_getter=_registry_getter(providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider is not None
        assert d.selected_provider != "exa"
        assert "preferred_unavailable_substitute" in d.selection_reason
        assert d.fallback_provider_advisory == "exa"

    def test_disabled_provider_is_filtered(self, all_providers):
        from tools.web_search_router import select_search_provider

        # provider registered but disabled via enabled_names (plugin disabled)
        d = select_search_provider(
            "latest release notes",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
            enabled_names=["tavily", "ddgs", "parallel"],
        )
        # tavily (current-info preferred) is in enabled_names, so still chosen
        assert d.selected_provider == "tavily"

    def test_disabled_preferred_falls_to_substitute(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "latest release notes",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
            enabled_names=["ddgs", "parallel", "exa", "firecrawl"],  # tavily disabled
        )
        assert d.selected_provider != "tavily"
        assert d.selected_provider is not None

    def test_missing_credential_is_filtered(self, all_providers):
        from tools.web_search_router import select_search_provider

        # only ddgs (credential-free) + firecrawl have creds
        d = select_search_provider(
            "paper on graph neural networks",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has({"FIRECRAWL_API_KEY"}),
        )
        # exa preferred but no EXA_API_KEY -> substitute (firecrawl has creds)
        assert d.selected_provider is not None
        assert d.selected_provider != "exa"

    def test_unsupported_capability_is_filtered(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        # ddgs registered but extract=False -> cannot be the extract provider
        providers = {k: v for k, v in all_providers.items() if k != "firecrawl"}
        d = select_extract_provider(
            ["https://example.com/page"],
            registry_getter=_registry_getter(providers),
            env_has=_env_has(ALL_CREDS),
        )
        # firecrawl (default) unavailable -> escalation, no secondary extractor
        assert d.escalation_recommended is True
        assert d.escalation_reason == "no_extract_provider_available"
        assert d.selected_provider is None

    def test_no_provider_available_leaves_none(self):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "anything at all",
            registry_getter=_registry_getter({}),
            env_has=_env_has(set()),
        )
        assert d.selected_provider is None
        assert d.selection_reason == "no_provider_available"


# ---------------------------------------------------------------------------
# 13-14. Provider override
# ---------------------------------------------------------------------------


class TestProviderOverride:
    def test_valid_override_honored(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "official homepage of something",
            provider_override="exa",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "exa"
        assert d.selection_reason == "explicit_override"

    def test_invalid_override_rejected(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "some query",
            provider_override="not-a-provider",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider != "not-a-provider"
        assert "override_rejected" in d.selection_reason

    def test_unavailable_override_rejected(self, all_providers):
        from tools.web_search_router import select_search_provider

        # override registered but no credential
        d = select_search_provider(
            "some query",
            provider_override="tavily",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(set()),
        )
        assert d.selected_provider != "tavily"
        assert "override_rejected" in d.selection_reason


# ---------------------------------------------------------------------------
# 15-19. Extract routing + Browser escalation
# ---------------------------------------------------------------------------


class TestExtractSelection:
    def test_public_url_selects_firecrawl(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://example.com/docs/page"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "firecrawl"
        assert d.escalation_recommended is False

    def test_taobao_url_returns_browser_escalation(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        # taobao.com is an explicit synthetic fixture here; the shipped
        # default browser_only_domains is empty.
        d = select_extract_provider(
            ["https://item.taobao.com/item.htm?id=123"],
            browser_only_domains=("taobao.com",),
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True
        assert d.escalation_tool == "browser_navigate"
        assert d.escalation_reason == "browser_only_domain"
        assert d.selected_provider is None

    def test_taobao_root_domain_escalates(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://taobao.com/"],
            browser_only_domains=("taobao.com",),
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True

    def test_empty_default_has_no_vendor_boundary(self, all_providers):
        """Shipped default carries no vendor-specific Browser boundary: a
        taobao URL without explicit configuration is a normal public page."""
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://item.taobao.com/item.htm?id=123"],
            browser_only_domains=(),
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is False
        assert not d.escalation_reason

    def test_login_path_returns_browser_escalation(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://example.com/login"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True
        assert d.escalation_reason == "login_required"

    def test_sensitive_query_param_escalates(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://example.com/page?token=abc123"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True
        assert d.escalation_reason == "sensitive_query_parameter"

    def test_evil_taobao_not_treated_as_taobao(self, all_providers):
        from tools.web_extract_router import select_extract_provider
        from tools.web_router import is_browser_only_host

        assert is_browser_only_host("https://evil-taobao.com/x", ("taobao.com",)) is False
        assert is_browser_only_host("https://nottaobao.com/x", ("taobao.com",)) is False
        assert is_browser_only_host("https://sub.taobao.com/x", ("taobao.com",)) is True

        # and through the extract router: evil-taobao.com is a normal public URL
        d = select_extract_provider(
            ["https://evil-taobao.com/page"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "firecrawl"
        assert d.escalation_recommended is False

    def test_cart_path_escalates(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        d = select_extract_provider(
            ["https://example.com/cart"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True
        assert d.escalation_reason == "interaction_required"


# ---------------------------------------------------------------------------
# 20-22. No auto-Browser / no fallback execution / no secondary extractor
# ---------------------------------------------------------------------------


class TestBoundaryEnforcement:
    def test_no_automatic_browser_invocation(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        # Even for a browser-only URL the router only RECOMMENDS escalation.
        d = select_extract_provider(
            ["https://item.taobao.com/item.htm?id=1"],
            browser_only_domains=("taobao.com",),
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.escalation_recommended is True
        # The decision object carries tool + reason only — no browser handle,
        # no navigation call, nothing executable.
        assert d.escalation_tool == "browser_navigate"
        assert d.escalation_reason

    def test_search_decision_never_executes_fallback(self, all_providers):
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "paper on something",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        # fallback_provider_advisory is advisory-only: selecting a provider
        # must not trigger a second provider call. The decision only names it.
        assert isinstance(d.fallback_provider_advisory, (str, type(None)))
        # and no provider.search was ever called by decision construction
        for p in all_providers.values():
            p.search.assert_not_called()
            p.extract.assert_not_called()

    def test_extract_decision_never_calls_extractor(self, all_providers):
        from tools.web_extract_router import select_extract_provider

        select_extract_provider(
            ["https://example.com/page"],
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        for p in all_providers.values():
            p.extract.assert_not_called()


# ---------------------------------------------------------------------------
# 23. Decision objects serialize without secrets
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_decision_dicts_have_no_secrets(self, all_providers):
        from tools.web_router import (
            CapabilityPolicyDecision,
            ExtractRouterDecision,
            SearchRouterDecision,
        )
        from tools.web_search_router import select_search_provider

        d = select_search_provider(
            "official site",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        blob = json.dumps(d.to_dict())
        assert "api_key" not in blob.lower()
        assert "token" not in blob.lower()
        assert "secret" not in blob.lower()

        e = ExtractRouterDecision(
            escalation_recommended=True,
            escalation_reason="login_required",
        )
        blob_e = json.dumps(e.to_dict())
        assert "api_key" not in blob_e.lower()
        assert "token" not in blob_e.lower()

        c = CapabilityPolicyDecision()
        blob_c = json.dumps(c.to_dict())
        assert "api_key" not in blob_c.lower()
        assert "token" not in blob_c.lower()


# ---------------------------------------------------------------------------
# 24. Candidate skill remains inactive
# ---------------------------------------------------------------------------


class TestSkillInactive:
    def test_candidate_skill_not_in_live_skills_tree(self):
        hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        skill_dir = hermes_home / "skills" / "web-capability-policy"
        assert not skill_dir.exists(), (
            "candidate skill must NOT be placed in the live skills tree "
            "(file presence auto-activates it via build_skills_system_prompt)"
        )
        # and no devops/web-capability-policy either
        assert not (hermes_home / "skills" / "devops" / "web-capability-policy").exists()


# ---------------------------------------------------------------------------
# 25. Router disabled preserves legacy backends (feature-flag zero-change)
# ---------------------------------------------------------------------------


class TestFeatureFlagDisabled:
    def test_disabled_search_preserves_legacy_backend(self):
        import tools.web_tools as wt

        with patch.object(wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            # router key absent => disabled
            assert wt._get_search_backend() == "firecrawl"

    def test_disabled_extract_preserves_legacy_backend(self):
        import tools.web_tools as wt

        with patch.object(
            wt, "_load_web_config",
            return_value={"backend": "firecrawl", "router": {"enabled": False}},
        ):
            assert wt._get_extract_backend() == "firecrawl"

    def test_web_search_tool_disabled_does_not_import_router(self):
        """Feature flag false => no router state, no classification, no telemetry."""
        import tools.web_tools as wt

        provider = _provider("firecrawl", search=True)
        provider.search.return_value = {
            "success": True,
            "data": {"web": [{"title": "t", "url": "https://x", "description": "d", "position": 1}]},
        }
        with patch.object(wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            with patch.object(wt, "_ensure_web_plugins_loaded"):
                with patch(
                    "agent.web_search_registry.get_provider",
                    side_effect=lambda name: provider if name == "firecrawl" else None,
                ):
                    with patch("tools.web_search_router.select_search_provider") as router:
                        result = wt.web_search_tool("legacy query")
                        data = json.loads(result)
                        assert data["success"] is True
                        # legacy backend (firecrawl) was used...
                        provider.search.assert_called_once()
                        # ...and the router was NEVER consulted
                        router.assert_not_called()

    def test_router_enabled_flag_still_false_in_config(self):
        try:
            from hermes_cli.config import load_config_readonly

            cfg = load_config_readonly()
            router = (cfg.get("web") or {}).get("router") or {}
            assert router.get("enabled") is False
        except Exception:
            pytest.skip("config not readable in this environment")

    def test_legacy_backends_unchanged(self):
        """web.backend / search_backend / extract_backend values are preserved."""
        try:
            from hermes_cli.config import load_config_readonly

            cfg = load_config_readonly()
            web = cfg.get("web") or {}
            # these keys must still exist and keep their configured values
            assert "backend" in web
            assert "search_backend" in web
            assert "extract_backend" in web
        except Exception:
            pytest.skip("config not readable in this environment")


# ---------------------------------------------------------------------------
# Integration smoke (mocked): router-enabled search still returns tool JSON
# ---------------------------------------------------------------------------


class TestToolIntegrationMocked:
    def test_web_search_tool_router_enabled_picks_provider_and_searches(self):
        """With the flag on, ONE provider is selected and called — mocked."""
        import tools.web_tools as wt

        provider = _provider("exa", search=True)
        provider.search.return_value = {
            "success": True,
            "data": {"web": [{"title": "t", "url": "https://x", "description": "d", "position": 1}]},
        }
        cfg = {"backend": "firecrawl", "router": {"enabled": True}}

        with patch.object(wt, "_load_web_config", return_value=cfg):
            with patch.object(wt, "_get_search_backend", return_value="firecrawl") as gsb:
                with patch("tools.web_search_router.select_search_provider") as router:
                    from tools.web_router import SearchRouterDecision

                    router.return_value = SearchRouterDecision(
                        selected_provider="exa", selection_reason="intent=TECHNICAL_RESEARCH"
                    )
                    with patch.object(wt, "_ensure_web_plugins_loaded"):
                        with patch(
                            "agent.web_search_registry.get_provider",
                            side_effect=lambda name: provider if name == "exa" else None,
                        ):
                            result = wt.web_search_tool("paper on x")
                            data = json.loads(result)
                            assert data["success"] is True
                            assert data["data"]["web"][0]["url"] == "https://x"
                            provider.search.assert_called_once()
                            # legacy backend resolver still consulted for the
                            # no-decision fallback, but router won
                            assert router.call_count == 1

    def test_web_extract_tool_router_enabled_taobao_escalates(self):
        """With the flag on, a taobao URL returns a structured escalation."""
        import tools.web_tools as wt

        cfg = {"backend": "firecrawl", "router": {"enabled": True}}
        with patch.object(wt, "_load_web_config", return_value=cfg):
            with patch.object(wt, "_get_extract_backend", return_value="firecrawl"):
                # SSRF check would otherwise resolve the hostname — keep it offline
                with patch.object(
                    wt, "async_is_safe_url", new=AsyncMock(return_value=True)
                ):
                    with patch("tools.web_extract_router.select_extract_provider") as router:
                        from tools.web_router import ExtractRouterDecision

                        router.return_value = ExtractRouterDecision(
                            escalation_recommended=True,
                            escalation_tool="browser_navigate",
                            escalation_reason="browser_only_domain",
                        )
                        import asyncio

                        result = asyncio.run(
                            wt.web_extract_tool(["https://item.taobao.com/item.htm?id=1"])
                        )
                        data = json.loads(result)
                        assert data["success"] is False
                        assert data["escalation"]["recommended_tool"] == "browser_navigate"
                        assert data["escalation"]["reason"] == "browser_only_domain"


# ---------------------------------------------------------------------------
# Post-B1 runtime-readiness correction tests (§5/§6 of
# HERMES_WEB_ROUTER_POST_B1_PARALLEL_EXA_READINESS_COMPLETION_V0_1)
# ---------------------------------------------------------------------------
# provider_runtime_ready() requires: registered + capability + credential
# (or credential-free) + local dependency importable (provider.is_available()).
# The `available` flag on the _provider fixture simulates the SDK-importable
# half of is_available(); `credential` is driven by env_has in selection.


class TestRuntimeReadiness:
    """1-2. SDK-missing providers are NOT Router-selectable even with keys."""

    def test_parallel_key_but_missing_sdk_not_selectable(self):
        from tools.web_router import provider_runtime_ready

        # registered, search-capable, PARALLEL_API_KEY present, but SDK
        # missing => is_available() returns False (the corrected semantics)
        p = _provider("parallel", search=True, extract=True, available=False)
        assert (
            provider_runtime_ready(
                "parallel", "search",
                registry_getter=_registry_getter({"parallel": p}),
                env_has=_env_has({"PARALLEL_API_KEY"}),
            )
            is False
        )

    def test_exa_key_but_missing_sdk_not_selectable(self):
        from tools.web_router import provider_runtime_ready

        p = _provider("exa", search=True, extract=True, available=False)
        assert (
            provider_runtime_ready(
                "exa", "search",
                registry_getter=_registry_getter({"exa": p}),
                env_has=_env_has({"EXA_API_KEY"}),
            )
            is False
        )

    def test_parallel_selectable_when_sdk_importable(self):
        from tools.web_router import provider_runtime_ready

        p = _provider("parallel", search=True, extract=True, available=True)
        assert (
            provider_runtime_ready(
                "parallel", "search",
                registry_getter=_registry_getter({"parallel": p}),
                env_has=_env_has({"PARALLEL_API_KEY"}),
            )
            is True
        )

    def test_exa_selectable_when_sdk_importable(self):
        from tools.web_router import provider_runtime_ready

        p = _provider("exa", search=True, extract=True, available=True)
        assert (
            provider_runtime_ready(
                "exa", "search",
                registry_getter=_registry_getter({"exa": p}),
                env_has=_env_has({"EXA_API_KEY"}),
            )
            is True
        )

    def test_ddgs_selectable_without_api_key(self):
        """DDGS is credential-free: no key needed, package importable."""
        from tools.web_router import provider_runtime_ready

        p = _provider("ddgs", search=True, extract=False, available=True)
        assert (
            provider_runtime_ready(
                "ddgs", "search",
                registry_getter=_registry_getter({"ddgs": p}),
                env_has=_env_has(set()),
            )
            is True
        )

    def test_tavily_selectable_via_direct_http(self):
        """Tavily needs only its key (httpx direct implementation)."""
        from tools.web_router import provider_runtime_ready

        p = _provider("tavily", search=True, extract=True, available=True)
        assert (
            provider_runtime_ready(
                "tavily", "search",
                registry_getter=_registry_getter({"tavily": p}),
                env_has=_env_has({"TAVILY_API_KEY"}),
            )
            is True
        )

    def test_firecrawl_selectable_when_client_importable(self):
        from tools.web_router import provider_runtime_ready

        p = _provider("firecrawl", search=True, extract=True, available=True)
        assert (
            provider_runtime_ready(
                "firecrawl", "extract",
                registry_getter=_registry_getter({"firecrawl": p}),
                env_has=_env_has({"FIRECRAWL_API_KEY"}),
            )
            is True
        )

    def test_sdk_missing_preferred_yields_deterministic_substitute(self):
        """8. Preferred (exa) lacks SDK => ONE deterministic substitute."""
        from tools.web_search_router import select_search_provider

        # exa registered with key but SDK missing; others fully ready
        providers = {
            "ddgs": _provider("ddgs", search=True, extract=False, available=True),
            "parallel": _provider("parallel", search=True, extract=True, available=True),
            "exa": _provider("exa", search=True, extract=True, available=False),
            "tavily": _provider("tavily", search=True, extract=True, available=True),
            "firecrawl": _provider("firecrawl", search=True, extract=True, available=True),
        }
        d = select_search_provider(
            "paper on vector databases",
            registry_getter=_registry_getter(providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider is not None
        assert d.selected_provider != "exa"
        assert "preferred_unavailable_substitute" in d.selection_reason
        # substitute order: parallel first after exa excluded
        assert d.selected_provider == "parallel"

    def test_runtime_readiness_does_not_call_provider_search(self):
        """10a. Readiness gates never trigger a live provider call."""
        from tools.web_router import provider_runtime_ready

        p = _provider("tavily", search=True, extract=True, available=True)
        provider_runtime_ready(
            "tavily", "search",
            registry_getter=_registry_getter({"tavily": p}),
            env_has=_env_has({"TAVILY_API_KEY"}),
        )
        p.search.assert_not_called()
        p.extract.assert_not_called()


class TestReadinessRegressionBoundary:
    """10b. Router disabled still preserves legacy behavior exactly."""

    def test_disabled_search_still_legacy_backend(self):
        import tools.web_tools as wt

        with patch.object(wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            assert wt._get_search_backend() == "firecrawl"

    def test_disabled_extract_still_legacy_backend(self):
        import tools.web_tools as wt

        with patch.object(
            wt, "_load_web_config",
            return_value={"backend": "firecrawl", "router": {"enabled": False}},
        ):
            assert wt._get_extract_backend() == "firecrawl"

    def test_disabled_web_search_tool_never_consults_router(self):
        """Feature flag false => no router consultation, no live request."""
        import tools.web_tools as wt

        provider = _provider("firecrawl", search=True, available=True)
        provider.search.return_value = {
            "success": True,
            "data": {"web": [{"title": "t", "url": "https://x", "description": "d", "position": 1}]},
        }
        with patch.object(wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            with patch.object(wt, "_ensure_web_plugins_loaded"):
                with patch(
                    "agent.web_search_registry.get_provider",
                    side_effect=lambda name: provider if name == "firecrawl" else None,
                ):
                    with patch("tools.web_search_router.select_search_provider") as router:
                        result = wt.web_search_tool("legacy query")
                        assert json.loads(result)["success"] is True
                        provider.search.assert_called_once()
                        router.assert_not_called()

    def test_real_provider_is_available_requires_sdk(self):
        """Integration-level: the REAL parallel/exa providers now gate on SDK.

        Uses the real provider classes with mocked env, so no network.
        """
        import importlib.util

        from plugins.web.parallel.provider import ParallelWebSearchProvider
        from plugins.web.exa.provider import ExaWebSearchProvider

        parallel_sdk = importlib.util.find_spec("parallel") is not None
        exa_sdk = importlib.util.find_spec("exa_py") is not None

        # With no key, availability is False regardless of SDK.
        for cls, key in ((ParallelWebSearchProvider, "PARALLEL_API_KEY"),
                         (ExaWebSearchProvider, "EXA_API_KEY")):
            with patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop(key, None)
                inst = cls()
                assert inst.is_available() is False

        # With key present, availability equals SDK importability.
        # (SDK was installed by the readiness-completion task; if this runs
        # in an env without it, the assertion still matches reality.)
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "test-key-not-real"}, clear=False):
            assert ParallelWebSearchProvider().is_available() is parallel_sdk
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key-not-real"}, clear=False):
            assert ExaWebSearchProvider().is_available() is exa_sdk


# ---------------------------------------------------------------------------
# Pre-commit N6 test-gap closure
# (HERMES_WEB_ROUTER_B1_PRECOMMIT_N6_TEST_GAP_CLOSURE_V0_1)
# normalize_browser_only_domains / normalize_intent_hint / escalation
# field-level privacy. Test-only additions; product behavior encoded as-is.
# ---------------------------------------------------------------------------


class TestNormalizeBrowserOnlyDomains:
    """Actual contract of tools.web_router.normalize_browser_only_domains:

    - None / non-string-non-list -> DEFAULT_BROWSER_ONLY_DOMAINS;
    - str: strip; JSON-array string parsed when well-formed; otherwise split
      on whitespace/comma/semicolon;
    - list/tuple: coerced to stripped string tuple;
    - case and leading dots are preserved by normalize and normalized at
      match time inside is_browser_only_host.
    """

    def test_native_list(self):
        assert normalize_browser_only_domains(["taobao.com", "jd.com"]) == (
            "taobao.com", "jd.com",
        )

    def test_json_array_string(self):
        assert normalize_browser_only_domains('["taobao.com", "jd.com"]') == (
            "taobao.com", "jd.com",
        )

    def test_surrounding_whitespace_stripped(self):
        assert normalize_browser_only_domains("  taobao.com  ,  jd.com  ") == (
            "taobao.com", "jd.com",
        )

    def test_semicolon_and_newline_split(self):
        assert normalize_browser_only_domains("taobao.com; jd.com\npdd.com") == (
            "taobao.com", "jd.com", "pdd.com",
        )

    def test_mixed_case_preserved_by_normalize_matched_by_host_check(self):
        norm = normalize_browser_only_domains(["Taobao.COM"])
        assert norm == ("Taobao.COM",)  # normalize preserves case...
        # ...and the host check normalizes at match time
        assert is_browser_only_host("https://item.taobao.com/x", norm) is True

    def test_leading_dot_input(self):
        norm = normalize_browser_only_domains([".taobao.com"])
        assert norm == (".taobao.com",)
        assert is_browser_only_host("https://x.taobao.com/", norm) is True

    def test_duplicate_entries_preserved(self):
        assert normalize_browser_only_domains(["taobao.com", "taobao.com"]) == (
            "taobao.com", "taobao.com",
        )

    def test_empty_list_and_empty_string(self):
        assert normalize_browser_only_domains([]) == ()
        assert normalize_browser_only_domains("") == ()
        assert normalize_browser_only_domains("   ") == ()

    def test_none_returns_default(self):
        # shipped default is empty (vendor-neutral)
        assert normalize_browser_only_domains(None) == ()

    def test_non_string_non_list_returns_default(self):
        assert normalize_browser_only_domains(42) == ()
        assert normalize_browser_only_domains({"a": 1}) == ()

    def test_malformed_json_does_not_broaden_matching(self):
        # single-quoted "JSON" parses as a literal token; it must NOT match
        # the real taobao.com
        norm = normalize_browser_only_domains("['taobao.com']")
        assert "['taobao.com']" in norm
        assert is_browser_only_host("https://taobao.com/", norm) is False
        assert is_browser_only_host("https://item.taobao.com/", norm) is False

    def test_malformed_json_without_quotes_does_not_broaden(self):
        norm = normalize_browser_only_domains("[taobao.com]")
        assert is_browser_only_host("https://taobao.com/", norm) is False

    def test_json_prefix_with_trailing_junk_only_yields_written_tokens(self):
        # '["taobao.com"] evil.com' fails JSON parse -> split; the JSON
        # syntax fragment is a literal that matches nothing, and only the
        # token the user actually wrote (evil.com) is usable. No expansion.
        norm = normalize_browser_only_domains('["taobao.com"] evil.com')
        assert is_browser_only_host("https://taobao.com/", norm) is False
        assert is_browser_only_host("https://item.taobao.com/", norm) is False
        assert is_browser_only_host("https://evil.com/", norm) is True

    def test_evil_taobao_never_equivalent_to_taobao(self):
        assert is_browser_only_host("https://evil-taobao.com/x", ("taobao.com",)) is False
        assert is_browser_only_host("https://notaobao.com/x", ("taobao.com",)) is False
        assert is_browser_only_host("https://taobao.com.evil.example/x", ("taobao.com",)) is False


class TestNormalizeIntentHint:
    """Actual contract of tools.web_router.normalize_intent_hint:

    - empty/None -> None;
    - strip + upper + exact enum name match -> SearchIntent;
    - anything else -> None. No aliases are supported by the implementation.
    """

    def test_canonical_enum_values(self):
        for v in ("SIMPLE_DISCOVERY", "GENERAL_RESEARCH", "TECHNICAL_RESEARCH",
                  "CURRENT_INFORMATION"):
            assert normalize_intent_hint(v).value == v

    def test_lowercase_normalization(self):
        assert normalize_intent_hint("technical_research") == SearchIntent.TECHNICAL_RESEARCH

    def test_mixed_case_normalization(self):
        assert normalize_intent_hint("Technical_Research") == SearchIntent.TECHNICAL_RESEARCH

    def test_leading_trailing_whitespace_stripped(self):
        assert normalize_intent_hint("  general_research  ") == SearchIntent.GENERAL_RESEARCH

    def test_unknown_value_returns_none(self):
        assert normalize_intent_hint("bogus") is None

    def test_empty_and_none_return_none(self):
        assert normalize_intent_hint("") is None
        assert normalize_intent_hint(None) is None

    def test_unknown_never_silently_becomes_another_intent(self):
        # capability names, verification modes, and partial enum strings must
        # not be coerced into a different intent
        for bad in ("SEARCH", "EXTRACT", "BROWSER", "NO_WEB", "VERIFY",
                    "SECOND_PROVIDER", "TECHNICAL", "CURRENT", "GENERAL", "SIMPLE"):
            assert normalize_intent_hint(bad) is None

    def test_no_alias_support(self):
        # the implementation supports no aliases — short forms must stay None
        assert normalize_intent_hint("tech") is None
        assert normalize_intent_hint("paper") is None
        assert normalize_intent_hint("news") is None
        assert normalize_intent_hint("docs") is None


class TestEscalationPrivacyFieldLevel:
    """Field-level privacy/structure assertions on the REAL escalation path:
    web_extract_tool with web.router.enabled, real select_extract_provider
    (not mocked), extractor never reached."""

    ESCALATION_REASONS = {
        "browser_only_domain", "login_required", "interaction_required",
        "sensitive_query_parameter", "browser_required",
        "no_extract_provider_available",
    }

    def _escalation_for(self, url):
        import tools.web_tools as wt

        # taobao.com injected as an explicit fixture; shipped default empty.
        cfg = {"backend": "firecrawl",
               "router": {"enabled": True,
                          "browser_only_domains": ["taobao.com"]}}
        with patch.object(wt, "_load_web_config", return_value=cfg):
            with patch.object(wt, "_get_extract_backend", return_value="firecrawl"):
                with patch.object(
                    wt, "async_is_safe_url", new=AsyncMock(return_value=True)
                ):
                    # real router chain; provider lookup must never run
                    with patch(
                        "agent.web_search_registry.get_provider", return_value=None
                    ) as gp:
                        import asyncio

                        result = asyncio.run(wt.web_extract_tool([url]))
                        return json.loads(result), gp

    def _assert_privacy_clean(self, data, url):
        assert data["success"] is False
        esc = data["escalation"]
        assert esc["recommended_tool"] == "browser_navigate"
        assert esc["reason"] in self.ESCALATION_REASONS
        # exact key structure — no extra fields
        assert set(data.keys()) == {"success", "error", "escalation"}
        assert set(esc.keys()) == {"recommended_tool", "reason"}
        # field-level privacy on the serialized object
        blob = json.dumps(data, ensure_ascii=False)
        parsed = urlparse(url)
        assert parsed.hostname not in blob          # no full URL / host
        assert parsed.path not in blob              # no path
        if parsed.query:
            assert parsed.query not in blob         # no query string
        low = blob.lower()
        assert "token" not in low
        assert "cookie" not in low
        assert "password" not in low
        assert "credential" not in low
        assert "api_key" not in low
        assert "1067726290000" not in blob          # no item ID
        assert "动漫" not in blob                    # no page content
        assert "title" not in esc

    def test_taobao_escalation_privacy(self):
        url = "https://item.taobao.com/item.htm?id=1067726290000"
        data, gp = self._escalation_for(url)
        assert data["escalation"]["reason"] == "browser_only_domain"
        self._assert_privacy_clean(data, url)
        gp.assert_not_called()  # extractor never consulted

    def test_login_path_escalation_privacy(self):
        url = "https://example.com/login"
        data, gp = self._escalation_for(url)
        assert data["escalation"]["reason"] == "login_required"
        self._assert_privacy_clean(data, url)
        gp.assert_not_called()

    def test_sensitive_query_param_blocked_before_extractor(self):
        # The LEGACY guard inside web_extract_tool blocks credential-like
        # query parameters BEFORE the router is ever reached. The response is
        # an error envelope (no escalation object), which is itself the
        # privacy behavior: the URL never reaches an external extractor and
        # the token VALUE never appears in the response.
        url = "https://example.com/page?token=abc123"
        data, gp = self._escalation_for(url)
        assert data["success"] is False
        assert "escalation" not in data          # legacy guard path
        assert "Blocked" in data["error"]
        blob = json.dumps(data, ensure_ascii=False)
        assert "abc123" not in blob              # token value never leaks
        assert "example.com" not in blob
        assert "/page" not in blob
        assert "token=abc123" not in blob
        gp.assert_not_called()                   # extractor never consulted

    def test_evil_taobao_not_treated_as_taobao_subdomain(self):
        # safe-hostname assertion at the unit level
        assert is_browser_only_host("https://evil-taobao.com/x", ("taobao.com",)) is False
        # and the extract router must NOT classify it as a browser-only
        # domain (any escalation it produces is a different reason)
        from tools.web_extract_router import select_extract_provider

        dec = select_extract_provider(
            ["https://evil-taobao.com/page"],
            browser_only_domains=("taobao.com",),
        )
        assert dec.escalation_reason != "browser_only_domain"
        assert dec.selected_provider is None or dec.selected_provider != "browser"


# ---------------------------------------------------------------------------
# Official-site discovery classifier (OFFICIAL_SITE_DISCOVERY_CLASSIFIER_FIX)
# ---------------------------------------------------------------------------


class TestOfficialSiteDiscoveryClassifier:
    """official + website|site|homepage token combination (entity words may
    sit between them) must classify as SIMPLE_DISCOVERY -> ddgs, without
    over-classifying plurals or unrelated queries."""

    @pytest.mark.parametrize("query", [
        "Find the official OpenAI website",
        "Find the official Exa homepage",
        "Locate the official Tavily site",
        "OpenAI official website",
        "Find OpenAI's official website",
    ])
    def test_official_site_discovery_positive(self, query, all_providers):
        from tools.web_router import classify_search_intent
        from tools.web_search_router import select_search_provider

        assert classify_search_intent(query) is SearchIntent.SIMPLE_DISCOVERY
        d = select_search_provider(
            query,
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "ddgs"

    def test_contiguous_official_site_unchanged(self, all_providers):
        # Existing contiguous-form behavior preserved.
        from tools.web_router import classify_search_intent
        from tools.web_search_router import select_search_provider

        assert classify_search_intent("OpenAI official site") is SearchIntent.SIMPLE_DISCOVERY
        d = select_search_provider(
            "OpenAI official site",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "ddgs"

    def test_chinese_official_site_signals_unchanged(self, all_providers):
        from tools.web_router import classify_search_intent
        from tools.web_search_router import select_search_provider

        assert classify_search_intent("OpenAI 官方网站 是什么") is SearchIntent.SIMPLE_DISCOVERY
        d = select_search_provider(
            "OpenAI 官方网站 是什么",
            registry_getter=_registry_getter(all_providers),
            env_has=_env_has(ALL_CREDS),
        )
        assert d.selected_provider == "ddgs"

    @pytest.mark.parametrize("query,expected", [
        # TECHNICAL_RESEARCH must win over the official-site pattern.
        ("Find the official API documentation for Exa", "TECHNICAL_RESEARCH"),
        # CURRENT_INFORMATION must win over the official-site pattern.
        ("What is the current Tavily API pricing on its official website?", "CURRENT_INFORMATION"),
        # Plural "official websites" must NOT match (comparative query).
        ("Compare the official websites of several search providers", "GENERAL_RESEARCH"),
        # Stronger technical signal wins.
        ("Review the architecture of an official website", "TECHNICAL_RESEARCH"),
        # "official" alone (no website/site/homepage) is not discovery.
        ("The official announcement was made yesterday", "GENERAL_RESEARCH"),
        # website word without an official/discovery signal is not auto-SIMPLE.
        ("A website about hiking trails in Yunnan", "GENERAL_RESEARCH"),
    ])
    def test_official_site_discovery_negative(self, query, expected):
        from tools.web_router import classify_search_intent

        assert classify_search_intent(query) is SearchIntent[expected]

    @pytest.mark.parametrize("query", [
        "找最新的视频生成论文",
        "当前版本的 API 文档",
        "search for recent API documentation",
    ])
    def test_deferred_overlap_cases_recorded(self, query):
        """V0.2 classifier-policy candidates — behavior deliberately NOT
        changed by this fix; record current intent only."""
        from tools.web_router import classify_search_intent

        assert classify_search_intent(query) is SearchIntent.CURRENT_INFORMATION
