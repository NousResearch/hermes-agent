"""Regression coverage for quota-aware web backend fallback chains."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from agent.web_search_provider import WebSearchProvider
from agent import web_search_registry


class FakeWebProvider(WebSearchProvider):
    def __init__(
        self,
        name: str,
        *,
        search_response: dict[str, Any] | Exception | None = None,
        extract_response: list[dict[str, Any]] | Exception | None = None,
        available: bool = True,
    ) -> None:
        self._name = name
        self.search_response = search_response
        self.extract_response = extract_response
        self.available = available
        self.search_calls: list[tuple[str, int]] = []
        self.extract_calls: list[tuple[list[str], dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self.available

    def supports_search(self) -> bool:
        return self.search_response is not None

    def supports_extract(self) -> bool:
        return self.extract_response is not None

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        self.search_calls.append((query, limit))
        if isinstance(self.search_response, Exception):
            raise self.search_response
        assert self.search_response is not None
        return self.search_response

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.extract_calls.append((urls, kwargs))
        if isinstance(self.extract_response, Exception):
            raise self.extract_response
        assert self.extract_response is not None
        return self.extract_response


def _write_user_chain_plugin(hermes_home: Path) -> None:
    """Install a deterministic user web plugin for subprocess E2E tests."""
    plugin_dir = hermes_home / "plugins" / "web" / "test_chain"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: web-test-chain\n"
        "version: 1.0.0\n"
        "kind: backend\n"
        "provides_web_providers:\n"
        "  - discovered-primary\n"
        "  - discovered-fallback\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            from agent.web_search_provider import WebSearchProvider


            class DiscoveredProvider(WebSearchProvider):
                def __init__(self, name, succeeds):
                    self._name = name
                    self._succeeds = succeeds

                @property
                def name(self):
                    return self._name

                def is_available(self):
                    return True

                def search(self, query, limit=5):
                    if not self._succeeds:
                        return {"success": False, "error": "primary exhausted"}
                    return {
                        "success": True,
                        "data": {
                            "web": [{
                                "title": "discovered fallback",
                                "url": "https://example.test/discovered",
                                "description": query,
                                "position": 1,
                            }]
                        },
                    }


            def register(ctx):
                ctx.register_web_search_provider(
                    DiscoveredProvider("discovered-primary", False)
                )
                ctx.register_web_search_provider(
                    DiscoveredProvider("discovered-fallback", True)
                )
            """
        ),
        encoding="utf-8",
    )


def _run_discovered_chain_subprocess(hermes_home: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_BUNDLED_PLUGINS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tools.web_tools import web_search_tool; "
                "print(web_search_tool('real discovery path', limit=1))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


@pytest.fixture(autouse=True)
def reset_registry():
    web_search_registry._reset_for_tests()
    yield
    web_search_registry._reset_for_tests()


def test_parse_backend_chain_accepts_lists_and_comma_strings():
    import tools.web_tools as wt

    assert wt._parse_backend_chain(["Primary", "secondary", "exa", "unknown", "exa"]) == [
        "primary",
        "secondary",
        "exa",
        "unknown",
    ]
    assert wt._parse_backend_chain("primary, secondary exa parallel  tavily") == [
        "primary",
        "secondary",
        "exa",
        "parallel",
        "tavily",
    ]


def test_web_search_falls_back_to_next_provider_on_error(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider(
        "brave-free",
        search_response={"success": False, "error": "Brave Search returned HTTP 429"},
    )
    exa = FakeWebProvider(
        "exa",
        search_response={
            "success": True,
            "data": {"web": [{"title": "Exa hit", "url": "https://example.com", "description": "ok"}]},
        },
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["brave-free", "exa"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("agent search", limit=3))

    assert result["success"] is True
    assert result["data"]["web"][0]["title"] == "Exa hit"
    assert brave.search_calls == [("agent search", 3)]
    assert exa.search_calls == [("agent search", 3)]


def test_web_search_falls_back_after_provider_exception(monkeypatch):
    import tools.web_tools as wt

    class RaisingProvider(FakeWebProvider):
        def search(self, query: str, limit: int = 5):
            self.search_calls.append((query, limit))
            raise RuntimeError("primary exploded")

    primary = RaisingProvider(
        "primary",
        search_response={"success": False, "error": "placeholder"},
    )
    secondary = FakeWebProvider(
        "secondary",
        search_response={
            "success": True,
            "data": {"web": [{"title": "Recovered", "url": "https://example.com"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(secondary)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["primary", "secondary"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("agent search", limit=3))

    assert result["success"] is True
    assert result["data"]["web"][0]["title"] == "Recovered"
    assert primary.search_calls == [("agent search", 3)]
    assert secondary.search_calls == [("agent search", 3)]


def test_web_search_does_not_spend_fallback_on_clean_empty_by_default(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider("brave-free", search_response={"success": True, "data": {"web": []}})
    exa = FakeWebProvider(
        "exa",
        search_response={"success": True, "data": {"web": [{"title": "would cost", "url": "https://example.com"}]}},
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["brave-free", "exa"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("definitely empty", limit=2))

    assert result == {"success": True, "data": {"web": []}}
    assert brave.search_calls == [("definitely empty", 2)]
    assert exa.search_calls == []


def test_web_search_can_opt_in_to_fallback_on_clean_empty(monkeypatch):
    import tools.web_tools as wt

    primary = FakeWebProvider(
        "primary",
        search_response={"success": True, "data": {"web": []}},
    )
    secondary = FakeWebProvider(
        "secondary",
        search_response={
            "success": True,
            "data": {"web": [{"title": "found", "url": "https://example.com"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(secondary)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {
            "search_backends": ["primary", "secondary"],
            "fallback_on_empty_search": True,
        },
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("empty then recover", limit=2))

    assert result["data"]["web"][0]["title"] == "found"
    assert primary.search_calls == [("empty then recover", 2)]
    assert secondary.search_calls == [("empty then recover", 2)]


def test_web_extract_skips_search_only_and_falls_back_to_exa(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider("brave-free", search_response={"success": True, "data": {"web": []}})
    firecrawl = FakeWebProvider(
        "firecrawl",
        extract_response=[{"url": "https://example.com", "title": "", "content": "", "error": "quota exhausted"}],
    )
    exa = FakeWebProvider(
        "exa",
        extract_response=[{"url": "https://example.com", "title": "Example", "content": "clean markdown"}],
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(firecrawl)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["brave-free", "firecrawl", "exa"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool(["https://example.com"])))

    assert result["results"][0]["title"] == "Example"
    assert result["results"][0]["content"] == "clean markdown"
    assert brave.extract_calls == []
    assert firecrawl.extract_calls
    assert exa.extract_calls


def test_xai_can_remain_first_in_search_chain(monkeypatch):
    import tools.web_tools as wt

    xai = FakeWebProvider(
        "xai",
        search_response={"success": True, "data": {"web": [{"title": "Grok hit", "url": "https://x.ai"}]}},
    )
    brave = FakeWebProvider(
        "brave-free",
        search_response={"success": True, "data": {"web": [{"title": "Brave hit", "url": "https://brave.com"}]}},
    )
    web_search_registry.register_provider(xai)
    web_search_registry.register_provider(brave)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["xai", "brave-free"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("X post context", limit=1))

    assert result["data"]["web"][0]["title"] == "Grok hit"
    assert xai.search_calls == [("X post context", 1)]
    assert brave.search_calls == []


def test_default_config_keeps_fallback_chains_opt_in():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["web"]["search_backends"] == []
    assert DEFAULT_CONFIG["web"]["extract_backends"] == []
    assert DEFAULT_CONFIG["web"]["fallback_on_empty_search"] is False


def test_parse_backend_chain_preserves_unknown_names_for_diagnostics():
    import tools.web_tools as wt

    assert wt._parse_backend_chain(["not-installed", "EXA", "not-installed"]) == [
        "not-installed",
        "exa",
    ]


def test_unknown_explicit_chain_does_not_fall_back_to_legacy_auto(monkeypatch):
    import tools.web_tools as wt

    exa = FakeWebProvider(
        "exa",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not run", "url": "https://example.com"}]},
        },
    )
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["typo-provider"]})
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "exa")

    result = json.loads(wt.web_search_tool("policy boundary"))

    assert result["success"] is False
    assert "typo-provider" in result["error"]
    assert "not registered" in result["error"]
    assert exa.search_calls == []


def test_malformed_explicit_chain_fails_closed(monkeypatch):
    import tools.web_tools as wt

    exa = FakeWebProvider(
        "exa",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not run", "url": "https://example.com"}]},
        },
    )
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": {"exa": 1}})
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "exa")

    result = json.loads(wt.web_search_tool("invalid policy"))

    assert "error" in result
    assert "search_backends" in result["error"]
    assert "list" in result["error"].lower()
    assert exa.search_calls == []


def test_extract_fallback_preserves_successes_and_retries_only_failed_urls(monkeypatch):
    import tools.web_tools as wt

    first_url = "https://example.com/first"
    second_url = "https://example.com/second"

    primary = FakeWebProvider(
        "primary",
        extract_response=[
            # Parallel/Tavily group successes before failures, so this order
            # intentionally differs from the input URL order.
            {"url": second_url, "title": "Second", "content": "primary content"},
            {"url": first_url, "title": "", "content": "", "error": "quota exhausted"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": first_url, "title": "First", "content": "fallback content"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([first_url, second_url])))

    assert [item["content"] for item in result["results"]] == [
        "fallback content",
        "primary content",
    ]
    assert primary.extract_calls == [([first_url, second_url], {"format": None})]
    assert fallback.extract_calls == [([first_url], {"format": None})]


def test_scalar_search_preserves_provider_failure_response(monkeypatch):
    import tools.web_tools as wt

    provider_response = {"success": False, "error": "primary quota exhausted"}
    primary = FakeWebProvider("primary", search_response=provider_response)
    web_search_registry.register_provider(primary)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backend": "primary"})

    assert json.loads(wt.web_search_tool("legacy scalar")) == provider_response
    assert primary.search_calls == [("legacy scalar", 5)]


def test_scalar_capability_mismatch_preserves_active_provider_walk(monkeypatch):
    import tools.web_tools as wt

    extract_only = FakeWebProvider(
        "extract-only",
        extract_response=[{"url": "https://example.test", "content": "extract"}],
    )
    search_fallback = FakeWebProvider(
        "search-fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "legacy walk", "url": "https://example.test"}]},
        },
    )
    web_search_registry.register_provider(extract_only)
    web_search_registry.register_provider(search_fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backend": "extract-only"},
    )
    monkeypatch.setattr(
        web_search_registry,
        "get_active_search_provider",
        lambda: search_fallback,
    )

    result = json.loads(wt.web_search_tool("scalar capability mismatch"))

    assert result["data"]["web"][0]["title"] == "legacy walk"
    assert search_fallback.search_calls == [("scalar capability mismatch", 5)]


def test_each_search_call_restarts_at_primary(monkeypatch):
    import tools.web_tools as wt

    class RecoveringPrimary(FakeWebProvider):
        def search(self, query, limit=5):
            self.search_calls.append((query, limit))
            if len(self.search_calls) == 1:
                return {"success": False, "error": "temporary quota"}
            return {
                "success": True,
                "data": {"web": [{"title": "primary recovered", "url": "https://primary.test"}]},
            }

    primary = RecoveringPrimary(
        "primary",
        search_response={"success": False, "error": "placeholder"},
    )
    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "fallback", "url": "https://fallback.test"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    first = json.loads(wt.web_search_tool("first"))
    second = json.loads(wt.web_search_tool("second"))

    assert first["data"]["web"][0]["title"] == "fallback"
    assert second["data"]["web"][0]["title"] == "primary recovered"
    assert fallback.search_calls == [("first", 5)]


def test_check_web_api_key_keeps_global_availability_with_malformed_chain(monkeypatch):
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": {"exa": 1}})
    monkeypatch.setattr(wt, "_is_backend_available", lambda name: name == "exa")

    assert wt.check_web_api_key() is True


def test_check_web_api_key_keeps_global_availability_with_unknown_scalar(monkeypatch):
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backend": "typo-provider"})
    monkeypatch.setattr(wt, "_is_backend_available", lambda name: name == "exa")

    assert wt.check_web_api_key() is True


def test_unavailable_search_chain_does_not_hide_active_extract_provider(monkeypatch):
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["unavailable"]})
    monkeypatch.setattr(wt, "_is_backend_available", lambda name: False)
    monkeypatch.setattr(web_search_registry, "get_active_search_provider", lambda: None)
    monkeypatch.setattr(web_search_registry, "get_active_extract_provider", object)

    assert wt.check_web_api_key() is True


@pytest.mark.parametrize("chain_key", ["search_backends", "extract_backends"])
def test_explicit_nonlegacy_chain_exposes_web_schemas(monkeypatch, chain_key):
    import tools.web_tools as wt
    from tools.registry import invalidate_check_fn_cache, registry

    for name in ("custom-primary", "custom-fallback"):
        web_search_registry.register_provider(
            FakeWebProvider(
                name,
                search_response={"success": True, "data": {"web": []}},
                extract_response=[],
            )
        )

    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {chain_key: ["custom-primary", "custom-fallback"]},
    )
    monkeypatch.setattr(wt, "_is_backend_available", lambda name: False)

    # Two available non-legacy providers do not activate the scalar resolver;
    # schema exposure must therefore come from the explicit chain itself.
    assert web_search_registry.get_active_search_provider() is None
    assert web_search_registry.get_active_extract_provider() is None

    invalidate_check_fn_cache()
    try:
        definitions = registry.get_definitions(
            {"web_search", "web_extract"},
            quiet=True,
        )
    finally:
        invalidate_check_fn_cache()

    assert {definition["function"]["name"] for definition in definitions} == {
        "web_search",
        "web_extract",
    }


def test_temp_hermes_home_config_drives_real_chain_resolution(tmp_path, monkeypatch):
    import hermes_cli.config as hermes_config
    import tools.web_tools as wt

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "web:\n"
        "  search_backends:\n"
        "    - primary\n"
        "    - fallback\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hermes_config, "get_hermes_home", lambda: hermes_home)

    primary = FakeWebProvider(
        "primary",
        search_response={"success": False, "error": "quota exhausted"},
    )
    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "from real config", "url": "https://example.com"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)

    assert wt._load_web_config()["search_backends"] == ["primary", "fallback"]
    result = json.loads(wt.web_search_tool("config propagation"))

    assert result["data"]["web"][0]["title"] == "from real config"
    assert primary.search_calls == [("config propagation", 5)]
    assert fallback.search_calls == [("config propagation", 5)]


def test_real_temp_home_discovers_and_dispatches_user_plugin_chain(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _write_user_chain_plugin(hermes_home)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - web/test_chain\n"
        "web:\n"
        "  search_backends:\n"
        "    - discovered-primary\n"
        "    - discovered-fallback\n",
        encoding="utf-8",
    )

    result = _run_discovered_chain_subprocess(hermes_home)

    assert result["success"] is True
    assert result["data"]["web"] == [
        {
            "title": "discovered fallback",
            "url": "https://example.test/discovered",
            "description": "real discovery path",
            "position": 1,
        }
    ]


def test_plural_chain_reports_real_disabled_bundled_plugin(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n"
        "  disabled:\n"
        "    - web/brave_free\n"
        "web:\n"
        "  search_backends:\n"
        "    - brave-free\n",
        encoding="utf-8",
    )

    result = _run_discovered_chain_subprocess(hermes_home)

    assert result["success"] is False
    assert "web.search_backends includes 'brave-free'" in result["error"]
    assert "plugin ('web/brave_free') is disabled" in result["error"]
    assert "hermes plugins enable web/brave_free" in result["error"]


def test_search_interruption_is_terminal_for_chain(monkeypatch):
    import tools.web_tools as wt

    primary = FakeWebProvider(
        "primary",
        search_response={"success": False, "error": "Interrupted"},
    )
    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not run", "url": "https://fallback.test"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("stop"))

    assert result == {"success": False, "error": "Interrupted"}
    assert primary.search_calls == [("stop", 5)]
    assert fallback.search_calls == []


def test_extract_policy_block_is_terminal_per_url(monkeypatch):
    import tools.web_tools as wt

    blocked_url = "https://public.example/redirects-private"
    retry_url = "https://public.example/transient"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {
                "url": "http://127.0.0.1/private",
                "title": "",
                "content": "",
                "error": "Blocked: URL targets a private or internal network address",
            },
            {"url": retry_url, "title": "", "content": "", "error": "HTTP 503"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": retry_url, "title": "retry", "content": "recovered"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    async def _safe_url(_url):
        return True

    monkeypatch.setattr(wt, "async_is_safe_url", _safe_url)

    result = json.loads(asyncio.run(wt.web_extract_tool([blocked_url, retry_url])))

    assert result["results"][0]["error"] == (
        "Blocked: URL targets a private or internal network address"
    )
    assert result["results"][1]["content"] == "recovered"
    assert primary.extract_calls == [([blocked_url, retry_url], {"format": None})]
    assert fallback.extract_calls == [([retry_url], {"format": None})]


def test_availability_exception_skips_to_configured_fallback(monkeypatch):
    import tools.web_tools as wt

    class BrokenAvailability(FakeWebProvider):
        def is_available(self):
            raise RuntimeError("credential probe exploded")

    primary = BrokenAvailability(
        "primary",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not run", "url": "https://primary.test"}]},
        },
    )
    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "fallback", "url": "https://fallback.test"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["primary", "fallback"]},
    )

    result = json.loads(wt.web_search_tool("availability"))

    assert result["data"]["web"][0]["title"] == "fallback"
    assert primary.search_calls == []
    assert fallback.search_calls == [("availability", 5)]


def test_unknown_entry_skips_only_to_later_configured_provider(monkeypatch):
    import tools.web_tools as wt

    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "configured", "url": "https://fallback.test"}]},
        },
    )
    auto = FakeWebProvider(
        "auto",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not auto", "url": "https://auto.test"}]},
        },
    )
    web_search_registry.register_provider(fallback)
    web_search_registry.register_provider(auto)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["typo", "fallback"]},
    )
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "auto")

    result = json.loads(wt.web_search_tool("configured only"))

    assert result["data"]["web"][0]["title"] == "configured"
    assert fallback.search_calls == [("configured only", 5)]
    assert auto.search_calls == []


def test_extract_fallback_correlates_omitted_rows_by_url(monkeypatch):
    import tools.web_tools as wt

    first_url = "https://example.com/omitted"
    second_url = "https://example.com/success"
    primary = FakeWebProvider(
        "primary",
        # Exa may return only successful rows.
        extract_response=[
            {"url": second_url, "title": "Second", "content": "primary success"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": first_url, "title": "First", "content": "fallback success"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([first_url, second_url])))

    assert [item["content"] for item in result["results"]] == [
        "fallback success",
        "primary success",
    ]
    assert fallback.extract_calls == [([first_url], {"format": None})]


def test_extract_correlation_preserves_duplicate_input_occurrences(monkeypatch):
    import tools.web_tools as wt

    url = "https://example.com/duplicate"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {"url": url, "title": "first", "content": "first occurrence"},
            {"url": url, "title": "", "content": "", "error": "HTTP 503"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": url, "title": "second", "content": "second occurrence"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda value: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([url, url])))

    assert [item["content"] for item in result["results"]] == [
        "first occurrence",
        "second occurrence",
    ]
    assert fallback.extract_calls == [([url], {"format": None})]


def test_multiple_unmatched_redirect_rows_are_not_positionally_guessed(monkeypatch):
    import tools.web_tools as wt

    first_url = "https://example.com/redirect-one"
    second_url = "https://example.com/redirect-two"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {"url": "https://final.example/two", "title": "", "content": "ambiguous"},
            {
                "url": "https://final.example/one",
                "title": "",
                "content": "",
                "error": "HTTP 503",
            },
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": first_url, "title": "one", "content": "fallback one"},
            {"url": second_url, "title": "two", "content": "fallback two"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda value: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([first_url, second_url])))

    assert [item["content"] for item in result["results"]] == [
        "fallback one",
        "fallback two",
    ]
    assert fallback.extract_calls == [([first_url, second_url], {"format": None})]


def test_uncorrelated_redirect_policy_result_stops_all_fallback(monkeypatch):
    import tools.web_tools as wt

    first_url = "https://example.com/redirect-one"
    second_url = "https://example.com/redirect-two"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {
                "url": "http://127.0.0.1/private",
                "title": "",
                "content": "",
                "error": "Blocked: URL targets a private or internal network address",
            },
            {"url": "https://final.example/two", "title": "", "content": "ambiguous"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": first_url, "title": "must not run", "content": "must not run"},
            {"url": second_url, "title": "must not run", "content": "must not run"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda value: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([first_url, second_url])))

    assert all(
        item["error"].startswith("Blocked: extract provider returned")
        for item in result["results"]
    )
    assert fallback.extract_calls == []


def test_one_interrupted_extract_row_terminates_whole_batch(monkeypatch):
    import tools.web_tools as wt

    first_url = "https://example.com/transient"
    second_url = "https://example.com/interrupted"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {"url": first_url, "title": "", "content": "", "error": "HTTP 503"},
            {"url": second_url, "title": "", "content": "", "error": "Interrupted"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": first_url, "title": "must not run", "content": "must not run"},
            {"url": second_url, "title": "must not run", "content": "must not run"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([first_url, second_url])))

    assert [item["error"] for item in result["results"]] == ["HTTP 503", "Interrupted"]
    assert fallback.extract_calls == []


def test_surplus_interrupted_extract_row_terminates_before_alignment(monkeypatch):
    import tools.web_tools as wt

    url = "https://example.com/duplicate-provider-row"
    primary = FakeWebProvider(
        "primary",
        extract_response=[
            {"url": url, "title": "", "content": "", "error": "HTTP 503"},
            {"url": url, "title": "", "content": "", "error": "Interrupted"},
        ],
    )
    fallback = FakeWebProvider(
        "fallback",
        extract_response=[
            {"url": url, "title": "must not run", "content": "must not run"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda value: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([url])))

    assert result["results"][0]["error"] == "Interrupted"
    assert fallback.extract_calls == []


def test_post_search_interrupt_flag_stops_fallback(monkeypatch):
    import tools.interrupt as interrupt
    import tools.web_tools as wt

    state = {"interrupted": False}

    class InterruptingProvider(FakeWebProvider):
        def search(self, query: str, limit: int = 5):
            self.search_calls.append((query, limit))
            state["interrupted"] = True
            return {"success": False, "error": "transient provider error"}

    primary = InterruptingProvider(
        "primary",
        search_response={"success": False, "error": "placeholder"},
    )
    fallback = FakeWebProvider(
        "fallback",
        search_response={
            "success": True,
            "data": {"web": [{"title": "must not run", "url": "https://fallback.test"}]},
        },
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(fallback)

    monkeypatch.setattr(interrupt, "is_interrupted", lambda: state["interrupted"])
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["primary", "fallback"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("stop after primary"))

    assert result == {"success": False, "error": "Interrupted"}
    assert fallback.search_calls == []


def test_registered_override_of_builtin_name_controls_availability(monkeypatch):
    import tools.web_tools as wt

    override = FakeWebProvider(
        "exa",
        search_response={
            "success": True,
            "data": {"web": [{"title": "override", "url": "https://override.test"}]},
        },
    )
    web_search_registry.register_provider(override)

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["exa"]})
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: None)

    result = json.loads(wt.web_search_tool("registry override"))

    assert result["data"]["web"][0]["title"] == "override"
    assert override.search_calls == [("registry override", 5)]


def test_malformed_extract_chain_fails_closed(monkeypatch):
    import tools.web_tools as wt

    provider = FakeWebProvider(
        "extractor",
        extract_response=[
            {"url": "https://example.com", "title": "must not run", "content": "must not run"},
        ],
    )
    web_search_registry.register_provider(provider)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"extract_backends": {"extractor": 1}})
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool(["https://example.com"])))

    assert "extract_backends" in result["error"]
    assert provider.extract_calls == []


def test_all_search_providers_failed_reports_each_attempt(monkeypatch):
    import tools.web_tools as wt

    primary = FakeWebProvider(
        "primary",
        search_response={"success": False, "error": "quota"},
    )
    secondary = FakeWebProvider(
        "secondary",
        search_response={"success": False, "error": "timeout"},
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(secondary)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"search_backends": ["primary", "secondary"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("all fail"))

    assert result["success"] is False
    assert "primary: quota" in result["error"]
    assert "secondary: timeout" in result["error"]


def test_non_list_extract_response_falls_through_and_reports_all_failures(monkeypatch):
    import tools.web_tools as wt

    class NonListProvider(FakeWebProvider):
        def extract(self, urls: list[str], **kwargs) -> Any:
            self.extract_calls.append((list(urls), dict(kwargs)))
            return {"error": "wrong envelope"}

    url = "https://example.com"
    primary = NonListProvider("primary", extract_response=[])
    secondary = FakeWebProvider(
        "secondary",
        extract_response=[
            {"url": url, "title": "", "content": "", "error": "timeout"},
        ],
    )
    web_search_registry.register_provider(primary)
    web_search_registry.register_provider(secondary)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["primary", "secondary"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda value: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool([url])))

    assert "primary: returned a non-list response" in result["results"][0]["error"]
    assert "secondary: timeout" in result["results"][0]["error"]
