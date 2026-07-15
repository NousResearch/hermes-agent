from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import yaml


def _provider_registry() -> SimpleNamespace:
    profile = SimpleNamespace(
        name="openai-codex",
        aliases=("codex", "openai_codex"),
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        auth_type="oauth_external",
        env_vars=(),
    )
    return SimpleNamespace(
        _REGISTRY={"openai-codex": profile},
        _ALIASES={"codex": "openai-codex", "openai_codex": "openai-codex"},
        _discovered=True,
        _discovery_error=None,
        _isolated_provider_allowlist=frozenset({"openai-codex"}),
        _isolated_discovery_validated=True,
    )


def test_exact_production_web_plugin_is_real_runnable_and_search_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.web_search_registry import _reset_for_tests, list_providers
    from gateway import production_model_sovereignty_runtime as runtime
    from hermes_cli import config as config_module
    from hermes_cli import plugins as plugin_module
    from hermes_cli.plugins import PluginManager
    from tools import web_tools
    from tools.registry import invalidate_check_fn_cache, registry

    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["evil"], "disabled": []},
                "web": {
                    "backend": "",
                    "search_backend": "ddgs",
                    "extract_backend": "",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # A user plugin is explicitly enabled, but strict discovery must never even
    # import it.  Its observable import side effect is the regression oracle.
    evil = home / "plugins" / "evil"
    evil.mkdir(parents=True)
    sentinel = tmp_path / "evil-imported"
    (evil / "plugin.yaml").write_text(
        "name: evil\nversion: 1.0.0\nkind: standalone\n",
        encoding="utf-8",
    )
    (evil / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n"
        "def register(ctx):\n    pass\n",
        encoding="utf-8",
    )

    # Supply a real importable module plus distribution metadata in an isolated
    # path.  This exercises the production version attestation without changing
    # the repository venv or reaching the network.
    deps = tmp_path / "deps"
    deps.mkdir()
    (deps / "ddgs.py").write_text("class DDGS:\n    pass\n", encoding="utf-8")
    dist_info = deps / "ddgs-9.14.4.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ddgs\nVersion: 9.14.4\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.syspath_prepend(str(deps))
    monkeypatch.delitem(sys.modules, "ddgs", raising=False)
    monkeypatch.setattr(plugin_module, "_plugin_manager", PluginManager())
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    importlib.invalidate_caches()
    _reset_for_tests()
    invalidate_check_fn_cache()

    try:
        plugin_module.discover_plugins(
            force=True,
            isolated_allowlist=runtime.PRODUCTION_PLUGIN_ALLOWLIST,
        )
        manager = plugin_module.get_plugin_manager()

        assert set(manager._plugins) == {runtime.PRODUCTION_WEB_PLUGIN_KEY}
        assert manager._hooks == {}
        assert manager._middleware == {}
        assert manager._plugin_tool_names == set()
        assert not sentinel.exists()
        assert [provider.name for provider in list_providers()] == ["ddgs"]

        runtime.validate_production_extension_surface(
            manager,
            SimpleNamespace(_handlers={}, _loaded_hooks=[]),
            _provider_registry(),
        )

        assert web_tools.check_web_search_available() is True
        assert web_tools.check_web_extract_available() is False
        definitions = registry.get_definitions({"web_search", "web_extract"})
        assert {
            definition["function"]["name"] for definition in definitions
        } == {"web_search"}

        from plugins.web.ddgs import provider as ddgs_provider

        monkeypatch.setattr(
            ddgs_provider,
            "_run_ddgs_search",
            lambda query, safe_limit: [
                {
                    "title": "Mechanical result",
                    "url": "https://example.com/evidence",
                    "description": f"{query}:{safe_limit}",
                    "position": 1,
                }
            ],
        )
        payload = json.loads(web_tools.web_search_tool("exact evidence", limit=1))
        assert payload == {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Mechanical result",
                        "url": "https://example.com/evidence",
                        "description": "exact evidence:1",
                        "position": 1,
                    }
                ]
            },
        }
    finally:
        _reset_for_tests()
        invalidate_check_fn_cache()
        config_module._LOAD_CONFIG_CACHE.clear()
        config_module._RAW_CONFIG_CACHE.clear()
        sys.modules.pop("ddgs", None)


def test_web_tool_schema_gates_are_capability_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.web_search_registry import _reset_for_tests, register_provider
    from plugins.web.ddgs.provider import DDGSWebSearchProvider
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        DDGSWebSearchProvider,
        "is_available",
        lambda self: True,
    )
    _reset_for_tests()
    try:
        assert web_tools.check_web_search_available() is False
        assert web_tools.check_web_extract_available() is False

        register_provider(DDGSWebSearchProvider())
        assert web_tools.check_web_search_available() is True
        assert web_tools.check_web_extract_available() is False
    finally:
        _reset_for_tests()


def test_strict_discovery_rejects_unaudited_bundled_backend() -> None:
    from hermes_cli.plugins import PluginManager

    with pytest.raises(RuntimeError, match="audited mechanical backend"):
        PluginManager().discover_and_load(
            isolated_allowlist=frozenset({"web/firecrawl"})
        )
