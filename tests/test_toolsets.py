"""Tests for toolsets.py — toolset resolution, validation, and composition."""

from tools.registry import ToolRegistry
from toolsets import (
    TOOLSETS,
    get_toolset,
    resolve_toolset,
    resolve_multiple_toolsets,
    get_all_toolsets,
    validate_toolset,
    create_custom_toolset,
    get_toolset_info,
)


def _dummy_handler(args, **kwargs):
    return "{}"


def _make_schema(name: str, description: str = "test tool"):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


class TestGetToolset:
    def test_known_toolset(self):
        ts = get_toolset("web")
        assert ts is not None
        assert "web_search" in ts["tools"]

    def test_merges_registry_tools_into_builtin_toolset(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="web_search_plus",
            toolset="web",
            schema=_make_schema("web_search_plus", "Plugin web search"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        ts = get_toolset("web")
        assert ts is not None
        assert set(ts["tools"]) == {"web_search", "web_extract", "web_search_plus"}

    def test_unknown_returns_none(self):
        assert get_toolset("nonexistent") is None


class TestResolveToolset:
    def test_leaf_toolset(self):
        tools = resolve_toolset("web")
        assert set(tools) == {"web_search", "web_extract"}

    def test_composite_toolset(self):
        tools = resolve_toolset("debugging")
        assert "terminal" in tools
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_cycle_detection(self):
        # Create a cycle: A includes B, B includes A
        TOOLSETS["_cycle_a"] = {"description": "test", "tools": ["t1"], "includes": ["_cycle_b"]}
        TOOLSETS["_cycle_b"] = {"description": "test", "tools": ["t2"], "includes": ["_cycle_a"]}
        try:
            tools = resolve_toolset("_cycle_a")
            # Should not infinite loop — cycle is detected
            assert "t1" in tools
            assert "t2" in tools
        finally:
            del TOOLSETS["_cycle_a"]
            del TOOLSETS["_cycle_b"]

    def test_unknown_toolset_returns_empty(self):
        assert resolve_toolset("nonexistent") == []

    def test_plugin_toolset_uses_registry_snapshot(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_b",
            toolset="plugin_example",
            schema=_make_schema("plugin_b", "B"),
            handler=_dummy_handler,
        )
        reg.register(
            name="plugin_a",
            toolset="plugin_example",
            schema=_make_schema("plugin_a", "A"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        assert resolve_toolset("plugin_example") == ["plugin_a", "plugin_b"]

    def test_all_alias(self):
        tools = resolve_toolset("all")
        assert len(tools) > 10  # Should resolve all tools from all toolsets

    def test_star_alias(self):
        tools = resolve_toolset("*")
        assert len(tools) > 10


class TestResolveMultipleToolsets:
    def test_combines_and_deduplicates(self):
        tools = resolve_multiple_toolsets(["web", "terminal"])
        assert "web_search" in tools
        assert "web_extract" in tools
        assert "terminal" in tools
        # No duplicates
        assert len(tools) == len(set(tools))

    def test_empty_list(self):
        assert resolve_multiple_toolsets([]) == []


class TestValidateToolset:
    def test_valid(self):
        assert validate_toolset("web") is True
        assert validate_toolset("terminal") is True

    def test_all_alias_valid(self):
        assert validate_toolset("all") is True
        assert validate_toolset("*") is True

    def test_invalid(self):
        assert validate_toolset("nonexistent") is False

    def test_mcp_alias_uses_live_registry(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="mcp__dynserver__ping",
            toolset="mcp-dynserver",
            schema=_make_schema("mcp__dynserver__ping", "Ping"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("dynserver", "mcp-dynserver")

        monkeypatch.setattr("tools.registry.registry", reg)

        assert validate_toolset("dynserver") is True
        assert validate_toolset("mcp-dynserver") is True
        assert "mcp__dynserver__ping" in resolve_toolset("dynserver")


class TestGetToolsetInfo:
    def test_leaf(self):
        info = get_toolset_info("web")
        assert info["name"] == "web"
        assert info["is_composite"] is False
        assert info["tool_count"] == 2

    def test_composite(self):
        info = get_toolset_info("debugging")
        assert info["is_composite"] is True
        assert info["tool_count"] > len(info["direct_tools"])

    def test_unknown_returns_none(self):
        assert get_toolset_info("nonexistent") is None


class TestCreateCustomToolset:
    def test_runtime_creation(self):
        create_custom_toolset(
            name="_test_custom",
            description="Test toolset",
            tools=["web_search"],
            includes=["terminal"],
        )
        try:
            tools = resolve_toolset("_test_custom")
            assert "web_search" in tools
            assert "terminal" in tools
            assert validate_toolset("_test_custom") is True
        finally:
            del TOOLSETS["_test_custom"]


class TestRegistryOwnedToolsets:
    def test_registry_membership_is_live(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="test_live_toolset_tool",
            toolset="test-live-toolset",
            schema=_make_schema("test_live_toolset_tool", "Live"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        assert validate_toolset("test-live-toolset") is True
        assert get_toolset("test-live-toolset")["tools"] == ["test_live_toolset_tool"]
        assert resolve_toolset("test-live-toolset") == ["test_live_toolset_tool"]


class TestToolsetConsistency:
    """Verify structural integrity of the built-in TOOLSETS dict."""

    def test_all_toolsets_have_required_keys(self):
        for name, ts in TOOLSETS.items():
            assert "description" in ts, f"{name} missing description"
            assert "tools" in ts, f"{name} missing tools"
            assert "includes" in ts, f"{name} missing includes"

    def test_all_includes_reference_existing_toolsets(self):
        for name, ts in TOOLSETS.items():
            for inc in ts["includes"]:
                assert inc in TOOLSETS, f"{name} includes unknown toolset '{inc}'"

    def test_hermes_platforms_share_core_tools(self):
        """All hermes-* platform toolsets share the same core tools.

        Platform-specific additions (e.g. ``discord`` / ``discord_admin``
        on hermes-discord, gated on DISCORD_BOT_TOKEN) are allowed on top —
        the invariant is that the core set is identical across platforms.
        """
        platforms = ["hermes-cli", "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-homeassistant"]
        tool_sets = [set(TOOLSETS[p]["tools"]) for p in platforms]
        # All platforms must contain the shared core; platform-specific
        # extras are OK (subset check, not equality).
        core = set.intersection(*tool_sets)
        for name, ts in zip(platforms, tool_sets):
            assert core.issubset(ts), f"{name} is missing core tools: {core - ts}"
        # Sanity: the shared core must be non-trivial (i.e. we didn't
        # silently let a platform diverge so far that nothing is shared).
        assert len(core) > 20, f"Suspiciously small shared core: {len(core)} tools"


class TestPluginToolsets:
    def test_get_all_toolsets_includes_plugin_toolset(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_tool",
            toolset="plugin_bundle",
            schema=_make_schema("plugin_tool", "Plugin tool"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        all_toolsets = get_all_toolsets()
        assert "plugin_bundle" in all_toolsets
        assert all_toolsets["plugin_bundle"]["tools"] == ["plugin_tool"]


class TestDefaultPlatformWebSearchCoverage:
    def test_hermes_whatsapp_toolset_includes_web_search(self):
        assert "web_search" in resolve_toolset("hermes-whatsapp")

    def test_hermes_api_server_toolset_includes_web_search(self):
        assert "web_search" in resolve_toolset("hermes-api-server")


class TestResolveToolsetIncludeRegistry:
    """include_registry flag exposes the static (pre-registry-merge) view used
    by platform reverse-mapping. Regression harness for issue #49622."""

    def test_include_registry_false_excludes_registry_tools(self):
        from tools.registry import discover_builtin_tools
        discover_builtin_tools()  # registers read_terminal into 'terminal'

        merged = set(resolve_toolset("terminal"))
        static = set(resolve_toolset("terminal", include_registry=False))

        assert static == {"terminal", "process"}, static
        # read_terminal is registered into 'terminal' but is desktop-only and
        # not part of the static definition — it must only appear in the merged view.
        assert "read_terminal" in merged
        assert "read_terminal" not in static

    def test_get_toolset_include_registry_false_is_static(self):
        ts = get_toolset("delegation", include_registry=False)
        assert ts is not None
        assert ts["tools"] == ["delegate_task"]

    def test_static_view_threads_through_includes(self):
        # 'debugging' has direct tools [terminal, process] and includes [web, file]
        static = set(resolve_toolset("debugging", include_registry=False))
        assert {"terminal", "process"} <= static
        assert "web_search" in static
        assert "read_file" in static

    def test_all_alias_accepts_include_registry(self):
        merged = set(resolve_toolset("all"))
        static = set(resolve_toolset("all", include_registry=False))
        assert static <= merged

    def test_registry_only_toolset_static_view_is_empty(self):
        assert resolve_toolset("__definitely_not_a_real_toolset__", include_registry=False) == []

class TestGetToolsetRegistryFailure:
    """Cover get_toolset branches that require registry import or alias wiring."""

    def test_registry_import_error_returns_static_toolset(self, monkeypatch):
        # Lines 570-571: except Exception on registry import returns static entry.
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("simulated missing registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        ts = get_toolset("web")
        assert ts is not None
        assert "web_search" in ts["tools"]

    def test_registry_import_error_returns_none_for_unknown(self, monkeypatch):
        # Lines 570-571: except Exception path when toolset not in TOOLSETS.
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("simulated missing registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        assert get_toolset("nonexistent_xyz") is None

    def test_alias_with_no_target_returns_none(self, monkeypatch):
        # Line 587: name not in plugin toolsets and alias_target is falsy.
        reg = ToolRegistry()
        reg.register_toolset_alias("dangling-alias", "nonexistent-canonical")
        monkeypatch.setattr("tools.registry.registry", reg)
        # Patch get_toolset_alias_target on the live registry object to return None,
        # so registry_toolset is falsy and the early return fires.
        monkeypatch.setattr(reg, "get_toolset_alias_target", lambda name: None)
        result = get_toolset("dangling-alias")
        assert result is None

    def test_canonical_with_reverse_alias_uses_alias_in_description(self, monkeypatch):
        # Line 597: canonical toolset found in plugin names; reverse alias found,
        # so description becomes "MCP server '<alias>' tools".
        reg = ToolRegistry()
        reg.register(
            name="srv_ping",
            toolset="mcp-myserver",
            schema=_make_schema("srv_ping", "Ping"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("myserver", "mcp-myserver")
        monkeypatch.setattr("tools.registry.registry", reg)
        ts = get_toolset("mcp-myserver")
        assert ts is not None
        assert ts["description"] == "MCP server 'myserver' tools"


class TestResolveToolsetHermesPlatform:
    """Cover the hermes-<platform> dynamic resolution branch in resolve_toolset."""

    def test_registered_platform_returns_core_tools(self, monkeypatch):
        # Lines 647-663: platform_registry.is_registered is True; result includes
        # _HERMES_CORE_TOOLS plus any tools the plugin registered under the platform name.
        from unittest.mock import MagicMock

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True

        reg = ToolRegistry()
        reg.register(
            name="my_platform_tool",
            toolset="myplatform",
            schema=_make_schema("my_platform_tool", "Platform tool"),
            handler=_dummy_handler,
        )
        monkeypatch.setattr("tools.registry.registry", reg)
        monkeypatch.setattr(
            "gateway.platform_registry.platform_registry", mock_registry
        )

        tools = resolve_toolset("hermes-myplatform")
        assert "web_search" in tools       # from _HERMES_CORE_TOOLS
        assert "my_platform_tool" in tools  # registered under the platform toolset

    def test_unregistered_platform_returns_empty(self, monkeypatch):
        # Lines 641-666: hermes-<platform> but is_registered is False; falls through to [].
        from unittest.mock import MagicMock

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        monkeypatch.setattr(
            "gateway.platform_registry.platform_registry", mock_registry
        )

        result = resolve_toolset("hermes-notregistered")
        assert result == []

    def test_platform_registry_import_error_returns_empty(self, monkeypatch):
        # Lines 647-665: outer except Exception; platform_registry import fails.
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "gateway.platform_registry":
                raise ImportError("no gateway")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        result = resolve_toolset("hermes-nomodule")
        assert result == []

    def test_platform_tool_registry_import_error_falls_back_to_core(self, monkeypatch):
        # Lines 656-660: inner except Exception; tools.registry import fails but
        # platform_registry import succeeds, so result is just _HERMES_CORE_TOOLS.
        from unittest.mock import MagicMock
        mock_pr = MagicMock()
        mock_pr.is_registered.return_value = True

        monkeypatch.setattr(
            "gateway.platform_registry.platform_registry", mock_pr
        )

        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("no registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        tools = resolve_toolset("hermes-fallbackplatform")
        assert "web_search" in tools


class TestPrivateHelperExceptionPaths:
    """Cover the except-fallback paths in _get_plugin_toolset_names and _get_registry_toolset_aliases."""

    def test_get_plugin_toolset_names_import_error_returns_empty(self, monkeypatch):
        # Lines 712-713: registry import raises -> returns set().
        import builtins
        import toolsets as ts_mod
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("no registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        result = ts_mod._get_plugin_toolset_names()
        assert result == set()

    def test_get_registry_toolset_aliases_import_error_returns_empty(self, monkeypatch):
        # Lines 721-722: registry import raises -> returns {}.
        import builtins
        import toolsets as ts_mod
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("no registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        result = ts_mod._get_registry_toolset_aliases()
        assert result == {}


class TestGetAllToolsetsAliasBranches:
    """Cover the alias-display-name loop and duplicate-skip guard in get_all_toolsets."""

    def test_plugin_toolset_shown_under_alias_name(self, monkeypatch):
        # Lines 739-743: canonical name is mcp-srv2; alias is "srv2";
        # get_all_toolsets should expose it as "srv2", not "mcp-srv2".
        reg = ToolRegistry()
        reg.register(
            name="srv2_op",
            toolset="mcp-srv2",
            schema=_make_schema("srv2_op", "Op"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("srv2", "mcp-srv2")
        monkeypatch.setattr("tools.registry.registry", reg)

        all_ts = get_all_toolsets()
        assert "srv2" in all_ts
        assert "mcp-srv2" not in all_ts  # canonical hidden behind alias

    def test_duplicate_display_name_skipped(self, monkeypatch):
        # Lines 741-742: display_name matches an existing static toolset; must be skipped.
        reg = ToolRegistry()
        reg.register(
            name="web_extra",
            toolset="web",  # same name as the static "web" toolset
            schema=_make_schema("web_extra", "Extra web"),
            handler=_dummy_handler,
        )
        monkeypatch.setattr("tools.registry.registry", reg)

        all_ts = get_all_toolsets()
        # "web" is in TOOLSETS; the plugin entry should not overwrite it.
        assert "web" in all_ts
        assert "web_search" in all_ts["web"]["tools"]


class TestGetToolsetNamesForElse:
    """Cover the for/else else-branch in get_toolset_names."""

    def test_unaliased_plugin_toolset_appears_under_canonical_name(self, monkeypatch):
        # Lines 763-765: plugin toolset "mcp-rawplugin" has no alias; else branch
        # adds "mcp-rawplugin" directly to names.
        reg = ToolRegistry()
        reg.register(
            name="rawplugin_op",
            toolset="mcp-rawplugin",
            schema=_make_schema("rawplugin_op", "Raw op"),
            handler=_dummy_handler,
        )
        monkeypatch.setattr("tools.registry.registry", reg)

        import toolsets as ts_mod
        names = ts_mod.get_toolset_names()
        assert "mcp-rawplugin" in names


class TestValidateToolsetPluginAndAlias:
    """Cover the plugin-name and registry-alias paths in validate_toolset."""

    def test_plugin_toolset_name_is_valid(self, monkeypatch):
        # Line 787: name is in _get_plugin_toolset_names() -> True.
        reg = ToolRegistry()
        reg.register(
            name="val_plugin_tool",
            toolset="mcp-valplugin",
            schema=_make_schema("val_plugin_tool", "Val"),
            handler=_dummy_handler,
        )
        monkeypatch.setattr("tools.registry.registry", reg)
        assert validate_toolset("mcp-valplugin") is True

    def test_registry_alias_name_is_valid(self, monkeypatch):
        # Line 789: name is in _get_registry_toolset_aliases() -> True.
        reg = ToolRegistry()
        reg.register(
            name="val_alias_tool",
            toolset="mcp-valcanon",
            schema=_make_schema("val_alias_tool", "Val alias"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("val-alias", "mcp-valcanon")
        monkeypatch.setattr("tools.registry.registry", reg)
        assert validate_toolset("val-alias") is True
