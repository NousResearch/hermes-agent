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

    def test_x_search_toolset_marks_read_only_and_points_to_xurl(self):
        ts = get_toolset("x_search")
        assert ts is not None
        assert ts["tools"] == ["x_search"]
        description = ts["description"].lower()
        assert "read-only" in description
        assert "xurl" in description
        assert "authenticated" in description

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




class TestResolveMultipleToolsets:
    def test_combines_and_deduplicates(self):
        tools = resolve_multiple_toolsets(["web", "terminal"])
        assert "web_search" in tools
        assert "web_extract" in tools
        assert "terminal" in tools
        # No duplicates
        assert len(tools) == len(set(tools))



class TestValidateToolset:
    def test_valid(self):
        assert validate_toolset("web") is True
        assert validate_toolset("terminal") is True


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



class TestResolveToolsetIncludeRegistry:
    """include_registry flag exposes the static (pre-registry-merge) view used
    by platform reverse-mapping. Regression harness for issue #49622."""

    def test_include_registry_false_excludes_registry_tools(self):
        from tools.registry import discover_builtin_tools, registry
        discover_builtin_tools()

        # Register a tool into `terminal` at runtime, the way plugins and MCP
        # servers do, so the split is exercised on the mechanism rather than on
        # whichever built-in currently happens to live where.
        registry.register(
            name="__probe_registry_only_tool__",
            toolset="terminal",
            schema={"name": "__probe_registry_only_tool__", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "",
        )
        try:
            merged = set(resolve_toolset("terminal"))
            static = set(resolve_toolset("terminal", include_registry=False))
        finally:
            registry.deregister("__probe_registry_only_tool__")

        assert static == {"terminal", "process"}, static
        # Registered into 'terminal' but not part of the static definition — it
        # must only appear in the merged view.
        assert "__probe_registry_only_tool__" in merged
        assert "__probe_registry_only_tool__" not in static


    def test_static_view_threads_through_includes(self):
        # 'debugging' has direct tools [terminal, process] and includes [web, file]
        static = set(resolve_toolset("debugging", include_registry=False))
        assert {"terminal", "process"} <= static
        assert "web_search" in static
        assert "read_file" in static


    def test_registry_only_toolset_static_view_is_empty(self):
        assert resolve_toolset("__definitely_not_a_real_toolset__", include_registry=False) == []

class TestGetAllToolsetsAliasBranches:
    """Cover the alias-display-name loop and duplicate-skip guard in get_all_toolsets."""

    def test_plugin_toolset_shown_under_alias_name(self, monkeypatch):
        # A registry-only canonical toolset is exposed under its non-static alias.
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
        # Two registry-only canonical names resolve to the same display key.
        reg = ToolRegistry()
        reg.register(
            name="alpha_op",
            toolset="alpha",
            schema=_make_schema("alpha_op", "Alpha"),
            handler=_dummy_handler,
        )
        reg.register(
            name="srv2_op",
            toolset="mcp-srv2",
            schema=_make_schema("srv2_op", "Op"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("alpha", "mcp-srv2")
        monkeypatch.setattr("tools.registry.registry", reg)

        import toolsets as ts_mod

        assert set(ts_mod._get_plugin_toolset_names()) == {"alpha", "mcp-srv2"}
        assert ts_mod._get_registry_toolset_aliases() == {"alpha": "mcp-srv2"}

        real_get_tool_names = reg.get_tool_names_for_toolset
        materializations = []

        def record_materialization(toolset):
            names = real_get_tool_names(toolset)
            if toolset == "alpha":
                marker = f"materialization_{len(materializations)}"
                materializations.append((tuple(names), marker))
                return [*names, marker]
            return names

        monkeypatch.setattr(reg, "get_tool_names_for_toolset", record_materialization)
        all_ts = ts_mod.get_all_toolsets()

        assert "alpha" in all_ts
        assert "mcp-srv2" not in all_ts
        assert len(materializations) == 1
        names, marker = materializations[0]
        assert all_ts["alpha"]["tools"] == sorted([*names, marker])
        assert "alpha_op" in all_ts["alpha"]["tools"]

class TestGetToolsetRegistryFailure:
    """Cover get_toolset branches that require registry import or alias wiring."""

    def test_registry_import_error_returns_static_toolset(self, monkeypatch):
        # A missing registry preserves the static toolset definition.
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
        # A missing registry leaves unknown names unresolved.
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "tools.registry":
                raise ImportError("simulated missing registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        assert get_toolset("nonexistent_xyz") is None

    def test_alias_with_no_target_returns_none(self, monkeypatch):
        # An alias without a canonical target is unresolved.
        reg = ToolRegistry()
        reg.register_toolset_alias("dangling-alias", "nonexistent-canonical")
        monkeypatch.setattr("tools.registry.registry", reg)
        # The live registry reports that the alias has no target.
        monkeypatch.setattr(reg, "get_toolset_alias_target", lambda name: None)
        result = get_toolset("dangling-alias")
        assert result is None

    def test_canonical_with_reverse_alias_uses_alias_in_description(self, monkeypatch):
        # A canonical registry toolset uses its reverse alias in the description.
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
        # A registered platform resolves core tools plus its registry tools.
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
        # An unregistered platform has no generated toolset.
        from unittest.mock import MagicMock

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        monkeypatch.setattr(
            "gateway.platform_registry.platform_registry", mock_registry
        )

        result = resolve_toolset("hermes-notregistered")
        assert result == []

    def test_platform_registry_import_error_returns_empty(self, monkeypatch):
        # An unavailable platform registry leaves the generated toolset empty.
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
        # A registered platform keeps core tools when its tool registry is unavailable.
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
        # An unavailable registry produces no plugin toolset names.
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
        # An unavailable registry produces no registry aliases.
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


class TestGetToolsetNamesForElse:
    """Cover the for/else else-branch in get_toolset_names."""

    def test_unaliased_plugin_toolset_appears_under_canonical_name(self, monkeypatch):
        # An unaliased registry toolset appears under its canonical name.
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
        # A registry-only canonical name is valid.
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
        # A registered alias is valid.
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
