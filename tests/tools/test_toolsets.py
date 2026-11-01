"""Tests for toolsets.py — toolset resolution, validation, and composition."""

import toolsets as toolsets_mod
from tools.registry import ToolRegistry
from toolsets import (
    TOOLSETS,
    get_toolset,
    resolve_toolset,
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
        assert {"web_search", "web_search_plus"} <= set(ts["tools"])

    def test_static_and_mcp_alias_with_same_name_are_merged(self, monkeypatch):
        # An MCP server named like a built-in toolset registers a bare alias to its
        # `mcp-<name>` toolset; the static entry must union those tools in (and keep
        # its own includes) instead of shadowing the server.
        TOOLSETS["_mergetest"] = {"description": "static", "tools": ["builtin_tool_a"], "includes": ["web"]}
        try:
            reg = ToolRegistry()
            reg.register(name="mcp__mergetest_call", toolset="mcp-_mergetest",
                         schema=_make_schema("mcp__mergetest_call", "Call"), handler=_dummy_handler)
            reg.register_toolset_alias("_mergetest", "mcp-_mergetest")
            monkeypatch.setattr("tools.registry.registry", reg)

            ts = get_toolset("_mergetest")
            assert {"builtin_tool_a", "mcp__mergetest_call"} <= set(ts["tools"])
            assert ts["includes"] == ["web"]
        finally:
            del TOOLSETS["_mergetest"]

class TestResolveToolset:

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

class TestValidateToolset:

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

        assert "terminal" in static, static
        # Registered into 'terminal' but not part of the static definition — it
        # must only appear in the merged view.
        assert "__probe_registry_only_tool__" in merged
        assert "__probe_registry_only_tool__" not in static

    def test_static_view_threads_through_includes(self):
        # 'debugging' has direct tools [terminal, process] and includes [web, file]
        static = set(resolve_toolset("debugging", include_registry=False))
        assert {"terminal", "process_manage"} <= static
        assert "web_search" in static
        assert "read_file" in static

    def test_registry_only_toolset_static_view_is_empty(self):
        assert resolve_toolset("__definitely_not_a_real_toolset__", include_registry=False) == []

class TestResolveToolsetMemo:
    """Measured-work pins for the generation-keyed resolution memo."""

    def test_generation_bump_invalidates_memo(self, monkeypatch):
        """A registry mutation (generation bump) must force a fresh resolve."""
        from tools.registry import registry

        toolsets_mod._resolve_toolset_memo.clear()
        get_toolset_calls = {"n": 0}

        orig_get_toolset = toolsets_mod.get_toolset

        def counting_get_toolset(name, *, include_registry=True):
            get_toolset_calls["n"] += 1
            return orig_get_toolset(name, include_registry=include_registry)

        monkeypatch.setattr(toolsets_mod, "get_toolset", counting_get_toolset)

        resolve_toolset("hermes-cli")
        assert get_toolset_calls["n"] == 1

        # Simulate a registry mutation bumping the generation.
        registry._generation += 1
        resolve_toolset("hermes-cli")
        assert get_toolset_calls["n"] == 2, (
            "generation bump must invalidate the memo and re-resolve"
        )


class TestMessagingOptInTools:
    """Tools registered with include_in_messaging_toolsets=True are unioned
    into every toolset that carries the full core tool set (CLI, cron, and all
    messaging platforms) at resolve time, without core hardcoding them."""

    def _registry_with_optin(self):
        reg = ToolRegistry()
        reg.register(
            name="optin_tool",
            toolset="notes_plugin",
            schema=_make_schema("optin_tool", "Opt-in tool"),
            handler=_dummy_handler,
            include_in_messaging_toolsets=True,
        )
        reg.register(
            name="plain_tool",
            toolset="notes_plugin",
            schema=_make_schema("plain_tool", "Plain tool"),
            handler=_dummy_handler,
        )
        return reg

    # Deliberately constrained/curated surfaces that must never receive
    # opt-in tools: webhook (untrusted input), ACP and API server (curated).
    _EXCLUDED_SURFACES = frozenset({"hermes-webhook", "hermes-acp", "hermes-api-server"})

    def test_core_family_membership(self):
        from toolsets import HERMES_CORE_FAMILY, _HERMES_CORE_TOOLS
        from hermes_cli.platforms import PLATFORMS

        # Every member must actually carry the full core tool set.
        for name in HERMES_CORE_FAMILY:
            assert set(_HERMES_CORE_TOOLS) <= set(TOOLSETS[name]["tools"]), name

        # Derive the expected members from the platform definitions instead
        # of a literal snapshot: every platform default toolset except the
        # deliberate exclusions must be in the family, so a newly added
        # messaging platform is covered automatically.
        expected_members = {
            info.default_toolset for info in PLATFORMS.values()
        } - self._EXCLUDED_SURFACES
        assert expected_members <= HERMES_CORE_FAMILY

        # Constrained/curated toolsets must never be picked up.
        for name in self._EXCLUDED_SURFACES | {"hermes-gateway", "coding", "web"}:
            assert name not in HERMES_CORE_FAMILY, name

    def test_optin_tool_resolves_into_core_family_toolsets(self, monkeypatch):
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        for ts in ["hermes-telegram", "hermes-discord", "hermes-slack", "hermes-cli", "hermes-cron"]:
            resolved = resolve_toolset(ts)
            assert "optin_tool" in resolved, f"{ts} missing opt-in tool"
            assert "plain_tool" not in resolved, f"{ts} leaked non-opt-in tool"

    def test_optin_tool_excluded_from_non_core_toolsets(self, monkeypatch):
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        # Webhook is intentionally constrained; acp/api-server are curated.
        for ts in ["hermes-webhook", "hermes-acp", "hermes-api-server", "web"]:
            assert "optin_tool" not in resolve_toolset(ts), f"{ts} wrongly got opt-in tool"

    def test_optin_tool_reaches_gateway_via_includes(self, monkeypatch):
        # hermes-gateway has no direct tools; it must inherit the opt-in tool
        # through its included platform toolsets.
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        assert "optin_tool" in resolve_toolset("hermes-gateway")

    def test_optin_tool_resolves_into_plugin_platform_toolsets(self, monkeypatch):
        # Auto-generated toolsets for plugin platforms (hermes-<name>) get the
        # core surface, so they must union opt-in tools too.
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        from gateway.platform_registry import platform_registry
        monkeypatch.setattr(platform_registry, "is_registered", lambda name: name == "notesplatform")
        resolved = resolve_toolset("hermes-notesplatform")
        assert "optin_tool" in resolved
        assert "terminal" in resolved  # core tools still present

    def test_static_view_excludes_optin_tools(self, monkeypatch):
        # include_registry=False is the pure-TOOLSETS view used by platform
        # reverse-mapping (#49622); registry-derived unions must not leak in.
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        assert "optin_tool" not in resolve_toolset("hermes-telegram", include_registry=False)

    def test_registry_query_returns_optin_names(self):
        reg = self._registry_with_optin()
        assert reg.get_messaging_optin_tool_names() == ["optin_tool"]

    def test_optin_tool_still_in_its_own_toolset(self, monkeypatch):
        monkeypatch.setattr("tools.registry.registry", self._registry_with_optin())
        assert resolve_toolset("notes_plugin") == ["optin_tool", "plain_tool"]
