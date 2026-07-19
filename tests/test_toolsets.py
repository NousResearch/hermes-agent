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
        from toolsets import _HERMES_CORE_FAMILY, _HERMES_CORE_TOOLS
        from hermes_cli.platforms import PLATFORMS

        # Every member must actually carry the full core tool set.
        for name in _HERMES_CORE_FAMILY:
            assert set(_HERMES_CORE_TOOLS) <= set(TOOLSETS[name]["tools"]), name

        # Derive the expected members from the platform definitions instead
        # of a literal snapshot: every platform default toolset except the
        # deliberate exclusions must be in the family, so a newly added
        # messaging platform is covered automatically.
        expected_members = {
            info.default_toolset for info in PLATFORMS.values()
        } - self._EXCLUDED_SURFACES
        assert expected_members <= _HERMES_CORE_FAMILY

        # Constrained/curated toolsets must never be picked up.
        for name in self._EXCLUDED_SURFACES | {"hermes-gateway", "coding", "web"}:
            assert name not in _HERMES_CORE_FAMILY, name

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
