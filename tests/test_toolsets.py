"""Tests for toolsets.py — toolset resolution, validation, and composition."""

import pytest

from tools.registry import registry
from toolsets import (
    LEGACY_TOOLSET_ALIASES,
    TOOLSETS,
    get_legacy_toolset_map,
    get_toolset,
    resolve_toolset,
    resolve_legacy_toolset,
    resolve_multiple_toolsets,
    get_all_toolsets,
    get_toolset_names,
    is_legacy_toolset,
    validate_toolset,
    create_custom_toolset,
    get_toolset_info,
)


class TestGetToolset:
    def test_known_toolset(self):
        ts = get_toolset("web")
        assert ts is not None
        assert "web_search" in ts["tools"]

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

    def test_mcp_alias_uses_live_registry(self):
        registry.register(
            name="mcp_dynserver_ping",
            toolset="mcp-dynserver",
            schema={"name": "mcp_dynserver_ping", "description": "Ping", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        try:
            assert validate_toolset("dynserver") is True
            assert validate_toolset("mcp-dynserver") is True
            assert "mcp_dynserver_ping" in resolve_toolset("dynserver")
        finally:
            registry.deregister("mcp_dynserver_ping")


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
    def test_registry_membership_is_live(self):
        registry.register(
            name="test_live_toolset_tool",
            toolset="test-live-toolset",
            schema={"name": "test_live_toolset_tool", "description": "Live", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        try:
            assert validate_toolset("test-live-toolset") is True
            assert get_toolset("test-live-toolset")["tools"] == ["test_live_toolset_tool"]
            assert resolve_toolset("test-live-toolset") == ["test_live_toolset_tool"]
        finally:
            registry.deregister("test_live_toolset_tool")


class TestLegacyToolsets:
    def test_legacy_aliases_are_owned_by_toolsets(self):
        assert is_legacy_toolset("web_tools") is True
        assert is_legacy_toolset("nonexistent_legacy_tools") is False
        assert LEGACY_TOOLSET_ALIASES["web_tools"] == ["web"]

    def test_resolve_legacy_toolset(self):
        tools = resolve_legacy_toolset("web_tools")
        assert set(tools) == {"web_search", "web_extract"}

    def test_legacy_toolset_map_is_live(self):
        registry.register(
            name="test_live_legacy_file_tool",
            toolset="file",
            schema={"name": "test_live_legacy_file_tool", "description": "Live", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        try:
            assert "test_live_legacy_file_tool" in get_legacy_toolset_map()["file_tools"]
        finally:
            registry.deregister("test_live_legacy_file_tool")


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
        """All hermes-* platform toolsets should have the same tools."""
        platforms = ["hermes-cli", "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-homeassistant"]
        tool_sets = [set(resolve_toolset(p)) for p in platforms]
        # All platform toolsets should be identical
        for ts in tool_sets[1:]:
            assert ts == tool_sets[0]
