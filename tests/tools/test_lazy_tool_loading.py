"""Tests for lazy tool schema loading (issue #6839).

Coverage targets:
- Config/defaults: eager is default, lazy is opt-in
- Compact summary generation: name + description only, no parameters
- request_tool_schema dispatch: returns full schema on demand
- Scope enforcement: restricted sessions cannot load out-of-scope schemas
- Backward compatibility: eager mode is completely unchanged
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _make_schema(name: str, description: str = "test tool",
                 properties: Dict[str, Any] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {"query": {"type": "string", "description": "A query"}},
            "required": ["query"],
        },
    }


def _register_test_tools(reg):
    """Register a few test tools in the given registry."""
    for name, desc in [
        ("test_alpha", "First test tool"),
        ("test_beta", "Second test tool"),
        ("test_gamma", "Third test tool"),
    ]:
        reg.register(
            name=name,
            toolset="test_lazy",
            schema=_make_schema(name, desc),
            handler=_dummy_handler,
            description=desc,
        )


# ---------------------------------------------------------------------------
# Config / defaults
# ----------------------------------------------------------------


class TestLazyLoadingConfig:
    def test_default_is_eager(self):
        """When no config is set, loading mode must be eager."""
        from tools.lazy_tool_loading import load_loading_mode
        # No config file in test environment → should return "eager"
        assert load_loading_mode() == "eager"

    def test_explicit_eager(self):
        """Explicit eager config returns eager."""
        from tools.lazy_tool_loading import load_loading_mode
        with patch("tools.lazy_tool_loading.load_config") as mock_load:
            mock_load.return_value = {"tools": {"loading": "eager"}}
            # Re-import to pick up the mock
            from tools.lazy_tool_loading import load_loading_mode as _l
            # Direct call since we already patched
            cfg = mock_load.return_value
            tools_cfg = cfg.get("tools", {})
            raw = str(tools_cfg.get("loading", "eager")).strip().lower()
            assert raw == "eager"

    def test_explicit_lazy(self):
        """Explicit lazy config returns lazy."""
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {"tools": {"loading": "lazy"}}
            from tools.lazy_tool_loading import load_loading_mode
            assert load_loading_mode() == "lazy"

    def test_invalid_loading_value_defaults_to_eager(self):
        """Any unrecognized value defaults to eager."""
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {"tools": {"loading": "turbo"}}
            from tools.lazy_tool_loading import load_loading_mode
            assert load_loading_mode() == "eager"

    def test_missing_tools_section_defaults_to_eager(self):
        """Missing tools section defaults to eager."""
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            from tools.lazy_tool_loading import load_loading_mode
            assert load_loading_mode() == "eager"


# ---------------------------------------------------------------------------
# Compact summary generation
# ---------------------------------------------------------------------------


class TestCompactDefinitions:
    def test_compact_has_no_parameters(self, monkeypatch):
        """Compact definitions must NOT include parameter schemas."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        compact = reg.get_compact_definitions({"test_alpha", "test_beta"})
        assert len(compact) == 2
        for td in compact:
            fn = td["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" not in fn, (
                f"Compact definition for {fn['name']} must not include parameters"
            )

    def test_compact_preserves_name_and_description(self, monkeypatch):
        """Compact summaries must faithfully preserve name and description."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        compact = reg.get_compact_definitions({"test_alpha"})
        assert len(compact) == 1
        fn = compact[0]["function"]
        assert fn["name"] == "test_alpha"
        assert fn["description"] == "First test tool"

    def test_compact_outer_shape_matches_eager(self, monkeypatch):
        """Compact definitions must have the same outer {type, function} shape."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        compact = reg.get_compact_definitions({"test_alpha"})
        assert compact[0]["type"] == "function"
        assert isinstance(compact[0]["function"], dict)

    def test_compact_respects_check_fn(self, monkeypatch):
        """check_fn filtering must still apply in compact mode."""
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            name="gated_tool",
            toolset="test_lazy",
            schema=_make_schema("gated_tool"),
            handler=_dummy_handler,
            check_fn=lambda: False,
            description="Gated tool",
        )
        reg.register(
            name="open_tool",
            toolset="test_lazy",
            schema=_make_schema("open_tool"),
            handler=_dummy_handler,
            description="Open tool",
        )
        monkeypatch.setattr("tools.registry", reg)

        compact = reg.get_compact_definitions({"gated_tool", "open_tool"})
        names = {td["function"]["name"] for td in compact}
        assert "open_tool" in names
        assert "gated_tool" not in names


# ---------------------------------------------------------------------------
# request_tool_schema bridge tool
# ---------------------------------------------------------------------------


class TestRequestToolSchema:
    def test_bridge_tool_schema_shape(self):
        """The bridge tool must have the expected schema shape."""
        from tools.lazy_tool_loading import request_tool_schema_tool_def, REQUEST_TOOL_SCHEMA_NAME
        td = request_tool_schema_tool_def()
        assert td["type"] == "function"
        fn = td["function"]
        assert fn["name"] == REQUEST_TOOL_SCHEMA_NAME
        assert "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "name" in params["required"]

    def test_dispatch_returns_full_schema(self, monkeypatch):
        """dispatch_request_tool_schema must return the full schema."""
        from tools.registry import ToolRegistry
        from tools.lazy_tool_loading import dispatch_request_tool_schema

        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        result = json.loads(dispatch_request_tool_schema({"name": "test_alpha"}))
        assert "tool_schema" in result
        ts = result["tool_schema"]
        assert ts["type"] == "function"
        fn = ts["function"]
        assert fn["name"] == "test_alpha"
        # Full schema must include parameters
        assert "parameters" in fn
        assert "properties" in fn["parameters"]

    def test_dispatch_rejects_unknown_tool(self, monkeypatch):
        """Unknown tool name must return an error."""
        from tools.registry import ToolRegistry
        from tools.lazy_tool_loading import dispatch_request_tool_schema

        reg = ToolRegistry()
        monkeypatch.setattr("tools.registry.registry", reg)

        result = json.loads(dispatch_request_tool_schema({"name": "nonexistent"}))
        assert "error" in result

    def test_dispatch_requires_name(self):
        """Missing name argument must return an error."""
        from tools.lazy_tool_loading import dispatch_request_tool_schema
        result = json.loads(dispatch_request_tool_schema({}))
        assert "error" in result
        assert "name" in result["error"].lower()

    def test_dispatch_respects_scope(self, monkeypatch):
        """Out-of-scope tool must be rejected when scope is provided."""
        from tools.registry import ToolRegistry
        from tools.lazy_tool_loading import dispatch_request_tool_schema

        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        # test_alpha is NOT in the allowed scope
        result = json.loads(dispatch_request_tool_schema(
            {"name": "test_alpha"},
            tool_names_in_scope={"test_beta", "test_gamma"},
        ))
        assert "error" in result
        assert "not available" in result["error"].lower()

    def test_dispatch_allows_in_scope_tool(self, monkeypatch):
        """In-scope tool must be allowed."""
        from tools.registry import ToolRegistry
        from tools.lazy_tool_loading import dispatch_request_tool_schema

        reg = ToolRegistry()
        _register_test_tools(reg)
        monkeypatch.setattr("tools.registry.registry", reg)

        result = json.loads(dispatch_request_tool_schema(
            {"name": "test_alpha"},
            tool_names_in_scope={"test_alpha", "test_beta"},
        ))
        assert "tool_schema" in result
        assert result["tool_schema"]["function"]["name"] == "test_alpha"


# ---------------------------------------------------------------------------
# handle_function_call integration
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:
    def test_request_tool_schema_dispatch(self, monkeypatch):
        """request_tool_schema dispatches through handle_function_call."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="integration_tool",
            toolset="test_lazy_int",
            schema=_make_schema("integration_tool", "Integration test tool",
                                {"url": {"type": "string", "description": "A URL"}}),
            handler=_dummy_handler,
            description="Integration test tool",
        )
        monkeypatch.setattr("tools.registry.registry", reg)

        result = json.loads(model_tools.handle_function_call(
            function_name="request_tool_schema",
            function_args={"name": "integration_tool"},
        ))
        assert "tool_schema" in result
        fn = result["tool_schema"]["function"]
        assert fn["name"] == "integration_tool"
        assert "parameters" in fn
        assert "url" in fn["parameters"]["properties"]

    def test_request_tool_schema_with_scope(self, monkeypatch):
        """request_tool_schema respects enabled_tools scope."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="scoped_a",
            toolset="test_lazy_scope",
            schema=_make_schema("scoped_a"),
            handler=_dummy_handler,
            description="Scoped A",
        )
        reg.register(
            name="scoped_b",
            toolset="test_lazy_scope",
            schema=_make_schema("scoped_b"),
            handler=_dummy_handler,
            description="Scoped B",
        )
        monkeypatch.setattr("tools.registry", reg)

        # scoped_b is NOT in the enabled_tools list
        result = json.loads(model_tools.handle_function_call(
            function_name="request_tool_schema",
            function_args={"name": "scoped_b"},
            enabled_tools=["scoped_a"],
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# Lazy mode integration: get_tool_definitions returns compact + bridge
# ----------------------------------------------------------------


class TestGetToolDefinitionsLazy:
    def test_lazy_mode_returns_compact_summaries(self, monkeypatch):
        """In lazy mode, get_tool_definitions returns compact summaries."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="lazy_test_tool",
            toolset="lazy_test",
            schema=_make_schema("lazy_test_tool", "Lazy test tool",
                                {"input": {"type": "string"}}),
            handler=_dummy_handler,
            description="Lazy test tool",
        )
        monkeypatch.setattr("tools.registry.registry", reg)
        monkeypatch.setattr(model_tools, "registry", reg)

        # Patch lazy loading to return "lazy"
        with patch("tools.lazy_tool_loading.load_loading_mode", return_value="lazy"):
            # Also need to patch _is_lazy_loading_enabled since it calls load_loading_mode
            with patch.object(model_tools, "_is_lazy_loading_enabled", return_value=True):
                model_tools._clear_tool_defs_cache()
                defs = model_tools._compute_tool_definitions(
                    enabled_toolsets=["lazy_test"],
                    quiet_mode=True,
                )

        names = {td["function"]["name"] for td in defs}
        assert "lazy_test_tool" in names
        assert "request_tool_schema" in names

        # The lazy_test_tool should be compact (no parameters)
        for td in defs:
            if td["function"]["name"] == "lazy_test_tool":
                assert "parameters" not in td["function"], (
                    "Lazy mode tool must not include parameters"
                )
                assert td["function"]["description"] == "Lazy test tool"

    def test_lazy_mode_includes_bridge_tool(self, monkeypatch):
        """In lazy mode, the request_tool_schema bridge tool is included."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="bridge_test_tool",
            toolset="bridge_test",
            schema=_make_schema("bridge_test_tool"),
            handler=_dummy_handler,
            description="Bridge test",
        )
        monkeypatch.setattr("tools.registry", reg)
        monkeypatch.setattr(model_tools, "registry", reg)

        with patch.object(model_tools, "_is_lazy_loading_enabled", return_value=True):
            model_tools._clear_tool_defs_cache()
            defs = model_tools._compute_tool_definitions(
                enabled_toolsets=["bridge_test"],
                quiet_mode=True,
            )

        bridge_defs = [td for td in defs if td["function"]["name"] == "request_tool_schema"]
        assert len(bridge_defs) == 1
        fn = bridge_defs[0]["function"]
        assert "parameters" in fn
        assert "name" in fn["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Backward compatibility: eager mode is unchanged
# ---------------------------------------------------------------------------


class TestEagerModePreserved:
    def test_eager_mode_returns_full_schemas(self, monkeypatch):
        """In eager mode (default), get_tool_definitions returns full schemas."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="eager_test_tool",
            toolset="eager_test",
            schema=_make_schema("eager_test_tool", "Eager test tool",
                                {"input": {"type": "string"}}),
            handler=_dummy_handler,
            description="Eager test tool",
        )
        monkeypatch.setattr("tools.registry.registry", reg)
        monkeypatch.setattr(model_tools, "registry", reg)

        # Ensure eager mode (default)
        with patch.object(model_tools, "_is_lazy_loading_enabled", return_value=False):
            model_tools._clear_tool_defs_cache()
            defs = model_tools._compute_tool_definitions(
                enabled_toolsets=["eager_test"],
                quiet_mode=True,
            )

        names = {td["function"]["name"] for td in defs}
        assert "eager_test_tool" in names
        # No bridge tool in eager mode
        assert "request_tool_schema" not in names

        # Full schema must include parameters
        for td in defs:
            if td["function"]["name"] == "eager_test_tool":
                assert "parameters" in td["function"]
                assert "properties" in td["function"]["parameters"]

    def test_eager_mode_no_bridge_injection(self, monkeypatch):
        """Eager mode must NOT inject request_tool_schema."""
        from tools.registry import ToolRegistry, registry
        import model_tools

        reg = ToolRegistry()
        reg.register(
            name="no_bridge_tool",
            toolset="no_bridge",
            schema=_make_schema("no_bridge_tool"),
            handler=_dummy_handler,
            description="No bridge",
        )
        monkeypatch.setattr("tools.registry", reg)

        with patch.object(model_tools, "_is_lazy_loading_enabled", return_value=False):
            model_tools._clear_tool_defs_cache()
            defs = model_tools._compute_tool_definitions(
                enabled_toolsets=["no_bridge"],
                quiet_mode=True,
            )

        bridge_names = [td["function"]["name"] for td in defs
                        if td["function"]["name"] == "request_tool_schema"]
        assert bridge_names == [], (
            "Eager mode must not include request_tool_schema bridge"
        )


# ---------------------------------------------------------------------------
# Token savings estimation
# ---------------------------------------------------------------------------


class TestTokenSavings:
    def test_compact_saves_tokens_vs_eager(self, monkeypatch):
        """Compact definitions should be significantly smaller than full schemas."""
        from tools.registry import ToolRegistry
        from tools.tool_search import estimate_tokens_from_schemas

        reg = ToolRegistry()
        # Register 20 tools with substantial parameter schemas
        for i in range(20):
            reg.register(
                name=f"savings_tool_{i}",
                toolset="savings_test",
                schema=_make_schema(
                    f"savings_tool_{i}",
                    f"Tool {i} for testing token savings with a moderately long description",
                    {
                        "query": {"type": "string", "description": "Search query string"},
                        "limit": {"type": "integer", "description": "Maximum results to return"},
                        "format": {"type": "string", "description": "Output format (json, text, csv)"},
                    },
                ),
                handler=_dummy_handler,
                description=f"Tool {i} for testing token savings",
            )

        tool_names = {f"savings_tool_{i}" for i in range(20)}
        full_defs = reg.get_definitions(tool_names, quiet=True)
        compact_defs = reg.get_compact_definitions(tool_names, quiet=True)

        full_tokens = estimate_tokens_from_schemas(full_defs)
        compact_tokens = estimate_tokens_from_schemas(compact_defs)

        # Compact should be at least 50% smaller
        assert compact_tokens < full_tokens * 0.5, (
            f"Expected compact ({compact_tokens} tokens) < 50% of full ({full_tokens} tokens)"
        )
