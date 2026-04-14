"""Tests for registry runtime metadata (Phase A1)."""

import importlib

import pytest
from tools.registry import ToolRegistry, registry


def _dummy_handler(args, **kw):
    return '{"ok": true}'


def _make_registry():
    return ToolRegistry()


def test_real_registry_send_message_metadata():
    importlib.import_module("tools.send_message_tool")
    meta = registry.get_metadata("send_message")
    assert meta["mutates_external_world"] is True
    assert meta["requires_confirmation_default"] is True
    assert meta["risk_level"] == "critical"


def test_real_registry_browser_click_metadata():
    importlib.import_module("tools.browser_tool")
    meta = registry.get_metadata("browser_click")
    assert meta["mutates_browser_session"] is True
    assert meta["risk_level"] == "medium"


def test_real_registry_terminal_metadata_uses_command_level_approval():
    importlib.import_module("tools.terminal_tool")
    meta = registry.get_metadata("terminal")
    assert meta["mutates_local_fs"] is True
    assert meta["mutates_external_world"] is False
    assert meta["requires_confirmation_default"] is False
    assert meta["risk_level"] == "medium"


class TestBackwardCompat:
    """Old-style register() calls must not break."""

    def test_register_without_new_params(self):
        reg = _make_registry()
        reg.register(
            name="read_file", toolset="core",
            schema={"description": "read"}, handler=_dummy_handler,
        )
        assert "read_file" in reg.get_all_tool_names()

    def test_defaults_correct(self):
        reg = _make_registry()
        reg.register(
            name="t1", toolset="core",
            schema={"description": "test"}, handler=_dummy_handler,
        )
        meta = reg.get_metadata("t1")
        assert meta["mutates_local_fs"] is False
        assert meta["mutates_agent_state"] is False
        assert meta["mutates_browser_session"] is False
        assert meta["mutates_external_world"] is False
        assert meta["requires_confirmation_default"] is False
        assert meta["allowed_in_plan_mode_default"] is False
        assert meta["parallel_safe_default"] is False
        assert meta["risk_level"] == "low"
        assert meta["deferred"] is False
        assert meta["always_load"] is False
        assert meta["search_hint"] == ""

    def test_schema_not_affected(self):
        reg = _make_registry()
        reg.register(
            name="t1", toolset="core",
            schema={"description": "test", "parameters": {}},
            handler=_dummy_handler,
        )
        defs = reg.get_definitions({"t1"})
        assert len(defs) == 1
        fn = defs[0]["function"]
        # Metadata fields must NOT leak into schema
        assert "mutates_local_fs" not in fn
        assert "risk_level" not in fn


class TestCustomMetadata:
    """Custom metadata values are stored and retrievable."""

    def test_custom_values(self):
        reg = _make_registry()
        reg.register(
            name="write_file", toolset="core",
            schema={"description": "write"}, handler=_dummy_handler,
            mutates_local_fs=True, risk_level="medium",
            requires_confirmation_default=True,
        )
        meta = reg.get_metadata("write_file")
        assert meta["mutates_local_fs"] is True
        assert meta["risk_level"] == "medium"
        assert meta["requires_confirmation_default"] is True
        # Other defaults still hold
        assert meta["mutates_external_world"] is False

    def test_deferred_and_search_hint(self):
        reg = _make_registry()
        reg.register(
            name="image_gen", toolset="media",
            schema={"description": "gen"}, handler=_dummy_handler,
            deferred=True, search_hint="Generate images with AI",
        )
        meta = reg.get_metadata("image_gen")
        assert meta["deferred"] is True
        assert meta["search_hint"] == "Generate images with AI"

    def test_unknown_tool_returns_empty(self):
        reg = _make_registry()
        assert reg.get_metadata("nonexistent") == {}

    def test_all_eleven_fields_present(self):
        reg = _make_registry()
        reg.register(
            name="t", toolset="x",
            schema={"description": "x"}, handler=_dummy_handler,
        )
        meta = reg.get_metadata("t")
        expected_keys = {
            "mutates_local_fs", "mutates_agent_state",
            "mutates_browser_session", "mutates_external_world",
            "requires_confirmation_default", "allowed_in_plan_mode_default",
            "parallel_safe_default", "risk_level",
            "deferred", "always_load", "search_hint",
        }
        assert set(meta.keys()) == expected_keys
