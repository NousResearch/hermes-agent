"""Workstation Browser schema capability is owned by the Desktop session."""

import pytest

import model_tools
from gateway.session_context import clear_session_vars, set_session_vars
from tools import browser_workstation
from tools.registry import registry

TARGET = "browser_navigate"
EXTENSION_TOOL = "browser_extension_install"


def _tool_names(platform: str, source: str) -> set[str]:
    tokens = set_session_vars(platform=platform, source=source, session_id=f"schema-{platform}")
    try:
        return {
            item["function"]["name"]
            for item in model_tools.get_tool_definitions(
                enabled_toolsets=["browser"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def clean_schema_cache(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.setattr(browser_workstation, "_browser_config", lambda: {"enabled": True, "routing_enabled": True})
    model_tools._clear_tool_defs_cache()
    yield
    model_tools._clear_tool_defs_cache()


def _force_probe_false(monkeypatch):
    entry = registry.get_entry(TARGET)
    assert entry is not None
    monkeypatch.setattr(entry, "check_fn", lambda: False)
    monkeypatch.setattr(model_tools, "validate_toolset", lambda name: name == "browser")
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda name: [TARGET] if name == "browser" else [])


def test_desktop_surface_preserves_browser_schema_when_probe_is_false(monkeypatch):
    _force_probe_false(monkeypatch)
    assert TARGET in _tool_names("desktop", "desktop")
    assert EXTENSION_TOOL in _tool_names("desktop", "desktop")


def test_desktop_schema_cache_does_not_leak_into_tui(monkeypatch):
    _force_probe_false(monkeypatch)
    assert TARGET in _tool_names("desktop", "desktop")
    assert TARGET not in _tool_names("tui", "tui")
    assert EXTENSION_TOOL not in _tool_names("tui", "tui")


def test_process_desktop_env_does_not_grant_tui_browser_capability(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    tokens = set_session_vars(platform="tui", source="tui", session_id="schema-tui-env")
    try:
        assert browser_workstation.workstation_schema_tools_for_current_session() == set()
    finally:
        clear_session_vars(tokens)


def test_desktop_source_grants_capability_without_process_env():
    tokens = set_session_vars(platform="desktop", source="desktop", session_id="schema-desktop-no-env")
    try:
        tools = browser_workstation.workstation_schema_tools_for_current_session()
        assert TARGET in tools
        assert "browser_snapshot" in tools
        assert EXTENSION_TOOL in tools
        assert "browser_exec" not in tools
    finally:
        clear_session_vars(tokens)


def test_desktop_surface_preserves_workstation_browser_tools_when_toolset_omitted_from_enabled(monkeypatch):
    """When enabled_toolsets lacks 'browser' (e.g. custom platform_toolsets.cli or focus posture),
    a Desktop session still receives Workstation Browser tools via surface capability,
    while a TUI session strictly does not.
    """
    tokens_desktop = set_session_vars(platform="desktop", source="desktop", session_id="schema-desktop-omitted")
    try:
        tools = {
            item["function"]["name"]
            for item in model_tools.get_tool_definitions(
                enabled_toolsets=["terminal", "file"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
        assert TARGET in tools
        assert "browser_snapshot" in tools
        assert EXTENSION_TOOL in tools
    finally:
        clear_session_vars(tokens_desktop)

    tokens_tui = set_session_vars(platform="tui", source="tui", session_id="schema-tui-omitted")
    try:
        tools_tui = {
            item["function"]["name"]
            for item in model_tools.get_tool_definitions(
                enabled_toolsets=["terminal", "file"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
        assert TARGET not in tools_tui
        assert "browser_snapshot" not in tools_tui
        assert EXTENSION_TOOL not in tools_tui
    finally:
        clear_session_vars(tokens_tui)
