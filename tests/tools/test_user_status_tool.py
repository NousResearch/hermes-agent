"""Tests for tools/user_status_tool.py (cross-bot user-status mutation tool).

Issue #21122 / kanban t_315c0bfc — Task B.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def fresh_tool(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir and reload modules so storage + tool
    pick up the patched home.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    import hermes_constants
    if hasattr(hermes_constants, "_profile_fallback_warned"):
        hermes_constants._profile_fallback_warned = False

    from agent import user_status as us_mod
    importlib.reload(us_mod)

    from tools import user_status_tool as ust_mod
    importlib.reload(ust_mod)

    return tmp_path, us_mod, ust_mod


def _state_file(home: Path) -> Path:
    return home / "state" / "user_status.json"


# ---------------------------------------------------------------------------
# get on empty file
# ---------------------------------------------------------------------------


def test_get_on_empty_file_returns_default_state(fresh_tool):
    home, _us, ust = fresh_tool
    assert not _state_file(home).exists()

    out = ust.get_user_status()
    data = json.loads(out)

    # All user-writable fields default to None; metadata fields are empty.
    assert data["device_mode"] is None
    assert data["afk_status"] is None
    assert data["focus_project"] is None
    assert data["quiet_hours_until"] is None
    assert data["location"] is None
    assert data["per_field_updated_at"] == {}
    assert data["updated_by"] is None


def test_dispatch_get_action(fresh_tool):
    _home, _us, ust = fresh_tool
    out = ust.user_status_tool(action="get")
    data = json.loads(out)
    assert "device_mode" in data


# ---------------------------------------------------------------------------
# set then get
# ---------------------------------------------------------------------------


def test_set_then_get_roundtrip(fresh_tool):
    _home, _us, ust = fresh_tool

    set_out = ust.set_user_status("device_mode", "mobile", writer="telegram")
    set_data = json.loads(set_out)
    assert set_data["device_mode"] == "mobile"
    assert set_data["updated_by"] == "telegram"
    assert "device_mode" in set_data["per_field_updated_at"]

    get_out = ust.get_user_status()
    get_data = json.loads(get_out)
    assert get_data["device_mode"] == "mobile"
    assert get_data["updated_by"] == "telegram"
    # other fields preserved as None
    assert get_data["afk_status"] is None


def test_set_preserves_other_fields(fresh_tool):
    _home, _us, ust = fresh_tool

    ust.set_user_status("device_mode", "desktop", writer="discord")
    ust.set_user_status("focus_project", "hermes", writer="discord")

    data = json.loads(ust.get_user_status())
    assert data["device_mode"] == "desktop"
    assert data["focus_project"] == "hermes"
    assert "device_mode" in data["per_field_updated_at"]
    assert "focus_project" in data["per_field_updated_at"]


# ---------------------------------------------------------------------------
# unknown field rejected
# ---------------------------------------------------------------------------


def test_set_unknown_field_rejected(fresh_tool):
    _home, _us, ust = fresh_tool

    out = ust.set_user_status("bogus_field", "x", writer="telegram")
    data = json.loads(out)
    assert "error" in data
    assert "bogus_field" in data["error"]
    assert "unknown" in data["error"].lower()
    # File should not have been created with bogus content.
    # (storage layer only writes on the happy path; a get should still
    # return defaults.)
    follow = json.loads(ust.get_user_status())
    assert follow["updated_by"] is None


def test_set_reserved_metadata_field_rejected(fresh_tool):
    _home, _us, ust = fresh_tool
    out = ust.set_user_status("updated_by", "evil", writer="telegram")
    data = json.loads(out)
    assert "error" in data
    assert "reserved" in data["error"].lower() or "unknown" in data["error"].lower()


def test_dispatch_set_without_field_errors(fresh_tool):
    _home, _us, ust = fresh_tool
    out = ust.user_status_tool(action="set", value="x")
    data = json.loads(out)
    assert "error" in data
    assert "field" in data["error"].lower()


def test_dispatch_unknown_action_errors(fresh_tool):
    _home, _us, ust = fresh_tool
    out = ust.user_status_tool(action="frobnicate")
    data = json.loads(out)
    assert "error" in data
    assert "frobnicate" in data["error"]


# ---------------------------------------------------------------------------
# writer attribution
# ---------------------------------------------------------------------------


def test_writer_attribution_explicit(fresh_tool):
    _home, _us, ust = fresh_tool
    out = ust.set_user_status("location", "PR", writer="slack")
    assert json.loads(out)["updated_by"] == "slack"


def test_writer_attribution_defaults_to_hermes_profile_env(fresh_tool, monkeypatch):
    _home, _us, ust = fresh_tool
    monkeypatch.setenv("HERMES_PROFILE", "telegram-prof")
    out = ust.set_user_status("afk_status", "lunch")
    data = json.loads(out)
    assert data["updated_by"] == "telegram-prof"


def test_writer_attribution_falls_back_to_agent(fresh_tool, monkeypatch):
    _home, _us, ust = fresh_tool
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    out = ust.set_user_status("afk_status", "deep-work")
    data = json.loads(out)
    assert data["updated_by"] == "agent"


def test_writer_explicit_overrides_env(fresh_tool, monkeypatch):
    _home, _us, ust = fresh_tool
    monkeypatch.setenv("HERMES_PROFILE", "from-env")
    out = ust.set_user_status("location", "office", writer="explicit-tag")
    assert json.loads(out)["updated_by"] == "explicit-tag"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_tool_is_registered(fresh_tool):
    _home, _us, ust = fresh_tool
    from tools.registry import registry
    entry = registry.get_entry("user_status")
    assert entry is not None
    assert entry.toolset == "user_status"
    assert callable(entry.handler)


def test_tool_dispatch_via_registry_get(fresh_tool):
    _home, _us, _ust = fresh_tool
    from tools.registry import registry
    out = registry.dispatch("user_status", {"action": "get"})
    data = json.loads(out)
    assert "device_mode" in data


def test_tool_dispatch_via_registry_set(fresh_tool):
    _home, _us, _ust = fresh_tool
    from tools.registry import registry
    out = registry.dispatch(
        "user_status",
        {"action": "set", "field": "device_mode", "value": "mobile", "writer": "tg"},
    )
    data = json.loads(out)
    assert data["device_mode"] == "mobile"
    assert data["updated_by"] == "tg"


def test_user_status_in_core_toolset():
    """Confirm the tool is wired into _HERMES_CORE_TOOLS so every profile
    gets it automatically."""
    import importlib
    import toolsets as ts_mod
    importlib.reload(ts_mod)
    assert "user_status" in ts_mod._HERMES_CORE_TOOLS
