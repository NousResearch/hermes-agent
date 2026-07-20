"""TDD coverage for the LLM-facing delegation_control tool."""

import json

import pytest


@pytest.fixture(autouse=True)
def _resume_spawning_after_test():
    from tools.delegate_tool import set_spawn_paused

    set_spawn_paused(False)
    yield
    set_spawn_paused(False)


def test_schema_exposes_all_control_actions():
    from tools.delegation_control_tool import DELEGATION_CONTROL_SCHEMA

    assert DELEGATION_CONTROL_SCHEMA["name"] == "delegation_control"
    actions = DELEGATION_CONTROL_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert actions == ["list", "status", "cancel", "pause", "resume"]


def test_list_returns_delegations_and_pause_state(monkeypatch):
    import tools.delegation_control_tool as control

    monkeypatch.setattr(
        control,
        "list_async_delegations",
        lambda **kwargs: [{"delegation_id": "deleg_1", "status": "stalled"}],
    )
    result = json.loads(control.delegation_control("list"))

    assert result["action"] == "list"
    assert result["spawn_paused"] is False
    assert result["delegations"][0]["status"] == "stalled"


def test_list_uses_configured_stalled_threshold(monkeypatch):
    import tools.delegate_tool as delegate
    import tools.delegation_control_tool as control

    captured = {}
    monkeypatch.setattr(
        delegate, "_load_config", lambda: {"stalled_after_seconds": 42}
    )
    monkeypatch.setattr(
        control,
        "list_async_delegations",
        lambda **kwargs: captured.update(kwargs) or [],
    )

    json.loads(control.delegation_control("list"))
    assert captured["stalled_after_seconds"] == 42.0


def test_status_requires_id_and_reports_unknown(monkeypatch):
    import tools.delegation_control_tool as control

    missing = json.loads(control.delegation_control("status"))
    assert "delegation_id is required" in missing["error"]

    monkeypatch.setattr(control, "get_async_delegation", lambda *args, **kwargs: None)
    unknown = json.loads(control.delegation_control("status", "deleg_missing"))
    assert unknown["status"] == "not_found"
    assert "deleg_missing" in unknown["error"]


def test_status_returns_one_record(monkeypatch):
    import tools.delegation_control_tool as control

    monkeypatch.setattr(
        control,
        "get_async_delegation",
        lambda *args, **kwargs: {"delegation_id": "deleg_1", "status": "running"},
    )
    result = json.loads(control.delegation_control("status", "deleg_1"))
    assert result["delegation"]["delegation_id"] == "deleg_1"


def test_cancel_requires_id_and_delegates_to_precise_interrupt(monkeypatch):
    import tools.delegation_control_tool as control

    missing = json.loads(control.delegation_control("cancel"))
    assert "delegation_id is required" in missing["error"]

    calls = []
    monkeypatch.setattr(
        control,
        "interrupt_async_delegation",
        lambda delegation_id, reason, owner_session_key: calls.append(
            (delegation_id, reason, owner_session_key)
        )
        or {"delegation_id": delegation_id, "status": "cancelling"},
    )
    result = json.loads(control.delegation_control("cancel", "deleg_1"))
    assert result["status"] == "cancelling"
    assert calls == [("deleg_1", "delegation_control tool", "")]


def test_pause_and_resume_are_scoped_to_current_session(monkeypatch):
    import tools.delegation_control_tool as control
    from tools.delegate_tool import is_spawn_paused

    monkeypatch.setattr(
        control, "get_current_session_key", lambda default="": "session:a"
    )
    paused = json.loads(control.delegation_control("pause"))
    assert paused == {"action": "pause", "spawn_paused": True}
    assert is_spawn_paused("session:a") is True
    assert is_spawn_paused("session:b") is False
    assert is_spawn_paused() is False

    resumed = json.loads(control.delegation_control("resume"))
    assert resumed == {"action": "resume", "spawn_paused": False}
    assert is_spawn_paused("session:a") is False


def test_control_passes_current_session_owner_to_registry(monkeypatch):
    import tools.delegation_control_tool as control

    captured = {}
    monkeypatch.setattr(
        control, "get_current_session_key", lambda default="": "session:a"
    )
    monkeypatch.setattr(
        control,
        "list_async_delegations",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    json.loads(control.delegation_control("list"))
    assert captured["owner_session_key"] == "session:a"

    monkeypatch.setattr(
        control,
        "get_async_delegation",
        lambda delegation_id, **kwargs: captured.update(kwargs)
        or {"delegation_id": delegation_id},
    )
    json.loads(control.delegation_control("status", "deleg_1"))
    assert captured["owner_session_key"] == "session:a"

    monkeypatch.setattr(
        control,
        "interrupt_async_delegation",
        lambda delegation_id, **kwargs: captured.update(kwargs)
        or {"delegation_id": delegation_id, "status": "cancelling"},
    )
    json.loads(control.delegation_control("cancel", "deleg_1"))
    assert captured["owner_session_key"] == "session:a"


def test_unknown_action_is_rejected():
    from tools.delegation_control_tool import delegation_control

    result = json.loads(delegation_control("explode"))
    assert "Unsupported action" in result["error"]


def test_tool_self_registers_and_is_in_core_toolset():
    import tools.delegation_control_tool  # noqa: F401
    from tools.registry import registry
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

    entry = registry.get_entry("delegation_control")
    assert entry is not None
    assert entry.toolset == "delegation_control"
    assert "delegation_control" in _HERMES_CORE_TOOLS
    assert TOOLSETS["delegation_control"]["tools"] == ["delegation_control"]


def test_subagents_strip_global_control_toolset():
    from tools.delegate_tool import _strip_blocked_tools

    child = _strip_blocked_tools(["terminal", "delegation_control"])
    assert child == ["terminal"]
