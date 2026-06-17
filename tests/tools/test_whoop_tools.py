from __future__ import annotations

import json

import pytest

from plugins.whoop import register as whoop_register
from plugins.whoop import tools as whoop_tool


def test_whoop_profile_tool_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def get_profile(self):
            return {"user_id": 99, "email": "test@example.com", "first_name": "Test"}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_profile({}))
    assert result["user_id"] == 99


def test_whoop_cycles_list_passes_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict] = []

    class _StubClient:
        def list_cycles(self, **kw):
            seen.append(kw)
            return {"records": [], "next_token": None}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    json.loads(whoop_tool._handle_whoop_cycles({
        "action": "list",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T00:00:00Z",
        "limit": 10,
        "nextToken": "abc",
        "max_pages": 2,
    }))
    assert seen
    assert seen[0].get("start") == "2024-01-01T00:00:00Z"
    assert seen[0].get("limit") == 10
    assert seen[0].get("next_token") == "abc"
    assert seen[0].get("max_pages") == 2


def test_whoop_recovery_latest_returns_first_record(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def list_recovery(self, **kw):
            return {"records": [{"score": {"recovery_score": 85}}], "next_token": None}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_recovery({"action": "latest"}))
    assert result["score"]["recovery_score"] == 85


def test_whoop_recovery_missing_action_defaults_to_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def list_recovery(self, **kw):
            return {"records": [{"score": {"recovery_score": 72}}], "next_token": None}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_recovery({}))
    assert result["score"]["recovery_score"] == 72


def test_whoop_sleep_get_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _StubClient:
        def get_sleep(self, sleep_id):
            seen.append(sleep_id)
            return {"id": sleep_id, "start": "2024-01-01T22:00:00Z"}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_sleep({"action": "get", "id": "42"}))
    assert seen == ["42"]
    assert result["id"] == "42"


def test_whoop_sleep_get_requires_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        pass

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_sleep({"action": "get"}))
    assert "error" in result or "id" in str(result).lower()


def test_whoop_workouts_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def list_workouts(self, **kw):
            return {"records": [{"id": 1, "sport_id": 0}], "next_token": None}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_workouts({"action": "list"}))
    assert result["records"][0]["id"] == 1


def test_whoop_tools_gated_when_not_authenticated() -> None:
    """_check_whoop_available returns False without auth state (isolated HERMES_HOME from conftest)."""
    assert whoop_tool._check_whoop_available() is False


def test_whoop_plugin_registers_five_tools() -> None:
    calls: list[dict] = []

    class _FakeCtx:
        def register_tool(self, **kw):
            calls.append(kw)

    whoop_register(_FakeCtx())
    expected = {"whoop_profile", "whoop_cycles", "whoop_recovery", "whoop_sleep", "whoop_workouts"}
    assert {call["name"] for call in calls} == expected
    assert {call["toolset"] for call in calls} == {"whoop"}
    assert all(call["check_fn"] is whoop_tool._check_whoop_available for call in calls)


def test_whoop_cycles_latest_returns_first_record(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def list_cycles(self, **kw):
            return {"records": [{"id": 10, "score": {"strain": 12.5}}], "next_token": None}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_cycles({"action": "latest"}))
    assert result["id"] == 10


def test_whoop_workouts_get_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _StubClient:
        def get_workout(self, workout_id):
            seen.append(workout_id)
            return {"id": workout_id, "sport_id": 71}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_workouts({"action": "get", "id": "99"}))
    assert seen == ["99"]
    assert result["sport_id"] == 71


def test_whoop_tool_unknown_action_returns_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        pass

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_cycles({"action": "dance"}))
    assert "error" in result


def test_whoop_limit_and_max_pages_are_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict] = []

    class _StubClient:
        def list_sleep(self, **kw):
            seen.append(kw)
            return {"records": []}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    json.loads(whoop_tool._handle_whoop_sleep({"action": "list", "limit": 999, "max_pages": 99}))
    assert seen[0]["limit"] == 25
    assert seen[0]["max_pages"] == 10

def test_whoop_recovery_get_uses_cycle_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _StubClient:
        def get_recovery(self, cycle_id):
            seen.append(cycle_id)
            return {"cycle_id": cycle_id, "score": {"recovery_score": 88}}

    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: _StubClient())
    result = json.loads(whoop_tool._handle_whoop_recovery({"action": "get", "cycle_id": "abc"}))
    assert seen == ["abc"]
    assert result["score"]["recovery_score"] == 88


def test_whoop_handler_returns_tool_error_when_client_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whoop_tool, "_whoop_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    result = json.loads(whoop_tool._handle_whoop_cycles({"action": "latest"}))
    assert "error" in result
    assert "boom" in result["error"]
