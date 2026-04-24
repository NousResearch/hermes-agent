import json
import sys
from types import SimpleNamespace

from tools.self_state_tool import self_state_tool
from tools.registry import registry
from toolsets import resolve_toolset


def test_self_state_lists_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "agent:main:discord:group:123": {
            "session_id": "20260423_120000_abcd",
            "platform": "discord",
            "chat_type": "group",
            "display_name": "#general",
            "updated_at": "2026-04-23T12:03:00",
            "created_at": "2026-04-23T12:00:00",
            "origin": {
                "platform": "discord",
                "chat_id": "123",
                "chat_name": "#general",
                "user_id": "u1",
                "user_name": "hands",
            },
        }
    }), encoding="utf-8")

    result = self_state_tool(action="sessions")

    assert result["action"] == "sessions"
    assert result["sessions"][0]["session_id"] == "20260423_120000_abcd"
    assert result["sessions"][0]["platform"] == "discord"
    assert result["sessions"][0]["chat_name"] == "#general"


def test_self_state_lists_local_crons(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    (cron_dir / "jobs.json").write_text(json.dumps({
        "jobs": [{
            "id": "job-1",
            "name": "Market monitor",
            "enabled": True,
            "deliver": "origin",
            "schedule": {"display": "every 10m"},
            "origin": {"platform": "hub", "chat_id": "hub:sal"},
            "next_run": "2026-04-23T12:10:00",
        }]
    }), encoding="utf-8")

    result = self_state_tool(action="crons")

    assert result["action"] == "crons"
    assert result["crons"][0]["id"] == "job-1"
    assert result["crons"][0]["deliver"] == "origin"
    assert result["crons"][0]["origin"]["chat_id"] == "hub:sal"


def test_self_state_recent_activity_uses_session_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "agent:main:hub:dm:hub:codex-cron": {
            "session_id": "sid-1",
            "platform": "hub",
            "chat_type": "dm",
            "updated_at": "2026-04-23T12:03:00",
            "origin": {
                "platform": "hub",
                "chat_id": "hub:codex-cron",
                "user_id": "codex-cron",
                "user_name": "codex-cron",
            },
        }
    }), encoding="utf-8")

    class FakeDB:
        def list_sessions_rich(self, **_kwargs):
            return [{"id": "sid-1", "source": "hub", "last_active": 1}]

        def get_messages(self, session_id):
            assert session_id == "sid-1"
            return [
                {
                    "timestamp": 1,
                    "role": "assistant",
                    "content": "Posting an update to #general",
                    "tool_calls": [{"function": {"name": "send_message"}}],
                }
            ]

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "hermes_state", SimpleNamespace(SessionDB=FakeDB))

    result = self_state_tool(action="recent_activity", session_filter="codex-cron")

    assert result["activity"][0]["session_id"] == "sid-1"
    assert result["activity"][0]["source"] == "hub:codex-cron"
    assert result["activity"][0]["tool_calls"] == ["send_message"]


def test_self_state_registry_dispatch_returns_json_string(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sessions.json").write_text(json.dumps({}), encoding="utf-8")

    result = registry.dispatch("self_state", {"action": "sessions"})

    assert isinstance(result, str)
    assert json.loads(result)["action"] == "sessions"


def test_self_state_is_in_core_toolsets():
    assert "self_state" in resolve_toolset("hermes-cli")
    assert "self_state" in resolve_toolset("hermes-discord")
