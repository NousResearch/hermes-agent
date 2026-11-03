"""Regression: message_agent delivery children must survive the sender's turn-end
lifecycle sweep. The delivery runner is spawned via terminal_tool; before the fix it
lacked persist_on_release=True, so ClientLifecycleMixin._close_task_resources →
kill_processes(source="agent_close") SIGTERM'd the runner mid-boot and the DM died
with termination_source=agent_close (observed live 2026-10-01 12:36:38)."""
import json

from tests.tools.test_bot_mode_dm import _capture_spawn, _FakeAgent, _managed_home


def test_spawn_delivery_sets_persist_on_release(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    import tools.bot_relay as bot_relay
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    from tools import bot_mode_dm

    home = _managed_home(tmp_path, teammates=("researcher",))
    agent = _FakeAgent(home, title="Bot Chat")

    out = json.loads(bot_mode_dm.message_agent_tool(
        target="@researcher", message="ping", agent=agent))

    assert out["status"] == "queued", out
    assert calls, "no terminal_tool call captured"
    assert all(c.get("persist_on_release") is True for c in calls), calls
