"""``reload.all`` in the TUI sidecar (``/reload``, #159205): consent first, refused while the chat's turn
runs, otherwise .env → plugins → MCP → skills and one ``preserve_prefix=False`` rebuild of the live
sessions' tools, all under ``_mcp_reload_lock`` (a concurrent ``reload.mcp`` must never tear the
registry down mid-pass; the plugin activation refresh raced it without the lock, #119861)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


@pytest.fixture()
def sidecar(monkeypatch):
    order, locked, refreshes, notes = [], [], [], []
    agent = SimpleNamespace(name="A", tools=[_tool("terminal")])

    def step(name, result=None):
        def run(*_a, **_k):
            order.append(name)
            locked.append(srv._mcp_reload_lock.locked())
            return result
        return run
    monkeypatch.setattr("hermes_cli.config.reload_env", step("env", 2))
    monkeypatch.setattr("hermes_cli.plugins_activation.rediscover_plugins",
                        step("plugins", {"loaded": ["probe"], "changed": ["probe"], "prompt": ["probe"]}))
    monkeypatch.setattr("hermes_cli.chat_reload.reload_mcp_servers",
                        step("mcp", {"added": [], "removed": [], "reconnected": ["docs"], "failed": []}))
    monkeypatch.setattr("agent.skill_commands.reload_skills", step("skills", {"added": [], "removed": [], "total": 3}))

    def refresh(home=None, *, preserve_prefix=False, plugin_note=None):
        order.append("refresh")
        locked.append(srv._mcp_reload_lock.locked())
        refreshes.append(preserve_prefix)
        agent.tools = [_tool("terminal"), _tool("probe_tool")]
    monkeypatch.setattr(srv, "_refresh_live_sessions", refresh)
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: "rev-r")
    monkeypatch.setattr(srv, "_mcp_reload_gen", 0)
    monkeypatch.setattr(srv, "_mcp_reload_loaded_rev", "")
    monkeypatch.setattr(srv, "_mcp_reload_confirm_required", lambda: True)
    monkeypatch.setattr("agent.plugin_live_notes.queue_plugin_notice",
                        lambda target, key, note: notes.append((target, key, note)))
    session = {"agent": agent, "history": [], "history_lock": threading.RLock(), "running": False,
               "profile_home": None}
    monkeypatch.setattr(srv, "_sessions", {"A": session})
    return SimpleNamespace(order=order, locked=locked, refreshes=refreshes, notes=notes, agent=agent,
                           session=session)


def test_reload_all_asks_first_while_the_gate_is_on(sidecar):
    resp = srv._methods["reload.all"](1, {"session_id": "A"})
    assert resp["result"]["status"] == "confirm_required"
    assert "/reload now" in resp["result"]["message"] and "/reload always" in resp["result"]["message"]
    assert sidecar.order == []


def test_reload_all_runs_every_step_under_the_mcp_reload_lock_and_rebuilds_tools(sidecar):
    resp = srv._methods["reload.all"](1, {"session_id": "A", "confirm": True})
    result = resp["result"]
    assert result["status"] == "reloaded", resp
    assert sidecar.order == ["env", "plugins", "mcp", "skills", "refresh"]
    assert all(sidecar.locked), "every step runs while _mcp_reload_lock is held"
    assert sidecar.refreshes == [False]  # the consented rebuild publishes schema changes
    assert not srv._mcp_reload_lock.locked()
    assert result["report"]["tools_added"] == ["probe_tool"]
    assert "changed: probe" in result["output"] and "next session: probe" in result["output"]
    # One model note, queued for the chat's next tool result or turn.
    assert [(target, key) for target, key, _ in sidecar.notes] == [(sidecar.agent, "reload")]
    assert "probe_tool" in sidecar.notes[0][2]
    assert srv._mcp_reload_loaded_rev == "rev-r" and srv._mcp_reload_gen == 1


def test_reload_all_is_refused_while_the_chat_runs_a_turn(sidecar):
    sidecar.session["running"] = True
    resp = srv._methods["reload.all"](1, {"session_id": "A", "confirm": True})
    assert resp["result"]["status"] == "busy" and "nothing was reloaded" in resp["result"]["message"]
    assert sidecar.order == []
