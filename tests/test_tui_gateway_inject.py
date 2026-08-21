"""External message injection reaches dashboard/desktop sessions.

``PluginContext.inject_message`` used to be CLI-only: in a headless serving
process (``hermes serve`` / the desktop app) ``_cli_ref`` is ``None``, so a
plugin's completion report was logged as a warning and dropped — dispatch
watchers finished work nobody ever heard about. The gateway now exposes
``tui_gateway.server.inject_external_message``, and ``inject_message`` routes
to it when no interactive CLI is attached: busy sessions get the message
merged into their queued next-turn prompt (never interrupting in-flight
work), idle sessions start a relay turn immediately.
"""

import threading
import time
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


# ── inject_external_message ────────────────────────────────────────────────

def test_idle_session_fires_a_turn_immediately(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid
        fired["text"] = text

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    session = _session(last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert _wait_until(lambda: fired.get("text") == "run finished")
    assert fired["sid"] == "sid-1"


def test_busy_session_queues_without_interrupting(monkeypatch):
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent, running=True, last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert session["queued_prompt"]["text"] == "run finished"
    assert calls["interrupt"] == 0


def test_busy_session_merges_behind_queued_user_prompt(monkeypatch):
    session = _session(running=True, last_active=100.0)
    server._enqueue_prompt(session, "user typed this", "ws-1")
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert session["queued_prompt"]["text"] == "user typed this\n\nrun finished"
    # The user's transport binding survives the merge.
    assert session["queued_prompt"]["transport"] == "ws-1"


def test_most_recently_active_session_wins(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    monkeypatch.setattr(server, "_sessions", {
        "old": _session(last_active=10.0),
        "new": _session(last_active=200.0),
    })

    assert server.inject_external_message("report") is True
    assert _wait_until(lambda: fired.get("sid") == "new")


def test_target_session_beats_more_recent_activity(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    monkeypatch.setattr(server, "_sessions", {
        "origin": _session(last_active=10.0),
        "chatty": _session(last_active=200.0),
    })

    assert server.inject_external_message("report", target_sid="origin") is True
    assert _wait_until(lambda: fired.get("sid") == "origin")


def test_closed_target_session_drops_instead_of_rerouting(monkeypatch):
    other = _session(last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"other": other})

    assert server.inject_external_message("report", target_sid="gone") is False
    assert other.get("queued_prompt") is None


def test_lazy_target_session_drops_instead_of_rerouting(monkeypatch):
    other = _session(last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {
        "watch": _session(lazy=True, last_active=10.0),
        "other": other,
    })

    assert server.inject_external_message("report", target_sid="watch") is False
    assert other.get("queued_prompt") is None


def test_lazy_watch_sessions_are_skipped(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {
        "watch": _session(lazy=True, last_active=200.0),
    })
    assert server.inject_external_message("report") is False


def test_no_sessions_returns_false(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    assert server.inject_external_message("report") is False


def test_blank_text_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {"sid-1": _session(last_active=1.0)})
    assert server.inject_external_message("   ") is False
    assert server.inject_external_message(None) is False


# ── PluginContext.inject_message routing ───────────────────────────────────

def _plugin_ctx():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    pm = PluginManager()
    manifest = PluginManifest(
        name="testplugin", version="1.0.0", description="test", source="user",
    )
    return PluginContext(manifest, pm)


def test_inject_message_routes_to_gateway_when_no_cli(monkeypatch):
    session = _session(running=True, last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    ctx = _plugin_ctx()
    assert ctx._manager._cli_ref is None
    assert ctx.inject_message("run finished") is True
    assert session["queued_prompt"]["text"] == "run finished"


def test_inject_message_session_id_targets_origin_session(monkeypatch):
    origin = _session(running=True, last_active=10.0)
    chatty = _session(running=True, last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"origin": origin, "chatty": chatty})

    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished", session_id="origin") is True
    assert origin["queued_prompt"]["text"] == "run finished"
    assert chatty.get("queued_prompt") is None


def test_inject_message_session_id_gone_drops(monkeypatch):
    chatty = _session(running=True, last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"chatty": chatty})

    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished", session_id="gone") is False
    assert chatty.get("queued_prompt") is None


def test_inject_message_still_false_without_cli_or_sessions(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished") is False


def test_inject_message_prefers_cli_when_attached(monkeypatch):
    session = _session(last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    class _Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    cli = types.SimpleNamespace(
        _agent_running=False, _pending_input=_Queue(), _interrupt_queue=_Queue(),
    )
    ctx = _plugin_ctx()
    ctx._manager._cli_ref = cli

    assert ctx.inject_message("hello") is True
    assert cli._pending_input.items == ["hello"]
    assert session.get("queued_prompt") is None
