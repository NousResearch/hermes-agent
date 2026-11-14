"""A one-shot ``hermes chat -q/-Q`` run lingers for notify_on_complete processes only
within ONE bounded budget (``terminal.oneshot_completion_wait_seconds``) after the turn
ends. The background spawn result must state that real bound, so an agent never ends
its turn expecting a notification that the run will exit before delivering.
"""

import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars


@pytest.fixture
def spawned():
    from tools.process_registry import process_registry

    ids = []
    process_registry.pending_watchers = []
    yield ids
    for sid in ids:
        session = process_registry.get(sid)
        if session is not None:
            # Teardown kill is not a completion: never leave one in the shared queue.
            session.notify_on_complete = False
        try:
            process_registry.kill_process(sid)
        except Exception:
            pass
    process_registry.pending_watchers = []


def _run_bg(spawned, **kwargs):
    from tools.terminal_tool import terminal_tool

    d = json.loads(terminal_tool(command="sleep 30", background=True, **kwargs))
    if d.get("session_id"):
        spawned.append(d["session_id"])
    return d


def _linger(monkeypatch, seconds):
    from tools.process_registry import ProcessRegistry

    monkeypatch.setattr(ProcessRegistry, "_oneshot_completion_wait_seconds", staticmethod(lambda: seconds))


def test_single_query_notify_states_the_real_linger_bound(monkeypatch, spawned):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _linger(monkeypatch, 600.0)

    d = _run_bg(spawned, notify_on_complete=True)

    assert d.get("notify_on_complete") is True
    note = d.get("oneshot_note")
    assert note, "a one-shot run must state its completion-linger bound"
    assert "600s" in note
    assert "terminal.oneshot_completion_wait_seconds" in note
    assert "process(action='wait')" in note


def test_single_query_note_follows_the_configured_value(monkeypatch, spawned):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _linger(monkeypatch, 90.0)

    note = _run_bg(spawned, notify_on_complete=True).get("oneshot_note") or ""

    assert "90s" in note
    assert "600s" not in note


def test_single_query_with_linger_disabled_says_it_will_not_wait(monkeypatch, spawned):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _linger(monkeypatch, 0.0)

    note = _run_bg(spawned, notify_on_complete=True).get("oneshot_note") or ""

    assert "without waiting" in note
    assert "process(action='wait')" in note


def test_interactive_session_gets_no_oneshot_note(monkeypatch, spawned):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    _linger(monkeypatch, 600.0)

    d = _run_bg(spawned, notify_on_complete=True)

    assert d.get("notify_on_complete") is True
    assert "oneshot_note" not in d


def test_single_query_without_notify_gets_no_oneshot_note(monkeypatch, spawned):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _linger(monkeypatch, 600.0)

    assert "oneshot_note" not in _run_bg(spawned)


def test_stateless_one_shot_keeps_the_unsupported_note_only(monkeypatch, spawned):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _linger(monkeypatch, 600.0)
    tokens = set_session_vars(platform="api_server", chat_id="s1", session_key="s1", async_delivery=False)
    try:
        d = _run_bg(spawned, notify_on_complete=True)
    finally:
        clear_session_vars(tokens)

    assert d.get("notify_on_complete") is False
    assert d.get("notify_unsupported")
    assert "oneshot_note" not in d
