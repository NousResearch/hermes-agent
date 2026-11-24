"""Gateway completion watchers are armed at launch, not only after the turn ends.

A ``notify_on_complete`` process that finishes while its launching turn is still running
used to sit in ``pending_watchers`` until the post-turn drain, leaving the chat mute for
the whole turn (#112033).
"""

import pytest
from types import SimpleNamespace

from tools.process_registry import ProcessRegistry
from tools.terminal_tool_background import _register_completion_watcher


def _proc_session():
    return SimpleNamespace(
        id="proc_active", watcher_platform="telegram", watcher_chat_id="123",
        watcher_user_id="u", watcher_user_name="Ada", watcher_thread_id="",
        watcher_message_id="", parent_session_id="",
    )


def _install_runner(monkeypatch, runner):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)


def test_live_gateway_arms_watcher_at_registration(monkeypatch):
    armed = []
    _install_runner(monkeypatch, SimpleNamespace(arm_process_watcher=lambda w: armed.append(w) or True))
    registry = ProcessRegistry()

    _register_completion_watcher(registry, _proc_session(), "agent:main:telegram:dm:123")

    assert [w["session_id"] for w in armed] == ["proc_active"]
    assert armed[0]["notify_on_complete"] is True and armed[0]["chat_id"] == "123"
    assert registry.pending_watchers == []


def test_watcher_stays_pending_without_a_serving_gateway(monkeypatch):
    """No runner (CLI-side import) and a runner that is not serving both keep the
    startup / post-turn drain path."""
    registry = ProcessRegistry()
    _install_runner(monkeypatch, None)
    _register_completion_watcher(registry, _proc_session(), "agent:main:telegram:dm:123")

    _install_runner(monkeypatch, SimpleNamespace(arm_process_watcher=lambda w: False))
    _register_completion_watcher(registry, _proc_session(), "agent:main:telegram:dm:123")

    assert [w["session_id"] for w in registry.pending_watchers] == ["proc_active", "proc_active"]


def test_a_registration_that_raises_leaves_the_no_watcher_marker(monkeypatch):
    """``watcher_interval`` is the "a watcher exists" marker callers re-arm on, and the spawn and
    yield paths call the registrar unwrapped: a raise before the descriptor is queued must not leave 5."""
    from types import SimpleNamespace

    import gateway.run as gateway_run
    from tools.terminal_tool_background import _register_completion_watcher

    def boom(_watcher):
        raise RuntimeError("loop torn down")

    _install_runner(monkeypatch, SimpleNamespace(arm_process_watcher=boom))
    session = _proc_session()
    session.watcher_interval = 0

    with pytest.raises(RuntimeError):
        _register_completion_watcher(ProcessRegistry(), session, "agent:main:telegram:dm:123")

    assert session.watcher_interval == 0
