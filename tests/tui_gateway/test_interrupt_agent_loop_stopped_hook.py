"""The TUI/desktop ``session.interrupt`` path is a sibling of the gateway's
``_interrupt_and_clear_session``: when a live turn is stopped, plugins holding
per-turn external resources need the same ``agent_loop_stopped`` signal.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_session(running: bool, profile_home: str | None = None) -> dict:
    session = {
        "history_lock": threading.Lock(),
        "running": running,
        "queued_prompt": None,
        "session_key": "agent:main:tui:dm:s1",
        "agent": MagicMock(),
        "_run_thread": None,
    }
    if profile_home is not None:
        session["profile_home"] = profile_home
    return session


def _hook_calls(mock_invoke_hook):
    return [
        call
        for call in mock_invoke_hook.call_args_list
        if call.args and call.args[0] == "agent_loop_stopped"
    ]


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupt_running_turn_fires_agent_loop_stopped(mock_invoke_hook):
    from tui_gateway import server

    session = _make_session(running=True)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    calls = _hook_calls(mock_invoke_hook)
    assert len(calls) == 1
    assert calls[0].kwargs == {
        "session_key": "agent:main:tui:dm:s1",
        "platform": "tui",
        "reason": "user_stop",
        "invalidation_reason": "session_interrupt",
        # Explicit target-profile identity (#125063): same home the dispatch scope binds.
        "profile_home": calls[0].kwargs.get("profile_home"),
    }
    assert calls[0].kwargs["profile_home"]  # resolved, non-empty


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupt_idle_session_does_not_fire_hook(mock_invoke_hook):
    """No live turn -> nothing for a plugin to cancel -> no hook noise."""
    from tui_gateway import server

    session = _make_session(running=False)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    assert _hook_calls(mock_invoke_hook) == []


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_failure_does_not_break_interrupt(mock_invoke_hook):
    """A misbehaving plugin must never prevent the interrupt itself."""
    mock_invoke_hook.side_effect = RuntimeError("plugin exploded")
    from tui_gateway import server

    session = _make_session(running=True)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    # The cancel flag was still set despite the hook blowing up.
    assert session["_turn_cancel_requested"] is True


def test_interrupt_hook_runs_in_selected_profile_scope(tmp_path, monkeypatch):
    """#125063: a multiplexed backend must dispatch the hook in the SELECTED session's
    profile scope, not the backend's launch profile. Two profile stores sharing one durable
    session key is exactly the ambiguity the explicit ``profile_home`` field resolves."""
    from hermes_cli import plugins
    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from tui_gateway import server

    launch = tmp_path / "launch"
    selected = tmp_path / "selected"
    launch.mkdir()
    selected.mkdir()
    token = set_hermes_home_override(
        launch
    )  # what the backend's launch profile looks like

    seen: list[tuple[str, dict]] = []
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    ctx = plugins.PluginContext(
        plugins.PluginManifest(name="profile-observer"), manager
    )
    ctx.register_hook(
        "agent_loop_stopped",
        lambda **event: seen.append((str(get_hermes_home().resolve()), event)),
    )

    session = {
        "profile_home": str(selected),
        "session_key": "same-id",  # exists in BOTH stores — the collision case
        "history_lock": threading.Lock(),
        "running": True,
        "agent": MagicMock(),
        "_sid": "selected-tab",
        "_run_thread": None,
    }
    try:
        with patch.object(server, "_clear_pending"):
            server._interrupt_session_turn("selected-tab", session)
        # The scope was released: the observer's selected-profile binding did not leak past
        # the dispatch (checked BEFORE the finally resets the outer launch binding).
        assert str(get_hermes_home().resolve()) == str(launch)
    finally:
        reset_hermes_home_override(token)

    assert len(seen) == 1
    home, event = seen[0]
    assert home == str(selected)  # observer read the selected profile, not launch
    assert event["profile_home"] == str(selected)
    assert event["session_key"] == "same-id"


def test_interrupt_hook_profile_field_hidden_from_narrow_callbacks(
    tmp_path, monkeypatch
):
    """Legacy narrow signatures declared before profile_home existed keep receiving only
    the four original fields (dispatch slices kwargs per signature)."""
    from hermes_cli import plugins
    from tui_gateway import server

    launch = tmp_path / "launch"
    launch.mkdir()
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="narrow-observer"), manager)

    narrow_seen: list[dict] = []

    def legacy_observer(
        session_key, platform, reason, invalidation_reason
    ):  # no **kwargs
        narrow_seen.append(
            dict(
                session_key=session_key,
                platform=platform,
                reason=reason,
                invalidation_reason=invalidation_reason,
            )
        )
        return None

    ctx.register_hook("agent_loop_stopped", legacy_observer)

    session = {
        "profile_home": None,
        "session_key": "launch-session",
        "history_lock": threading.Lock(),
        "running": True,
        "agent": MagicMock(),
        "_run_thread": None,
    }
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    assert narrow_seen == [
        {
            "session_key": "launch-session",
            "platform": "tui",
            "reason": "user_stop",
            "invalidation_reason": "session_interrupt",
        }
    ]


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupt_rpc_path_fires_hook_in_session_profile_scope(
    mock_invoke_hook, tmp_path, monkeypatch
):
    """The session.interrupt RPC handler itself (not just the helper) dispatches
    the hook inside the session's profile scope; same-key stores disambiguate."""
    from tui_gateway import server

    launch = tmp_path / "launch"
    selected = tmp_path / "selected"
    launch.mkdir()
    selected.mkdir()
    sid = "profiled-tab"
    session = _make_session(running=True, profile_home=str(selected))
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    try:
        response = server.handle_request({
            "id": "intr",
            "method": "session.interrupt",
            "params": {"session_id": sid},
        })
        assert response["result"]["status"] == "interrupted"
        calls = _hook_calls(mock_invoke_hook)
        assert len(calls) == 1
        assert calls[0].kwargs["profile_home"] == str(selected)
        assert calls[0].kwargs["session_key"] == "agent:main:tui:dm:s1"
    finally:
        server._sessions.pop(sid, None)
