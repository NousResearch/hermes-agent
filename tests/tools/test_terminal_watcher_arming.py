"""Gateway completion watchers are armed at launch, not only after the turn ends.

A ``notify_on_complete`` process that finishes while its launching turn is still running
used to sit in ``pending_watchers`` until the post-turn drain, leaving the chat mute for
the whole turn (#112033).
"""

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


# ---------------------------------------------------------------------------
# Route guard: a delegate child never registers a route-following watcher
# ---------------------------------------------------------------------------


def test_is_route_guard_platform_accepts_members_and_names():
    """The decision predicate takes a Platform member or a plain string (#57498)."""
    from gateway.config import Platform

    from tools.terminal_tool_background import is_route_guard_platform

    assert is_route_guard_platform(Platform.DISCORD) is True
    assert is_route_guard_platform("discord") is True
    assert is_route_guard_platform("Discord") is True
    assert is_route_guard_platform(Platform.TELEGRAM) is False
    assert is_route_guard_platform(None) is False


class _StubProcSession:
    """The surface ``spawn_background_process`` stamps and reads back."""

    def __init__(self, platform):
        self.id = "proc_child"
        self.pid = 4242
        self.watcher_platform = platform
        self.watcher_chat_id = "chan1"
        self.watcher_user_id = "u"
        self.watcher_user_name = "Ada"
        self.watcher_thread_id = ""
        self.watcher_message_id = ""
        self.parent_session_id = "sess_chat"
        self.notify_on_complete = False
        self.completion_output_chars = 0
        self.watch_patterns = []
        self.watcher_interval = 0


def _spawn_and_record(monkeypatch, platform, *, delegated_child):
    """Run the real spawn decision with the process launch stubbed out.

    Returns (parsed result JSON, list that gains an entry when a completion
    watcher is registered).
    """
    import agent.delegation_context as delegation_context
    import tools.terminal_tool as terminal_tool
    import tools.terminal_tool_background as ttb

    armed: list[str] = []
    monkeypatch.setattr(ttb, "_spawn", lambda *a, **kw: _StubProcSession(platform))
    monkeypatch.setattr(ttb, "_apply_async_support", lambda ps, rd, notify, watch: (notify, watch))
    monkeypatch.setattr(ttb, "_register_completion_watcher", lambda *a, **kw: armed.append("armed"))
    monkeypatch.setattr(
        terminal_tool, "_resolve_command_cwd", lambda **kw: kw["default_cwd"]
    )
    monkeypatch.setattr(
        delegation_context, "is_delegated_child_context", lambda: delegated_child
    )

    import json

    raw = ttb.spawn_background_process(
        command="sleep 1",
        env=SimpleNamespace(),
        env_type="local",
        effective_task_id="task-1",
        task_id=None,
        session_key="agent:main:discord:group:chan1",
        workdir=None,
        cwd="/tmp",
        effective_pty=False,
        notify_on_complete=True,
        watch_patterns=None,
        approval_note=None,
        pty_disabled_reason=None,
    )
    return json.loads(raw), armed


def test_delegate_child_on_a_route_guard_platform_registers_no_watcher(monkeypatch):
    """The child's pin would BE the route: decide before registering, not after
    (2026-09-17 dev-Discord hijack, #57498)."""
    from gateway.config import Platform

    result, armed = _spawn_and_record(monkeypatch, Platform.DISCORD, delegated_child=True)

    assert armed == []
    assert result["notify_on_complete"] is False
    assert "subagent_note" in result


def test_same_spawn_on_a_guarded_platform_still_arms_for_a_non_child(monkeypatch):
    """The guard is about delegate children, not about the platform."""
    from gateway.config import Platform

    result, armed = _spawn_and_record(monkeypatch, Platform.DISCORD, delegated_child=False)

    assert armed == ["armed"]
    assert result["notify_on_complete"] is True
    assert "subagent_note" not in result


def test_delegate_child_on_an_unguarded_platform_keeps_todays_behaviour(monkeypatch):
    """Discord-only by decision: no other platform changes behaviour."""
    from gateway.config import Platform

    result, armed = _spawn_and_record(monkeypatch, Platform.TELEGRAM, delegated_child=True)

    assert armed == ["armed"]
    assert result["notify_on_complete"] is False
    assert "subagent_note" in result

