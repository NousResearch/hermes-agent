"""Handoff preserves process notification policy and restart routing (#135685)."""

import json
import os
import time
import weakref
from types import SimpleNamespace

import pytest

import tools.process_registry as registry_module
from tools.process_registry import ProcessRegistry, ProcessSession, _handle_process
from tools.process_registry_notifications import should_surface_notification


class _Agent:
    pass


@pytest.fixture
def handoff_state(monkeypatch):
    import gateway.run as gateway_run
    import tools.delegate_tool_registry as delegates
    import tools.terminal_tool as terminal

    registry = ProcessRegistry()
    registry._completions_restored = True
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(terminal, "_resolve_container_task_id", lambda owner: owner)
    parent = _Agent()
    parent.session_id = "parent-session"
    parent._current_task_id = "parent-turn"
    parent._gateway_session_key = "agent:main:discord:thread:chat:thread"
    child = _Agent()
    child._delegate_parent_ref = weakref.ref(parent)
    monkeypatch.setattr(delegates, "_active_subagents", {"sa-child": {"agent": child}})
    session = ProcessSession(
        id="proc_handoff", command="worker", pid=os.getpid(), started_at=time.time(),
        task_id="sa-child", owner_task_id="sa-child", parent_session_id="child-session",
        session_key=parent._gateway_session_key, watcher_platform="discord", watcher_chat_id="chat",
    )
    registry._running[session.id] = session
    armed = []
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(
        arm_process_watcher=lambda descriptor: armed.append(descriptor) or True))
    return SimpleNamespace(registry=registry, session=session, parent=parent, child=child, armed=armed)


def _handoff(state, task_id="sa-child"):
    result = json.loads(_handle_process(
        {"action": "handoff", "session_id": state.session.id, "data": "report worker result"},
        task_id=task_id,
    ))
    assert result["status"] == "handed_off", result


def test_handoff_checkpoint_restores_parent_and_armed_watcher(handoff_state, monkeypatch):
    state = handoff_state
    state.registry._write_checkpoint()
    _handoff(state)

    recovered = ProcessRegistry()
    monkeypatch.setattr(recovered, "_detached_host_fate", lambda *args: "running")
    assert recovered.recover_from_checkpoint() == 1
    session = recovered.get(state.session.id)
    assert session.owner_task_id == "parent-turn"
    assert session.session_key == state.parent._gateway_session_key
    assert session.parent_session_id == "parent-session"
    assert session.handoff_note == "report worker result"
    assert session.notify_on_complete is True
    assert recovered.pending_watchers[0]["parent_session_id"] == "parent-session"
    assert recovered.pending_watchers[0]["notify_on_complete"] is True


def test_handoff_purpose_redacts_inline_credentials_in_checkpoint_and_notice(handoff_state):
    secret = "sk-secret1234567890"
    handoff_state.session.handoff_note = f"worker with Authorization: Bearer {secret}"
    handoff_state.registry._write_checkpoint()
    record = json.loads(registry_module._checkpoint_path().read_text())[0]
    assert secret not in record["handoff_note"]
    assert "worker" in record["handoff_note"]
    handoff_state.session.notify_on_complete = True
    handoff_state.session.mark_exited(0)
    handoff_state.registry._move_to_finished(handoff_state.session)
    notice = handoff_state.registry.completion_queue.get_nowait()
    assert secret not in notice["handoff_note"]
    assert "worker" in notice["handoff_note"]


@pytest.mark.parametrize("event_type", ["watch_match", "heartbeat"])
def test_handoff_preserves_queued_identity_and_stamps_future_events(handoff_state, event_type):
    state = handoff_state
    session = state.session
    if event_type == "watch_match":
        session.watch_patterns = ["READY"]
        emit = lambda text: state.registry._check_watch_patterns(session, text)
    else:
        session.heartbeat_seconds = 30

        def emit(text):
            session.append_output(text)
            state.registry._emit_heartbeat(session, time.time())

    emit("READY before handoff")
    before = state.registry.completion_queue.get_nowait()
    _handoff(state)
    session._watch_cooldown_until = 0
    emit("READY after handoff")
    after = state.registry.completion_queue.get_nowait()

    assert before["owner_task_id"] == "sa-child"
    assert before["parent_session_id"] == "child-session"
    assert should_surface_notification(before, surface_child=False) is False
    assert after["owner_task_id"] == "parent-turn"
    assert after["parent_session_id"] == "parent-session"
    assert after["session_key"] == state.parent._gateway_session_key
    assert should_surface_notification(after, surface_child=False) is True
    if event_type == "watch_match":
        assert session.watch_patterns == ["READY"]
        assert session.notify_on_complete is False
        assert not state.armed


def test_nested_handoff_stays_child_owned_until_root_takes_over(handoff_state, monkeypatch):
    import tools.delegate_tool_registry as delegates

    state = handoff_state
    state.parent._current_task_id = "sa-parent"
    _handoff(state)
    assert not should_surface_notification(
        state.registry._watch_event_base(state.session), surface_child=False)

    root = _Agent()
    root.session_id = "root-session"
    root._current_task_id = "root-turn"
    root._gateway_session_key = state.parent._gateway_session_key
    state.parent._delegate_parent_ref = weakref.ref(root)
    monkeypatch.setitem(delegates._active_subagents, "sa-parent", {"agent": state.parent})
    _handoff(state, task_id="sa-parent")
    assert should_surface_notification(
        state.registry._watch_event_base(state.session), surface_child=False)
    assert state.session.parent_session_id == "root-session"
    assert state.session.session_key == root._gateway_session_key
    assert len(state.armed) == 1


@pytest.mark.parametrize("surface_child", [False, True])
def test_discord_child_spawn_retains_completion_for_policy_and_accounting(handoff_state, monkeypatch, surface_child):
    import agent.delegation_context as delegation_context
    import tools.terminal_tool_background as background

    state = handoff_state
    monkeypatch.setattr(background, "_spawn", lambda *args, **kwargs: state.session)
    monkeypatch.setattr(background, "_apply_async_support", lambda ps, result, notify, watch: (notify, watch))
    monkeypatch.setattr(delegation_context, "is_delegated_child_context", lambda: True)
    monkeypatch.setattr(ProcessRegistry, "_surface_child_process_notifications", staticmethod(lambda: surface_child))
    result = json.loads(background.spawn_background_process(
        command="worker", env=SimpleNamespace(), env_type="local", effective_task_id="sa-child",
        task_id="sa-child", session_key=state.session.session_key, workdir=None, cwd="/tmp",
        effective_pty=False, notify_on_complete=True, watch_patterns=None, approval_note=None,
        pty_disabled_reason=None, completion_output_chars=6000,
    ))
    assert result["error"] is None
    assert state.session.notify_on_complete is True
    assert state.session.completion_output_chars == 6000
    assert len(state.armed) == 1
    state.session.append_output("worker result")
    state.session.mark_exited(0)
    state.registry._move_to_finished(state.session)
    assert state.registry.unread_completions_owned_by("sa-child") == [state.session]
    delivered = state.registry.drain_notifications(session_key=state.session.session_key)
    assert bool(delivered) is surface_child


@pytest.mark.parametrize("notification", ["completion", "pattern"])
def test_spawn_checkpoint_retains_final_notification_stamps(handoff_state, monkeypatch, notification):
    import gateway.session_context as context
    import tools.terminal_tool_background as background

    state = handoff_state
    state.session.watcher_platform = ""
    state.session.parent_session_id = ""

    def spawn(*args, **kwargs):
        state.registry._write_checkpoint()
        return state.session

    monkeypatch.setattr(background, "_spawn", spawn)
    monkeypatch.setattr(context, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(context, "get_session_env", lambda key, default="": {
        "HERMES_SESSION_PLATFORM": "discord", "HERMES_SESSION_ID": "child-session",
        "HERMES_SESSION_CHAT_ID": "chat",
    }.get(key, default))
    monkeypatch.setattr(state.registry, "_ensure_heartbeat_thread", lambda: None)
    completion = notification == "completion"
    result = json.loads(background.spawn_background_process(
        command="worker", env=SimpleNamespace(), env_type="local", effective_task_id="sa-child",
        task_id="sa-child", session_key=state.session.session_key, workdir=None, cwd=os.getcwd(),
        effective_pty=False, notify_on_complete=completion, watch_patterns=None if completion else ["READY"],
        approval_note=None, pty_disabled_reason=None, heartbeat_seconds=60 if completion else 0,
    ))
    assert result["error"] is None

    recovered = ProcessRegistry()
    monkeypatch.setattr(recovered, "_detached_host_fate", lambda *args: "running")
    assert recovered.recover_from_checkpoint() == 1
    session = recovered.get(state.session.id)
    assert session.watcher_platform == "discord"
    assert session.parent_session_id == "child-session"
    assert session.notify_on_complete is completion
    assert session.watch_patterns == ([] if completion else ["READY"])
    assert session.heartbeat_seconds == (60 if completion else 0)


@pytest.mark.parametrize("async_supported", [False, True])
def test_yield_checkpoint_retains_final_delivery_capability(handoff_state, monkeypatch, async_supported):
    import gateway.session_context as context
    from tools.terminal_tool_background import yield_to_background_handler

    state = handoff_state
    state.session.watcher_platform = ""
    state.session.parent_session_id = ""

    def adopt(*args, **kwargs):
        state.session.notify_on_complete = True
        state.registry._write_checkpoint()
        return state.session

    monkeypatch.setattr(state.registry, "adopt_local", adopt)
    monkeypatch.setattr(context, "async_delivery_supported", lambda: async_supported)
    monkeypatch.setattr(context, "get_session_env", lambda key, default="": {
        "HERMES_SESSION_PLATFORM": "discord", "HERMES_SESSION_ID": "child-session",
        "HERMES_SESSION_CHAT_ID": "chat",
    }.get(key, default))
    handler = yield_to_background_handler(
        command="worker", env_type="local", cwd=os.getcwd(), effective_task_id="sa-child",
        task_id="sa-child", session_key=state.session.session_key,
    )
    assert handler(object(), "partial")["yielded_session_id"] == state.session.id

    recovered = ProcessRegistry()
    monkeypatch.setattr(recovered, "_detached_host_fate", lambda *args: "running")
    assert recovered.recover_from_checkpoint() == 1
    session = recovered.get(state.session.id)
    assert session.notify_on_complete is async_supported
    if async_supported:
        assert recovered.pending_watchers[0]["parent_session_id"] == "child-session"
