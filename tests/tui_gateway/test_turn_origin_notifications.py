import queue
import threading
import types

import pytest

from tui_gateway import server
from tools.process_registry import process_registry


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "transport": None,
        **extra,
    }


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        self._target()

    def is_alive(self):
        return False


def _completion(process_id="proc-1"):
    return {
        "type": "completion",
        "session_id": process_id,
        "session_key": "session-key",
        "command": "echo done",
        "exit_code": 0,
        "output": "done",
    }


def test_turn_origin_token_prevents_stale_clear():
    session = _session()

    with session["history_lock"]:
        old = server._set_turn_origin_locked(session, "user")
        current = server._set_turn_origin_locked(session, "notification")
        server._clear_turn_origin_locked(session, old)
        assert session["turn_origin"] == "notification"
        server._clear_turn_origin_locked(session, current)
        assert session["turn_origin"] is None


def test_human_and_queued_prompts_dispatch_as_user(monkeypatch):
    calls = []
    session = _session()

    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_a, **_kw: None)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    server._sessions["sid-user"] = session

    try:
        response = server._methods["prompt.submit"](
            "rid-user", {"session_id": "sid-user", "text": "hello"}
        )
        assert response["result"]["status"] == "streaming"
        assert calls[-1][1]["origin"] == "user"

        session["running"] = False
        session["queued_prompt"] = {"text": "queued", "transport": None}
        assert server._drain_queued_prompt("rid-queued", "sid-user", session) is True
        assert calls[-1][1]["origin"] == "user"
    finally:
        server._sessions.pop("sid-user", None)


def test_goal_followup_dispatches_as_goal(monkeypatch):
    calls = []
    session = _session()
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: calls.append((a, kw)))

    assert server._dispatch_goal_followup("rid", "sid", session, "continue") is True
    assert calls == [
        (("rid", "sid", session, "continue"), {"origin": "goal"})
    ]


@pytest.mark.parametrize("shutdown", [False, True], ids=["idle-poller", "shutdown-drain"])
def test_notification_poller_entry_paths_dispatch_as_notification(monkeypatch, shutdown):
    isolated_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc-origin")
    isolated_queue.put(_completion("proc-origin"))

    calls = []
    emitted = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))

    if shutdown:
        stop = threading.Event()
        stop.set()
    else:
        class _StopAfterOne:
            checks = 0

            def is_set(self):
                self.checks += 1
                return self.checks > 1

        stop = _StopAfterOne()

    session = _session()
    server._notification_poller_loop(stop, "sid", session)

    assert len(calls) == 1
    assert calls[0][1]["origin"] == "notification"
    starts = [event for event in emitted if event[0] == "message.start"]
    assert starts == []  # _run_prompt_submit is the sole message.start owner.
    statuses = [event for event in emitted if event[0] == "status.update"]
    assert statuses and statuses[0][2]["kind"] == "process"


def test_post_turn_notification_emits_status_and_notification_origin(monkeypatch):
    event = _completion("proc-post-turn")
    text = "[IMPORTANT: background result]"
    calls = []
    emitted = []
    session = _session()

    monkeypatch.setattr(
        process_registry,
        "drain_notifications",
        lambda **_kwargs: [(event, text)],
    )
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))

    server._drain_post_turn_notifications("rid", "sid", session)

    assert calls[0][1]["origin"] == "notification"
    assert ("status.update", "sid", {"kind": "process", "text": text}) in emitted


def test_busy_post_turn_drain_requeues_once_without_hot_loop(monkeypatch):
    pairs = [
        (_completion("proc-a"), "result a"),
        (_completion("proc-b"), "result b"),
    ]
    isolated_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(process_registry, "drain_notifications", lambda **_kwargs: pairs)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )

    session = _session(running=True)
    server._drain_post_turn_notifications("rid", "sid", session)

    assert isolated_queue.qsize() == 2


def test_max_iteration_defers_completion_then_next_user_claims_clean_context(monkeypatch):
    isolated_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc-max")
    isolated_queue.put(_completion("proc-max"))

    emitted = []
    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
        ):
            calls.append(
                {
                    "prompt": prompt,
                    "persist_user_message": persist_user_message,
                }
            )
            history = list(conversation_history or [])
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "answer"},
                ]
            )
            return {
                "final_response": "forced final report" if len(calls) == 1 else "answer",
                "messages": history,
                "turn_exit_reason": (
                    "max_iterations_reached(60/60)"
                    if len(calls) == 1
                    else "text_response(finish_reason=stop)"
                ),
            }

    session = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_a, **_kw: ".")
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    from hermes_cli.goals import GoalManager

    monkeypatch.setattr(GoalManager, "is_active", lambda _self: False)

    server._run_prompt_submit("rid-1", "sid", session, "First request", origin="user")

    assert len(calls) == 1
    assert session["deferred_notification_texts"]
    assert not [
        event
        for event in emitted
        if event[0] == "message.start" and event[2].get("turn_origin") == "notification"
    ]
    pending_statuses = [
        event[2]["text"]
        for event in emitted
        if event[0] == "status.update" and "pending" in event[2]["text"].lower()
    ]
    assert pending_statuses

    server._run_prompt_submit("rid-2", "sid", session, "What now?", origin="user")

    assert len(calls) == 2
    assert "BACKGROUND COMPLETION CONTEXT" in calls[1]["prompt"]
    assert "proc-max" in calls[1]["prompt"]
    assert calls[1]["prompt"].endswith("What now?")
    assert calls[1]["persist_user_message"] == "What now?"
    user_messages = [message for message in session["history"] if message.get("role") == "user"]
    assert user_messages[-1]["content"] == "What now?"
    assert session["deferred_notification_texts"] == []
    assert session["defer_notifications_until_user"] is False


def _patch_prompt_turn_runtime(monkeypatch, emitted):
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_a, **_kw: ".")
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_drain_post_turn_notifications", lambda *_a, **_kw: None)
    from hermes_cli.goals import GoalManager

    monkeypatch.setattr(GoalManager, "is_active", lambda _self: False)


def _successful_turn(prompt, conversation_history):
    history = list(conversation_history or [])
    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "answer"},
        ]
    )
    return {
        "final_response": "answer",
        "messages": history,
        "turn_exit_reason": "text_response(finish_reason=stop)",
    }


def test_failed_user_turn_restores_claimed_notifications_before_concurrent_arrivals(monkeypatch):
    emitted = []
    calls = []
    session = None

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
        ):
            calls.append(prompt)
            if len(calls) == 1:
                with session["history_lock"]:
                    session.setdefault("deferred_notification_texts", []).append("concurrent result")
                raise RuntimeError("agent failed")
            return _successful_turn(prompt, conversation_history)

    session = _session(
        agent=_Agent(),
        deferred_notification_texts=["claimed result a", "claimed result b"],
        defer_notifications_until_user=True,
    )
    _patch_prompt_turn_runtime(monkeypatch, emitted)

    server._run_prompt_submit("rid-1", "sid", session, "First attempt", origin="user")

    assert session["deferred_notification_texts"] == [
        "claimed result a",
        "claimed result b",
        "concurrent result",
    ]
    assert session["defer_notifications_until_user"] is True

    server._run_prompt_submit("rid-2", "sid", session, "Retry", origin="user")

    delivered = calls[-1]
    for notification in ("claimed result a", "claimed result b", "concurrent result"):
        assert delivered.count(notification) == 1
    assert delivered.endswith("Retry")
    assert session["deferred_notification_texts"] == []
    assert session["defer_notifications_until_user"] is False


def test_context_reference_block_restores_claimed_notifications_for_next_user_turn(monkeypatch):
    emitted = []
    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
        ):
            calls.append(prompt)
            return _successful_turn(prompt, conversation_history)

    session = _session(
        agent=_Agent(),
        deferred_notification_texts=["claimed before block"],
        defer_notifications_until_user=True,
    )
    _patch_prompt_turn_runtime(monkeypatch, emitted)

    from agent import context_references, model_metadata

    def block_context(*_args, **_kwargs):
        with session["history_lock"]:
            session.setdefault("deferred_notification_texts", []).append("arrived during block")
        return types.SimpleNamespace(blocked=True, warnings=["blocked"], message="")

    monkeypatch.setattr(context_references, "preprocess_context_references", block_context)
    monkeypatch.setattr(model_metadata, "get_model_context_length", lambda *_a, **_kw: 32_000)

    server._run_prompt_submit("rid-1", "sid", session, "Read @file:outside", origin="user")

    assert calls == []
    assert session["deferred_notification_texts"] == ["claimed before block", "arrived during block"]
    assert session["defer_notifications_until_user"] is True

    server._run_prompt_submit("rid-2", "sid", session, "Continue safely", origin="user")

    delivered = calls[-1]
    assert delivered.count("claimed before block") == 1
    assert delivered.count("arrived during block") == 1
    assert delivered.endswith("Continue safely")
    assert session["deferred_notification_texts"] == []
    assert session["defer_notifications_until_user"] is False


def test_post_turn_dispatch_failure_requeues_current_and_remaining_events(monkeypatch):
    isolated_queue = queue.Queue()
    events = [_completion("proc-dispatch-fail-1"), _completion("proc-dispatch-fail-2")]

    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(
        process_registry,
        "drain_notifications",
        lambda **_kwargs: [(events[0], "first"), (events[1], "second")],
    )
    monkeypatch.setattr(
        server,
        "_dispatch_notification_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatch failed")),
    )

    server._drain_post_turn_notifications("rid", "sid", _session())

    assert [isolated_queue.get_nowait(), isolated_queue.get_nowait()] == events
    assert isolated_queue.empty()


def test_notification_poller_retries_after_synchronous_dispatch_failure(monkeypatch):
    isolated_queue = queue.Queue()
    event = _completion("proc-poller-retry")
    isolated_queue.put(event)
    stop = threading.Event()
    attempts = []

    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(process_registry, "is_completion_consumed", lambda _session_id: False)
    monkeypatch.setattr(server, "_notification_event_belongs_elsewhere", lambda *_args: False)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)

    def dispatch(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("dispatch failed")
        stop.set()
        return "deferred"

    monkeypatch.setattr(server, "_dispatch_notification_turn", dispatch)

    server._notification_poller_loop(stop, "sid", _session())

    assert len(attempts) == 2
    assert isolated_queue.empty()
