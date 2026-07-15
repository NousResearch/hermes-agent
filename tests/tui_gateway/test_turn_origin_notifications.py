import queue
import sqlite3
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
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_kwargs):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

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

def _durable_completion(tmp_path, monkeypatch, delegation_id):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import async_delegation

    async_delegation._persist_dispatch(
        {
            "delegation_id": delegation_id,
            "session_key": "session-key",
            "origin_ui_session_id": "sid",
            "parent_session_id": None,
            "dispatched_at": 1.0,
        }
    )
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "session-key",
        "origin_ui_session_id": "sid",
        "status": "completed",
        "completed_at": 2.0,
        "summary": "done",
    }
    async_delegation._persist_completion(event, {"status": "completed", "summary": "done"})
    return async_delegation, event


def _adoption_event_ids(db_path):
    with sqlite3.connect(db_path) as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT event_id FROM deferred_notification_adoptions ORDER BY event_id"
            ).fetchall()
        ]


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


def _patch_prompt_turn_runtime(
    monkeypatch,
    emitted,
    *,
    immediate_thread=True,
    disable_post_turn_drain=True,
):
    if immediate_thread:
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
    if disable_post_turn_drain:
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


def test_history_version_mismatch_restores_claimed_notifications(monkeypatch):
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
                assert session is not None
                with session["history_lock"]:
                    session["history_version"] += 1
                    session.setdefault("deferred_notification_texts", []).append(
                        "arrived during mismatch"
                    )
            return _successful_turn(prompt, conversation_history)

    session = _session(
        agent=_Agent(),
        deferred_notification_texts=["claimed before mismatch"],
        defer_notifications_until_user=True,
    )
    _patch_prompt_turn_runtime(monkeypatch, emitted)

    server._run_prompt_submit("rid-1", "sid", session, "First attempt", origin="user")

    assert session["history"] == []
    assert session["deferred_notification_texts"] == [
        "claimed before mismatch",
        "arrived during mismatch",
    ]
    assert session["defer_notifications_until_user"] is True

    server._run_prompt_submit("rid-2", "sid", session, "Retry", origin="user")

    delivered = calls[-1]
    assert delivered.count("claimed before mismatch") == 1
    assert delivered.count("arrived during mismatch") == 1
    assert delivered.endswith("Retry")
    assert session["deferred_notification_texts"] == []
    assert session["defer_notifications_until_user"] is False


def test_complete_history_adoption_without_durable_proof_restores_batch(
    tmp_path, monkeypatch
):
    """Live history cannot consume an owned durable batch without DB proof."""
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-history-only"
    )
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    event_id = "async_delegation:deleg-history-only"
    db = SessionDB(db_path)
    db.create_session("session-key", source="tui")
    assert async_delegation.persist_deferred_notification(
        "session-key",
        event_id,
        "claimed without proof",
        event,
        db_path=db_path,
    )

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
            deferred_notification_ids=None,
        ):
            assert deferred_notification_ids == [event_id]
            return _successful_turn(prompt, conversation_history)

    session = _session(
        agent=_Agent(),
        profile_home=str(tmp_path),
        deferred_notification_texts=["claimed without proof"],
        deferred_notification_event_ids={event_id},
        defer_notifications_until_user=True,
    )
    emitted = []
    ack_threads = []

    class _PromptOnlyThread:
        def __init__(
            self,
            target=None,
            args=(),
            kwargs=None,
            daemon=None,
            name=None,
            **_kwargs,
        ):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            self._name = name

        def start(self):
            if self._name:
                ack_threads.append(self._name)
                return
            self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

    monkeypatch.setattr(server.threading, "Thread", _PromptOnlyThread)
    _patch_prompt_turn_runtime(monkeypatch, emitted, immediate_thread=False)

    try:
        server._run_prompt_submit(
            "rid-history-only", "sid", session, "Next request", origin="user"
        )

        assert session["history"][-1] == {"role": "assistant", "content": "answer"}
        assert session["deferred_notification_texts"] == ["claimed without proof"]
        assert session["deferred_notification_event_ids"] == {event_id}
        assert session["defer_notifications_until_user"] is True
        assert ack_threads == []
        assert _adoption_event_ids(db_path) == []
        assert [
            row["event_id"]
            for row in async_delegation.load_deferred_notifications(
                "session-key", db_path=db_path
            )
        ] == [event_id]
    finally:
        db.close()


def test_history_version_mismatch_keeps_live_history_when_db_adoption_won(
    tmp_path, monkeypatch
):
    """A committed SessionDB adoption wins over stale in-memory restoration."""
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-mismatch-adopted"
    )
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    event_id = "async_delegation:deleg-mismatch-adopted"
    db = SessionDB(db_path)
    db.create_session("session-key", source="tui")
    db.append_message("session-key", "user", "Original request")
    assert async_delegation.persist_deferred_notification(
        "session-key",
        event_id,
        "claimed durable result",
        event,
        db_path=db_path,
    )

    session = None

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
            deferred_notification_ids=None,
        ):
            assert deferred_notification_ids == [event_id]
            db.append_message("session-key", "user", persist_user_message)
            db.append_message(
                "session-key",
                "assistant",
                "answer",
                deferred_notification_ids=deferred_notification_ids,
            )
            assert session is not None
            with session["history_lock"]:
                session["history"].append(
                    {"role": "assistant", "content": "newer live history"}
                )
                session["history_version"] += 1
            return _successful_turn(prompt, conversation_history)

    original_history = [{"role": "user", "content": "Original request"}]
    session = _session(
        agent=_Agent(),
        profile_home=str(tmp_path),
        history=list(original_history),
        history_version=4,
        deferred_notification_texts=["claimed durable result"],
        deferred_notification_event_ids={event_id},
        defer_notifications_until_user=True,
    )
    emitted = []
    _patch_prompt_turn_runtime(monkeypatch, emitted)

    server._run_prompt_submit(
        "rid-mismatch", "sid", session, "Next request", origin="user"
    )

    assert session["history"] == [
        *original_history,
        {"role": "assistant", "content": "newer live history"},
    ]
    assert session["deferred_notification_texts"] == []
    assert session["deferred_notification_event_ids"] == set()
    assert session["defer_notifications_until_user"] is False
    assert async_delegation.load_deferred_notifications(
        "session-key", db_path=db_path
    ) == []
    assert event_id not in _adoption_event_ids(db_path)

    restarted_history = db.get_messages_as_conversation("session-key")
    db.close()
    restarted = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd=".",
        history=restarted_history,
        lease=None,
        profile_home=tmp_path,
    )
    later_prompts = []

    class _LaterAgent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            later_prompts.append(prompt)
            return _successful_turn(prompt, conversation_history)

    restarted["agent"] = _LaterAgent()
    assert restarted["deferred_notification_texts"] == []
    assert restarted["defer_notifications_until_user"] is False

    server._run_prompt_submit(
        "rid-later", "sid", restarted, "Later request", origin="user"
    )

    assert later_prompts == ["Later request"]
    assert "claimed durable result" not in later_prompts[0]


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


def test_deferred_durable_ack_failure_retries_without_duplicate(tmp_path, monkeypatch):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-deferred-ack"
    )
    session = _session(
        defer_notifications_until_user=True,
        deferred_notification_texts=[],
    )
    emitted = []
    acknowledged = threading.Event()
    attempts = []
    original_complete = async_delegation.complete_event_delivery

    def flaky_complete(evt, claim_id):
        attempts.append(claim_id)
        if len(attempts) == 1:
            raise OSError("temporary sqlite failure")
        completed = original_complete(evt, claim_id)
        if completed:
            acknowledged.set()
        return completed

    monkeypatch.setattr(async_delegation, "complete_event_delivery", flaky_complete)
    monkeypatch.setattr(server, "_NOTIFICATION_ACK_RETRY_DELAYS", (0.0,))
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    outcome = server._dispatch_notification_turn(
        "rid",
        "sid",
        session,
        "durable result",
        event=event,
        consumer="test-deferred",
    )

    assert outcome == "deferred"
    assert acknowledged.wait(3)
    assert session["deferred_notification_texts"] == ["durable result"]
    assert session["deferred_notification_event_ids"] == {
        "async_delegation:deleg-deferred-ack"
    }
    durable = async_delegation.get_durable_delegation("deleg-deferred-ack")
    assert durable["delivery_attempts"] == 1
    assert durable["delivery_state"] == "delivered"
    assert durable["result"] == {"status": "completed", "summary": "done"}
    assert durable["state"] == "completed"

    assert server._dispatch_notification_turn(
        "rid-duplicate",
        "sid",
        session,
        "durable result",
        event=event,
        consumer="test-deferred-duplicate",
    ) == "claimed"
    assert session["deferred_notification_texts"] == ["durable result"]
    assert len([item for item in emitted if item[0] == "status.update"]) == 1


def test_deferred_durable_completion_survives_session_recreation_and_consumes_once(
    tmp_path, monkeypatch
):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-deferred-recreate"
    )
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-key", source="tui")
    db.close()

    first_session = _session(
        defer_notifications_until_user=True,
        deferred_notification_texts=[],
        profile_home=str(tmp_path),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    assert server._dispatch_notification_turn(
        "rid-defer",
        "sid",
        first_session,
        "durable result",
        event=event,
        consumer="test-recreate",
    ) == "deferred"
    assert async_delegation.get_durable_delegation("deleg-deferred-recreate")[
        "delivery_state"
    ] == "delivered"

    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
            deferred_notification_ids=None,
        ):
            calls.append((prompt, persist_user_message))
            if deferred_notification_ids:
                adoption_db = SessionDB(tmp_path / "state.db")
                adoption_db.append_message(
                    "session-key",
                    "assistant",
                    "answer",
                    deferred_notification_ids=deferred_notification_ids,
                )
                adoption_db.close()
            return _successful_turn(prompt, conversation_history)

    recreated = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd=".",
        history=[],
        lease=None,
        profile_home=tmp_path,
    )
    recreated["agent"] = _Agent()
    _patch_prompt_turn_runtime(monkeypatch, [])

    assert recreated["defer_notifications_until_user"] is True
    assert recreated["deferred_notification_texts"] == ["durable result"]

    server._run_prompt_submit("rid-consume", "sid", recreated, "Next request", origin="user")

    assert len(calls) == 1
    assert calls[0][0].count("durable result") == 1
    assert calls[0][0].endswith("Next request")
    assert calls[0][1] == "Next request"
    assert [
        message["content"]
        for message in recreated["history"]
        if message.get("role") == "user"
    ] == ["Next request"]

    recreated_again = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd=".",
        history=list(recreated["history"]),
        lease=None,
        profile_home=tmp_path,
    )
    recreated_again["agent"] = _Agent()
    assert recreated_again.get("deferred_notification_texts", []) == []
    assert recreated_again.get("defer_notifications_until_user", False) is False

    server._run_prompt_submit(
        "rid-after-consume", "sid", recreated_again, "Later request", origin="user"
    )

    assert len(calls) == 2
    assert calls[1] == ("Later request", None)
    assert sum("durable result" in prompt for prompt, _persisted in calls) == 1


def test_committed_adoption_reconciles_after_crash_before_ledger_ack(
    tmp_path, monkeypatch
):
    """A restart suppresses a row whose assistant/adoption commit already won."""
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-crash-after-adoption"
    )
    from hermes_state import SessionDB

    event_id = "async_delegation:deleg-crash-after-adoption"
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-key", source="tui")
    assert async_delegation.persist_deferred_notification(
        "session-key",
        event_id,
        "durable result",
        event,
        db_path=tmp_path / "state.db",
    )
    db.append_message("session-key", "user", "Next request")
    db.append_message(
        "session-key",
        "assistant",
        "answer",
        deferred_notification_ids=[event_id],
    )
    db.close()

    # Deterministic process-death window: transcript + adoption committed, but
    # _ack_consumed_deferred_notifications never ran.
    assert [
        row["event_id"]
        for row in async_delegation.load_deferred_notifications(
            "session-key", db_path=tmp_path / "state.db"
        )
    ] == [event_id]

    restarted_db = SessionDB(tmp_path / "state.db")
    history = restarted_db.get_messages_as_conversation("session-key")
    restarted_db.close()
    assert [(message["role"], message["content"]) for message in history] == [
        ("user", "Next request"),
        ("assistant", "answer"),
    ]

    recreated = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd=".",
        history=history,
        lease=None,
        profile_home=tmp_path,
    )

    assert recreated["deferred_notification_texts"] == []
    assert recreated["deferred_notification_event_ids"] == set()
    assert recreated["defer_notifications_until_user"] is False
    assert async_delegation.load_deferred_notifications(
        "session-key", db_path=tmp_path / "state.db"
    ) == []
    assert event_id not in _adoption_event_ids(tmp_path / "state.db")
    assert async_delegation.complete_deferred_notifications(
        "session-key", [event_id], db_path=tmp_path / "state.db"
    )


def test_deferred_ack_retry_survives_committed_proof_cleanup(tmp_path, monkeypatch):
    """A lost ack return retries after the first transaction pruned its proof."""
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-ack-retry-cleanup"
    )
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    event_id = "async_delegation:deleg-ack-retry-cleanup"
    db = SessionDB(db_path)
    db.create_session("session-key", source="tui")
    assert async_delegation.persist_deferred_notification(
        "session-key", event_id, "durable result", event, db_path=db_path
    )
    db.append_message(
        "session-key",
        "assistant",
        "answer",
        deferred_notification_ids=[event_id],
    )
    db.close()

    attempts = []
    original_complete = async_delegation.complete_deferred_notifications

    def committed_then_lost(session_id, event_ids, *, db_path=None):
        attempts.append(tuple(event_ids))
        completed = original_complete(session_id, event_ids, db_path=db_path)
        if len(attempts) == 1:
            raise OSError("lost committed ack result")
        return completed

    monkeypatch.setattr(
        async_delegation,
        "complete_deferred_notifications",
        committed_then_lost,
    )
    monkeypatch.setattr(server, "_NOTIFICATION_ACK_RETRY_DELAYS", (0.0,))
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    server._ack_consumed_deferred_notifications(
        _session(profile_home=str(tmp_path)), {event_id}
    )

    assert attempts == [(event_id,), (event_id,)]
    assert async_delegation.load_deferred_notifications(
        "session-key", db_path=db_path
    ) == []
    assert event_id not in _adoption_event_ids(db_path)


def test_adoption_cleanup_is_scoped_and_does_not_accumulate(tmp_path, monkeypatch):
    """Delivered proofs vanish; an unrelated proof is never swept with them."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from tools import async_delegation

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("session-key", source="tui")
    unrelated_id = "async_delegation:unrelated-proof"
    db.append_message(
        "session-key",
        "assistant",
        "unrelated",
        deferred_notification_ids=[unrelated_id],
    )

    for index in range(12):
        event_id = f"async_delegation:retention-{index}"
        event = {
            "type": "async_delegation",
            "delegation_id": f"retention-{index}",
            "session_key": "session-key",
        }
        assert async_delegation.persist_deferred_notification(
            "session-key",
            event_id,
            f"result {index}",
            event,
            db_path=db_path,
        )
        db.append_message(
            "session-key",
            "assistant",
            f"answer {index}",
            deferred_notification_ids=[event_id],
        )
        assert async_delegation.complete_deferred_notifications(
            "session-key", [event_id], db_path=db_path
        )
        assert async_delegation.complete_deferred_notifications(
            "session-key", [event_id], db_path=db_path
        )
        assert _adoption_event_ids(db_path) == [unrelated_id]

    db.close()


def test_compression_between_claim_and_ack_migrates_owner_and_delivers_once(
    tmp_path, monkeypatch
):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-compress-before-ack"
    )
    from hermes_state import SessionDB

    parent = "session-key"
    child = "session-key-compressed"
    event_id = "async_delegation:deleg-compress-before-ack"
    db = SessionDB(tmp_path / "state.db")
    db.create_session(parent, source="tui")
    db.end_session(parent, "compression")
    db.create_session(child, source="tui", parent_session_id=parent)
    assert async_delegation.persist_deferred_notification(
        parent,
        event_id,
        "durable result",
        event,
        db_path=tmp_path / "state.db",
    )

    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = parent

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            task_id=None,
            persist_user_message=None,
            deferred_notification_ids=None,
        ):
            calls.append(
                {
                    "prompt": prompt,
                    "persist_user_message": persist_user_message,
                    "ids": list(deferred_notification_ids or ()),
                }
            )
            db.append_message(parent, "user", persist_user_message)
            self.session_id = child
            db.append_message(
                child,
                "assistant",
                "answer",
                deferred_notification_ids=deferred_notification_ids,
            )
            return _successful_turn(prompt, conversation_history)

    session = _session(
        agent=_Agent(),
        profile_home=str(tmp_path),
        deferred_notification_texts=["durable result"],
        deferred_notification_event_ids={event_id},
        defer_notifications_until_user=True,
    )
    emitted = []
    real_sync = server._sync_session_key_after_compress
    _patch_prompt_turn_runtime(monkeypatch, emitted)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", real_sync)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_kw: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_kw: None)

    server._run_prompt_submit("rid", "sid", session, "Next request", origin="user")

    assert len(calls) == 1
    assert calls[0]["prompt"].count("durable result") == 1
    assert calls[0]["persist_user_message"] == "Next request"
    assert calls[0]["ids"] == [event_id]
    assert session["session_key"] == child
    assert async_delegation.load_deferred_notifications(
        parent, db_path=tmp_path / "state.db"
    ) == []
    assert async_delegation.load_deferred_notifications(
        child, db_path=tmp_path / "state.db"
    ) == []
    assert not async_delegation.complete_deferred_notifications(
        parent, [event_id], db_path=tmp_path / "state.db"
    )
    assert async_delegation.complete_deferred_notifications(
        child, [event_id], db_path=tmp_path / "state.db"
    )

    recreated = server._deferred_session_record(
        child,
        cols=80,
        cwd=".",
        history=list(session["history"]),
        lease=None,
        profile_home=tmp_path,
    )
    assert recreated["deferred_notification_texts"] == []
    assert recreated["defer_notifications_until_user"] is False
    db.close()


def test_deferred_ack_fails_closed_for_missing_or_wrong_owner(tmp_path, monkeypatch):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-owner-fence"
    )
    event_id = "async_delegation:deleg-owner-fence"
    assert async_delegation.persist_deferred_notification(
        "session-key",
        event_id,
        "durable result",
        event,
        db_path=tmp_path / "state.db",
    )

    assert not async_delegation.complete_deferred_notifications(
        "wrong-owner", [event_id], db_path=tmp_path / "state.db"
    )
    assert not async_delegation.complete_deferred_notifications(
        "session-key", ["missing-event"], db_path=tmp_path / "state.db"
    )
    assert [
        row["event_id"]
        for row in async_delegation.load_deferred_notifications(
            "session-key", db_path=tmp_path / "state.db"
        )
    ] == [event_id]


def test_durable_claim_releases_when_turn_fails_before_acceptance(tmp_path, monkeypatch):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-before-accept"
    )
    session = _session()
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("start failed")),
    )

    with pytest.raises(RuntimeError, match="start failed"):
        server._dispatch_notification_turn(
            "rid",
            "sid",
            session,
            "durable result",
            event=event,
            consumer="test-before-accept",
        )

    assert session["running"] is False
    durable = async_delegation.get_durable_delegation("deleg-before-accept")
    assert durable["delivery_state"] == "pending"
    retry_claim = async_delegation.claim_event_delivery(event, "test-retry")
    assert retry_claim
    assert async_delegation.release_event_delivery(event, retry_claim)


def test_dispatched_durable_ack_failure_keeps_live_turn_reserved(tmp_path, monkeypatch):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-live-ack"
    )
    emitted = []
    started = threading.Event()
    finish = threading.Event()
    acknowledged = threading.Event()
    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            calls.append(prompt)
            started.set()
            if not finish.wait(3):
                raise TimeoutError("test did not release live turn")
            return _successful_turn(prompt, conversation_history)

    session = _session(agent=_Agent())
    _patch_prompt_turn_runtime(
        monkeypatch,
        emitted,
        immediate_thread=False,
        disable_post_turn_drain=True,
    )
    attempts = []
    original_complete = async_delegation.complete_event_delivery

    def flaky_complete(evt, claim_id):
        attempts.append(claim_id)
        if len(attempts) == 1:
            raise OSError("temporary sqlite failure")
        completed = original_complete(evt, claim_id)
        if completed:
            acknowledged.set()
        return completed

    monkeypatch.setattr(async_delegation, "complete_event_delivery", flaky_complete)
    monkeypatch.setattr(server, "_NOTIFICATION_ACK_RETRY_DELAYS", (0.0,))

    assert server._dispatch_notification_turn(
        "rid",
        "sid",
        session,
        "durable result",
        event=event,
        consumer="test-live",
    ) == "dispatched"
    run_thread = session["_run_thread"]

    assert started.wait(3)
    assert acknowledged.wait(3)
    assert session["running"] is True
    assert server._dispatch_notification_turn(
        "rid-duplicate",
        "sid",
        session,
        "durable result",
        event=event,
        consumer="test-live-duplicate",
    ) == "busy"

    finish.set()
    run_thread.join(3)
    assert not run_thread.is_alive()
    assert calls == ["durable result"]
    assert session["running"] is False
    assert async_delegation.get_durable_delegation("deleg-live-ack")[
        "delivery_state"
    ] == "delivered"


def test_notification_reservation_wins_foreground_submit_barrier(tmp_path, monkeypatch):
    async_delegation, event = _durable_completion(
        tmp_path, monkeypatch, "deleg-reservation-race"
    )
    claim_entered = threading.Event()
    release_claim = threading.Event()
    notification_calls = []
    errors = []

    class _ObservedLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.waiter = threading.Event()

        def __enter__(self):
            if self._lock.locked():
                self.waiter.set()
            self._lock.acquire()
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self._lock.release()

    class _Agent:
        def __init__(self):
            self.interrupts = 0

        def interrupt(self):
            self.interrupts += 1

    observed_lock = _ObservedLock()
    agent = _Agent()
    session = _session(agent=agent, history_lock=observed_lock)
    original_claim = async_delegation.claim_event_delivery

    def blocked_claim(evt, consumer):
        claim_id = original_claim(evt, consumer)
        claim_entered.set()
        if not release_claim.wait(3):
            raise TimeoutError("test did not release durable claim")
        return claim_id

    monkeypatch.setattr(async_delegation, "claim_event_delivery", blocked_claim)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *args, **kwargs: notification_calls.append((args, kwargs)),
    )
    server._sessions["sid-race"] = session
    notification_result = {}
    foreground_result = {}

    def dispatch_notification():
        try:
            notification_result["value"] = server._dispatch_notification_turn(
                "rid-notification",
                "sid-race",
                session,
                "background result",
                event=event,
                consumer="test-reservation",
            )
        except BaseException as exc:
            errors.append(exc)

    def submit_foreground():
        try:
            foreground_result["value"] = server._methods["prompt.submit"](
                "rid-user", {"session_id": "sid-race", "text": "user request"}
            )
        except BaseException as exc:
            errors.append(exc)

    notification_thread = threading.Thread(target=dispatch_notification)
    foreground_thread = threading.Thread(target=submit_foreground)
    try:
        notification_thread.start()
        assert claim_entered.wait(3)
        foreground_thread.start()
        assert observed_lock.waiter.wait(3)
        release_claim.set()
        notification_thread.join(3)
        foreground_thread.join(3)

        assert not errors
        assert notification_result["value"] == "dispatched"
        assert foreground_result["value"]["result"]["status"] == "queued"
        assert session["queued_prompt"]["text"] == "user request"
        assert agent.interrupts == 1
        assert len(notification_calls) == 1
        assert notification_calls[0][1]["origin"] == "notification"
        assert async_delegation.get_durable_delegation("deleg-reservation-race")[
            "delivery_state"
        ] == "delivered"
    finally:
        release_claim.set()
        notification_thread.join(3)
        foreground_thread.join(3)
        server._sessions.pop("sid-race", None)


def test_delayed_final_session_info_keeps_older_generation(monkeypatch):
    emitted = []
    old_info_waiting = threading.Event()
    release_old_info = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            calls.append(prompt)
            if len(calls) == 2:
                second_started.set()
                if not release_second.wait(3):
                    raise TimeoutError("test did not release second turn")
            return _successful_turn(prompt, conversation_history)

    session = _session(agent=_Agent(), running=True)
    _patch_prompt_turn_runtime(
        monkeypatch,
        emitted,
        immediate_thread=False,
        disable_post_turn_drain=True,
    )

    def barrier_emit(event_type, sid, payload=None):
        if (
            event_type == "session.info"
            and payload.get("turn_generation") == 1
            and payload.get("running") is False
        ):
            old_info_waiting.set()
            if not release_old_info.wait(3):
                raise TimeoutError("test did not release stale session.info")
        emitted.append((event_type, sid, payload or {}))

    monkeypatch.setattr(server, "_emit", barrier_emit)

    server._run_prompt_submit("rid-1", "sid", session, "first", origin="user")
    first_thread = session["_run_thread"]
    assert old_info_waiting.wait(3)

    with session["history_lock"]:
        assert session["running"] is False
        session["running"] = True
    server._run_prompt_submit("rid-2", "sid", session, "second", origin="notification")
    second_thread = session["_run_thread"]
    assert second_started.wait(3)

    start_two = next(
        event
        for event in emitted
        if event[0] == "message.start" and event[2].get("turn_generation") == 2
    )
    assert start_two[2]["turn_origin"] == "notification"
    assert not [
        event
        for event in emitted
        if event[0] == "session.info" and event[2].get("turn_generation") == 1
    ]

    release_old_info.set()
    release_second.set()
    first_thread.join(3)
    second_thread.join(3)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    stale_info_index = next(
        index
        for index, event in enumerate(emitted)
        if event[0] == "session.info" and event[2].get("turn_generation") == 1
    )
    start_two_index = emitted.index(start_two)
    assert start_two_index < stale_info_index
    stale_info = emitted[stale_info_index][2]
    assert stale_info["running"] is False
    assert stale_info["turn_origin"] is None


def test_live_session_reconnect_preserves_and_settles_turn_generation(monkeypatch):
    agent = types.SimpleNamespace(
        model="test-model",
        provider="test-provider",
        session_id="session-key",
    )
    session = _session(agent=agent)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    with session["history_lock"]:
        session["running"] = True
        turn_token = server._set_turn_origin_locked(session, "notification")

    active = server._live_session_payload("sid", session)
    assert active["running"] is True
    assert active["info"]["running"] is True
    assert active["info"]["turn_generation"] == turn_token
    assert active["info"]["turn_state_revision"] == 1
    assert active["info"]["turn_origin"] == "notification"

    with session["history_lock"]:
        session["running"] = False
        assert server._clear_turn_origin_locked(session, turn_token)

    settled = server._live_session_payload("sid", session)
    assert settled["running"] is False
    assert settled["info"]["running"] is False
    assert settled["info"]["turn_generation"] == turn_token
    assert settled["info"]["turn_state_revision"] == 2
    assert settled["info"]["turn_origin"] is None


def test_message_events_advance_turn_state_revision_on_start_and_settle(monkeypatch):
    emitted = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            return _successful_turn(prompt, conversation_history)

    session = _session(agent=_Agent(), running=True)
    _patch_prompt_turn_runtime(monkeypatch, emitted)

    server._run_prompt_submit("rid", "sid", session, "hello", origin="user")

    start = next(payload for event, _sid, payload in emitted if event == "message.start")
    complete = next(
        payload for event, _sid, payload in emitted if event == "message.complete"
    )
    settled = [
        payload for event, _sid, payload in emitted if event == "session.info"
    ][-1]
    assert start["turn_generation"] == complete["turn_generation"] == 1
    assert start["turn_state_revision"] == 1
    assert complete["turn_state_revision"] == 2
    assert settled["turn_state_revision"] == 2
    assert settled["running"] is False


def test_notification_preemption_drains_user_before_later_completion(monkeypatch):
    isolated_queue = queue.Queue()
    emitted = []
    first_started = threading.Event()
    release_first = threading.Event()
    third_settled = threading.Event()
    calls = []

    class _Agent:
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""
        service_tier = ""
        session_id = "session-key"

        def __init__(self):
            self.interrupts = 0

        def clear_interrupt(self):
            return None

        def interrupt(self):
            self.interrupts += 1

        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                first_started.set()
                if not release_first.wait(3):
                    raise TimeoutError("test did not release notification turn")
                result = _successful_turn(prompt, conversation_history)
                result["interrupted"] = True
                return result
            return _successful_turn(prompt, conversation_history)

    agent = _Agent()
    session = _session(agent=agent)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    _patch_prompt_turn_runtime(
        monkeypatch,
        emitted,
        immediate_thread=False,
        disable_post_turn_drain=False,
    )

    def recording_emit(event_type, sid, payload=None):
        emitted.append((event_type, sid, payload or {}))
        if (
            event_type == "session.info"
            and payload.get("turn_generation") == 3
            and payload.get("running") is False
        ):
            third_settled.set()

    monkeypatch.setattr(server, "_emit", recording_emit)
    process_registry._completion_consumed.discard("proc-after-user")
    server._sessions["sid"] = session
    try:
        assert server._dispatch_notification_turn(
            "rid-notification",
            "sid",
            session,
            "first background result",
        ) == "dispatched"
        assert first_started.wait(3)

        response = server._methods["prompt.submit"](
            "rid-user", {"session_id": "sid", "text": "foreground request"}
        )
        assert response["result"]["status"] == "queued"
        assert agent.interrupts == 1

        isolated_queue.put(_completion("proc-after-user"))
        release_first.set()
        assert third_settled.wait(5)
        latest_thread = session["_run_thread"]
        latest_thread.join(3)

        assert calls[0] == "first background result"
        assert calls[1] == "foreground request"
        assert "proc-after-user" in calls[2]
        assert len(calls) == 3
        starts = [event[2] for event in emitted if event[0] == "message.start"]
        assert [event["turn_origin"] for event in starts] == [
            "notification",
            "user",
            "notification",
        ]
        assert [event["turn_generation"] for event in starts] == [1, 2, 3]
        assert isolated_queue.empty()
        assert session["running"] is False
    finally:
        release_first.set()
        run_thread = session.get("_run_thread")
        if run_thread is not None:
            run_thread.join(3)
        server._sessions.pop("sid", None)
        process_registry._completion_consumed.discard("proc-after-user")
