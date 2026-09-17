from __future__ import annotations

import json
import threading
import time
import types

import pytest

from agent.turn_authorization import (
    TurnAuthorization,
    current_fizko_authorization_header,
)
from tui_gateway import server as srv


def _authorization(raw, *, expires_at=None, principal_id="a" * 64, admission_id="1" * 32):
    if raw is None:
        return TurnAuthorization.from_raw(None)
    return TurnAuthorization.from_raw(
        raw,
        expires_at=time.time() + 3600 if expires_at is None else expires_at,
        principal_id=principal_id,
        admission_id=admission_id,
    )


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
        self.name = name

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent):
    return {
        "agent": agent,
        "session_key": "agent-session-key",
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
        "inflight_turn": None,
        "transport": None,
    }


def test_prompt_submit_rejects_malformed_token_after_popping_it():
    params = {
        "session_id": "missing",
        "text": "hello",
        "_fizko_person_access_token": "contains whitespace",
    }

    response = srv._methods["prompt.submit"]("r", params)

    assert response["error"]["code"] == 4004
    assert params == {"session_id": "missing", "text": "hello"}


def test_prompt_submit_rejects_person_token_without_expiry():
    response = srv._methods["prompt.submit"](
        "r",
        {
            "session_id": "missing",
            "text": "hello",
            "_fizko_person_access_token": "person-token",
        },
    )

    assert response["error"]["code"] == 4004
    assert "expiry is required" in response["error"]["message"]


def test_prompt_submit_requires_private_admission_id():
    response = srv._methods["prompt.submit"](
        "r",
        {
            "session_id": "missing",
            "text": "hello",
            "_fizko_person_access_token": "person-token",
            "_fizko_person_access_token_expires_at": time.time() + 3600,
            "_fizko_person_principal_id": "a" * 64,
        },
    )

    assert response["error"]["code"] == 4004
    assert "admission id is required" in response["error"]["message"]


def test_prompt_submit_rejects_expired_person_token():
    response = srv._methods["prompt.submit"](
        "r",
        {
            "session_id": "missing",
            "text": "hello",
            "_fizko_person_access_token": "person-token",
            "_fizko_person_access_token_expires_at": time.time() - 1,
            "_fizko_person_principal_id": "a" * 64,
            "_fizko_person_admission_id": "0" * 32,
        },
    )

    assert response["error"]["code"] == 4004
    assert "expired" in response["error"]["message"]


def test_failed_admission_delivery_can_retry_without_duplicate_terminal(monkeypatch):
    holder = _authorization("person", admission_id="e" * 32)
    attempts = []

    def emit(*args):
        attempts.append(args)
        return len(attempts) > 1

    monkeypatch.setattr(srv, "_emit", emit)

    assert not srv._emit_person_admission("sid", holder, "terminal", reason="failed")
    assert srv._emit_person_admission("sid", holder, "terminal", reason="failed")
    assert not srv._emit_person_admission("sid", holder, "terminal", reason="failed")
    assert len(attempts) == 2


def test_raising_admission_delivery_does_not_break_cleanup_and_can_retry(monkeypatch):
    holder = _authorization("person", admission_id="f" * 32)
    attempts = []

    def emit(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise BrokenPipeError("transport closed")
        return True

    monkeypatch.setattr(srv, "_emit", emit)

    assert not srv._emit_person_admission("sid", holder, "terminal", reason="failed")
    assert srv._emit_person_admission("sid", holder, "terminal", reason="failed")
    assert len(attempts) == 2


def test_batch_admission_delivery_failure_does_not_abort_remaining_cleanup(monkeypatch):
    first = _authorization("first", admission_id="d" * 32)
    second = _authorization("second", admission_id="e" * 32)
    delivered = []

    def emit(_event, _sid, payload):
        if payload["admission_id"] == "d" * 32:
            raise RuntimeError("dead transport")
        delivered.append(payload)
        return True

    monkeypatch.setattr(srv, "_emit", emit)

    srv._emit_person_admissions("sid", [first, second], reason="session_finalized")

    assert delivered == [{
        "admission_id": "e" * 32,
        "status": "terminal",
        "reason": "session_finalized",
    }]


def test_person_authorized_submit_fails_closed_when_compute_isolation_is_required(monkeypatch):
    admission_id = "1" * 32
    session = _session(types.SimpleNamespace())
    events = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: True)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    srv._sessions["sid"] = session
    try:
        response = srv._methods["prompt.submit"](
            "r",
            {
                "session_id": "sid",
                "text": "hello",
                "_fizko_person_access_token": "person-token",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "a" * 64,
                "_fizko_person_admission_id": admission_id,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4126
    assert session["running"] is False
    assert "_active_turn_authorization" not in session
    assert events == [(
        "person.admission", "sid", {
            "admission_id": admission_id,
            "status": "terminal",
            "reason": "admission_rejected",
        },
    )]


def test_personal_submit_thread_start_failure_terminally_releases_admission(monkeypatch):
    admission_id = "f" * 32
    session = _session(types.SimpleNamespace())
    session["agent_ready"] = threading.Event()
    events = []

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(
        srv, "_persist_session_row_for_submit", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(srv, "_restart_completed_failed_agent_build", lambda *_args: True)
    monkeypatch.setattr(srv.threading, "Thread", FailingThread)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    srv._sessions["sid"] = session
    try:
        with pytest.raises(RuntimeError, match="thread unavailable"):
            srv._methods["prompt.submit"](
                "r",
                {
                    "session_id": "sid",
                    "text": "hello",
                    "_fizko_person_access_token": "person-token",
                    "_fizko_person_access_token_expires_at": time.time() + 3600,
                    "_fizko_person_principal_id": "a" * 64,
                    "_fizko_person_admission_id": admission_id,
                },
            )
    finally:
        srv._sessions.pop("sid", None)

    assert session["running"] is False
    assert "_active_turn_authorization" not in session
    assert events == [(
        "person.admission", "sid", {
            "admission_id": admission_id,
            "status": "terminal",
            "reason": "submit_failed",
        },
    )]


def test_queued_person_authorized_turn_does_not_bypass_compute_isolation(monkeypatch):
    holder = _authorization("queued-person")
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "queued work",
        "transport": None,
        "turn_authorization": holder,
    }
    events = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: True)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))

    assert srv._drain_queued_prompt("r", "sid", session) is True

    assert session["running"] is False
    assert session.get("queued_prompt") is None
    assert "_active_turn_authorization" not in session
    assert any(event[0] == "error" for event in events)
    assert any(event[0] == "person.admission" for event in events)


def test_expired_queued_person_authorization_is_dropped_fail_closed(monkeypatch):
    holder = _authorization("expired-person", expires_at=time.time() - 1)
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "queued work",
        "transport": None,
        "turn_authorization": holder,
    }
    events = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: pytest.fail("expired queued prompt was dispatched"),
    )

    assert srv._drain_queued_prompt("r", "sid", session) is True

    assert session["running"] is False
    assert session.get("queued_prompt") is None
    assert "_active_turn_authorization" not in session
    errors = [event for event in events if event[0] == "error"]
    assert errors and "expired" in errors[0][2]["message"]
    assert any(event[0] == "person.admission" for event in events)


def test_expired_queued_head_does_not_block_later_static_prompt(monkeypatch):
    admission_id = "e" * 32
    expired = _authorization(
        "expired-person", expires_at=time.time() - 1, admission_id=admission_id
    )
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "expired work", "transport": None, "turn_authorization": expired,
    }
    session["queued_prompts"] = [{"text": "ordinary work", "transport": None}]
    events = []
    dispatched = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **kwargs: dispatched.append((text, kwargs)) or True,
    )

    assert srv._drain_queued_prompt("r", "sid", session) is True

    assert [text for text, _kwargs in dispatched] == ["ordinary work"]
    ordinary_authorization = dispatched[0][1]["turn_authorization"]
    assert ordinary_authorization.is_personal is False
    assert ordinary_authorization.has_token is False
    assert any(
        "expired" in args[2]["message"]
        for args in events
        if args[0] == "error"
    )
    assert (
        "person.admission",
        "sid",
        {"admission_id": admission_id, "status": "terminal", "reason": "expired"},
    ) in events


def test_rejected_queue_batch_schedules_bounded_continuation(monkeypatch):
    expired = _authorization("expired-person", expires_at=time.time() - 1)
    rejected = {"text": "expired", "transport": None, "turn_authorization": expired}
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = dict(rejected)
    session["queued_prompts"] = [dict(rejected) for _ in range(32)] + [
        {"text": "ordinary", "transport": None}
    ]
    dispatched = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_emit", lambda *_args: None)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **kwargs: dispatched.append(text) or True,
    )

    class _ImmediateThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(srv.threading, "Thread", _ImmediateThread)

    assert srv._drain_queued_prompt("r", "sid", session) is True
    assert dispatched == ["ordinary"]
    assert session.get("queued_prompt") is None


def test_personal_post_turn_steer_keeps_blocked_descendant_marker(monkeypatch):
    session = _session(types.SimpleNamespace())
    blocked = TurnAuthorization.blocked()
    monkeypatch.setattr(srv, "_drain_queued_prompt", lambda *_args: True)

    srv._run_post_turn_followups(
        "r",
        "sid",
        session,
        {"pending_steer": "continue safely"},
        None,
        descendant_authorization=blocked,
    )

    queued = session["queued_prompt"]
    assert queued["turn_authorization"] is blocked
    assert queued["turn_authorization"].is_personal is True
    assert queued["turn_authorization"].has_token is False


def test_prompt_submit_pops_token_scopes_it_to_run_and_resets_without_leaks(monkeypatch, tmp_path):
    secret = "person-token-never-persist"
    seen = []
    events = []

    def run_conversation(user_message, **kwargs):
        seen.append((user_message, current_fizko_authorization_header(), kwargs))
        return {"final_response": "ok"}

    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    params = {
        "session_id": "sid",
        "text": "hello",
        "_fizko_person_access_token": secret,
        "_fizko_person_access_token_expires_at": time.time() + 3600,
        "_fizko_person_principal_id": "a" * 64,
        "_fizko_person_admission_id": "2" * 32,
    }
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *a, **k: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    srv._sessions["sid"] = session
    try:
        response = srv._methods["prompt.submit"]("r", params)
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"]["status"] == "streaming"
    assert params == {"session_id": "sid", "text": "hello"}
    assert seen[0][0:2] == ("hello", "Bearer " + secret)
    assert secret not in json.dumps(seen[0][2], default=repr)
    assert secret not in json.dumps(session, default=repr)
    assert secret not in json.dumps(events, default=repr)
    assert current_fizko_authorization_header() == ""
    admission_events = [event for event in events if event[0] == "person.admission"]
    assert admission_events == [
        ("person.admission", "sid", {"admission_id": "2" * 32, "status": "started"}),
        (
            "person.admission",
            "sid",
            {"admission_id": "2" * 32, "status": "terminal", "reason": "finished"},
        ),
    ]


def test_different_person_cannot_redirect_or_interrupt_the_active_turn(monkeypatch):
    redirects = []
    interrupts = []
    active = _authorization("active-person")
    incoming = _authorization("incoming-person")
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: redirects.append(text) or True,
        interrupt=lambda: interrupts.append(True),
    )
    session = _session(agent)
    session.update(running=True, _active_turn_authorization=active)
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(srv, "_interrupt_busy_session", lambda *args: interrupts.append(True))
    srv._sessions["sid"] = session
    try:
        response = srv._methods["prompt.submit"](
            "r",
            {
                "session_id": "sid",
                "text": "do not steer active",
                "_fizko_person_access_token": "incoming-person",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "b" * 64,
                "_fizko_person_admission_id": "3" * 32,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"] == {"status": "queued"}
    assert redirects == []
    assert interrupts == []
    queued = session["queued_prompt"]
    assert queued["turn_authorization"].same_credential(incoming)
    assert "incoming-person" not in repr(queued)


@pytest.mark.parametrize(
    ("mode", "active_raw", "incoming_raw"),
    [
        ("interrupt", "same-person", "same-person"),
        ("steer", "same-person", "same-person"),
        ("interrupt", None, "incoming-person"),
        ("steer", None, "incoming-person"),
    ],
)
def test_any_token_bearing_busy_submit_queues_without_live_agent_mutation(
    monkeypatch, mode, active_raw, incoming_raw,
):
    calls = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: calls.append(("redirect", text)) or True,
        steer=lambda text: calls.append(("steer", text)) or True,
        interrupt=lambda: calls.append(("interrupt", None)),
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_authorization=_authorization(active_raw),
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: mode)
    monkeypatch.setattr(
        srv,
        "_interrupt_busy_session",
        lambda *_args: calls.append(("hard-interrupt", None)),
    )

    response = srv._handle_busy_submit(
        "r", "sid", session, "wait your turn", None,
        turn_authorization=_authorization(incoming_raw),
    )

    assert response["result"] == {"status": "queued"}
    assert calls == []
    assert session["queued_prompt"]["text"] == "wait your turn"


def test_authorization_clears_when_turn_is_cancelled_before_agent_ready(monkeypatch):
    holder = _authorization("cancelled-person")
    session = _session(types.SimpleNamespace())
    session.update(running=True, _active_turn_authorization=holder, _active_turn_route="inline")
    events = []
    monkeypatch.setattr(srv, "_wait_agent_for_prompt", lambda *args: {"error": {"message": "cancelled"}})
    monkeypatch.setattr(srv, "_emit_terminal_turn_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(srv, "_emit", lambda *args, **kwargs: events.append(args))
    monkeypatch.setattr(srv, "_session_info", lambda *args: {})

    srv._run_after_agent_ready("r", "sid", session, "hello", None, None, None, holder)

    assert session["running"] is False
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session
    assert current_fizko_authorization_header() == ""
    assert events.count((
        "person.admission", "sid", {
            "admission_id": "1" * 32,
            "status": "terminal",
            "reason": "agent_not_ready",
        },
    )) == 1


@pytest.mark.parametrize("failure_stage", ["readiness", "dispatch"])
def test_exception_after_streaming_terminally_clears_active_admission(
    monkeypatch, failure_stage,
):
    holder = _authorization("failed-person", admission_id="8" * 32)
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        _active_turn_authorization=holder,
        _active_turn_route="inline",
    )
    events = []

    if failure_stage == "readiness":
        monkeypatch.setattr(
            srv,
            "_wait_agent_for_prompt",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("readiness failed")),
        )
    else:
        monkeypatch.setattr(srv, "_wait_agent_for_prompt", lambda *_args: None)
        monkeypatch.setattr(
            srv,
            "_run_prompt_submit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatch failed")),
        )
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))

    with pytest.raises(RuntimeError, match="failed"):
        srv._run_after_agent_ready(
            "r", "sid", session, "hello", None, None, None, holder
        )

    assert session["running"] is False
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session
    assert sum(event[0] == "person.admission" for event in events) == 1


def test_cancel_after_streaming_terminally_accounts_person_admission(monkeypatch):
    holder = _authorization("cancelled-person", admission_id="9" * 32)
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        _turn_cancel_requested=True,
        _active_turn_authorization=holder,
        _active_turn_route="inline",
    )
    events = []
    monkeypatch.setattr(srv, "_wait_agent_for_prompt", lambda *args: None)
    monkeypatch.setattr(srv, "_emit", lambda *args, **kwargs: events.append(args))

    srv._run_after_agent_ready("r", "sid", session, "hello", None, None, None, holder)

    assert events.count((
        "person.admission", "sid", {
            "admission_id": "9" * 32,
            "status": "terminal",
            "reason": "cancelled_before_ready",
        },
    )) == 1
    assert "_active_turn_authorization" not in session


def test_missing_queue_authorization_never_merges_into_token_envelope():
    holder = _authorization("queued-person")
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "personal work",
        "transport": None,
        "turn_authorization": holder,
    }

    srv._enqueue_prompt(session, "internal follow-up", None)

    assert session["queued_prompt"]["text"] == "personal work"
    assert session["queued_prompt"]["turn_authorization"] is holder
    assert session["queued_prompts"] == [
        {"text": "internal follow-up", "transport": None}
    ]


def test_queued_prompts_keep_distinct_person_authorizations(monkeypatch):
    active = _authorization("person-a")
    session = _session(types.SimpleNamespace())
    session.update(running=True, _active_turn_authorization=active)
    monkeypatch.setattr(srv, "_interrupt_busy_session", lambda *args: None)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *args: False)
    dispatched = []
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda rid, sid, st, text, **kwargs: (
            dispatched.append((text, kwargs["turn_authorization"])) or True
        ),
    )
    srv._sessions["sid"] = session
    try:
        for person in ("person-b", "person-c"):
            response = srv._methods["prompt.submit"](
                "r",
                {
                    "session_id": "sid",
                    "text": f"from {person}",
                    "queued": True,
                    "_fizko_person_access_token": person,
                    "_fizko_person_access_token_expires_at": time.time() + 3600,
                    "_fizko_person_principal_id": ("b" if person == "person-b" else "c") * 64,
                    "_fizko_person_admission_id": ("4" if person == "person-b" else "5") * 32,
                },
            )
            assert response["result"] == {"status": "queued"}
        assert session["queued_prompt"]["text"] == "from person-b"
        assert [item["text"] for item in session["queued_prompts"]] == ["from person-c"]

        for _ in range(2):
            session["running"] = False
            session.pop("_active_turn_authorization", None)
            assert srv._drain_queued_prompt("d", "sid", session)
    finally:
        srv._sessions.pop("sid", None)

    assert [text for text, _holder in dispatched] == ["from person-b", "from person-c"]
    assert dispatched[0][1].same_credential(_authorization("person-b"))
    assert dispatched[1][1].same_credential(_authorization("person-c"))


def test_same_text_from_different_person_is_not_deduplicated(monkeypatch):
    active = _authorization("person-a")
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        _active_turn_authorization=active,
        inflight_turn={"user": "same words", "assistant": "", "streaming": True, "error": ""},
    )
    monkeypatch.setattr(srv, "_interrupt_busy_session", lambda *args: None)
    srv._sessions["sid"] = session
    try:
        response = srv._methods["prompt.submit"](
            "r",
            {
                "session_id": "sid",
                "text": "same words",
                "queued": True,
                "_fizko_person_access_token": "person-b",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "b" * 64,
                "_fizko_person_admission_id": "6" * 32,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"] == {"status": "queued"}
    assert session["queued_prompt"]["text"] == "same words"


def test_personal_same_principal_prompts_are_never_merged_or_deduplicated():
    active = _authorization("person-a", admission_id="a" * 32)
    first = _authorization("person-b", admission_id="b" * 32)
    second = _authorization("person-c", admission_id="c" * 32)
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        _active_turn_authorization=active,
        inflight_turn={"user": "same words", "assistant": "", "streaming": True, "error": ""},
    )

    srv._enqueue_prompt(session, "same words", None, turn_authorization=first)
    srv._enqueue_prompt(session, "same words", None, turn_authorization=second)

    assert session["queued_prompt"]["text"] == "same words"
    assert session["queued_prompt"]["turn_authorization"] is first
    assert session["queued_prompts"] == [
        {"text": "same words", "transport": None, "turn_authorization": second}
    ]


@pytest.mark.parametrize("failure", ["marker", "finalizer"])
def test_prompt_worker_failure_always_releases_generation(
    monkeypatch, tmp_path, failure
):
    holder = TurnAuthorization.from_raw(None)
    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=lambda prompt, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
            ],
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="failure-turn",
        _active_turn_authorization=holder,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    if failure == "marker":
        monkeypatch.setattr(
            srv,
            "_record_turn_marker",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("marker unavailable")
            ),
        )
    else:
        monkeypatch.setattr(
            srv,
            "_finish_turn",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("tts sentinel failed")
            ),
        )
        monkeypatch.setattr(
            srv,
            "_hook_failure",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("diagnostic sink failed")
            ),
        )

    assert srv._run_prompt_submit(
        "r", "sid", session, "hello",
        turn_authorization=holder, expected_turn_id="failure-turn",
    )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session
    assert "_run_thread" not in session
    assert "_run_thread_turn_id" not in session


@pytest.mark.parametrize("failure", ["transport", "started_event"])
def test_worker_setup_failure_releases_admitted_generation(
    monkeypatch, tmp_path, failure
):
    holder = TurnAuthorization.from_raw(None)
    agent = types.SimpleNamespace(session_id="a", clear_interrupt=lambda: None)
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="bind-failure-turn",
        _active_turn_authorization=holder,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    if failure == "transport":
        monkeypatch.setattr(
            srv,
            "bind_transport",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("transport binding failed")
            ),
        )
    else:
        real_admission_emit = srv._emit_person_admission

        def fail_started(*args, **kwargs):
            if args[2] == "started":
                raise RuntimeError("started admission event failed")
            return real_admission_emit(*args, **kwargs)

        monkeypatch.setattr(srv, "_emit_person_admission", fail_started)

    assert srv._run_prompt_submit(
        "r",
        "sid",
        session,
        "hello",
        turn_authorization=holder,
        expected_turn_id="bind-failure-turn",
    )
    with session["history_lock"]:
        worker = session.get("_run_thread")
    if worker is not None:
        worker.join(timeout=5.0)
    for _ in range(100):
        with session["history_lock"]:
            if "_run_thread" not in session:
                break
        time.sleep(0.01)
    else:
        pytest.fail("failed setup worker did not exit")

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_followup_diagnostic_failure_releases_exact_generation(monkeypatch):
    authorization = TurnAuthorization.from_raw(None)
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        _active_turn_id="followup-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("followup dispatch failed")
        ),
    )
    monkeypatch.setattr(
        srv,
        "_hook_failure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("diagnostic failed")
        ),
    )

    srv._dispatch_followup_turn(
        "rid",
        "sid",
        session,
        "follow up",
        "goal followup",
        expected_turn_id="followup-turn",
        turn_authorization=authorization,
    )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_worker_revalidates_generation_after_marker_publication(monkeypatch):
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="6" * 32)
    agent = types.SimpleNamespace(clear_interrupt=lambda: None)
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=old_authorization,
        _active_turn_route="inline",
    )
    prepared = []

    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *_args: True)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_emit_person_admission", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_finish_turn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *_args, **_kwargs: None)

    def stale_marker(*_args, **_kwargs):
        with session["history_lock"]:
            session.update(
                running=True,
                _active_turn_id="new-turn",
                _active_turn_authorization=new_authorization,
                _active_turn_route="inline",
                _active_turn_marker_key="new-marker",
            )
            srv._start_inflight_turn(session, "new prompt")
        return "old-marker"

    monkeypatch.setattr(srv, "_record_turn_marker", stale_marker)
    monkeypatch.setattr(
        srv,
        "_prepare_turn_input",
        lambda *_args, **_kwargs: prepared.append("old") or None,
    )

    assert srv._run_prompt_submit(
        "rid",
        "sid",
        session,
        "old prompt",
        turn_authorization=old_authorization,
        expected_turn_id="old-turn",
    )

    assert prepared == []
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization
    assert session["_active_turn_marker_key"] == "new-marker"
    assert session["inflight_turn"]["user"] == "new prompt"


def test_authorization_resets_after_agent_error(monkeypatch, tmp_path):
    holder = _authorization("error-person")
    events = []

    def fail(*args, **kwargs):
        assert current_fizko_authorization_header() == "Bearer error-person"
        raise RuntimeError("boom")

    agent = types.SimpleNamespace(session_id="a", run_conversation=fail, clear_interrupt=lambda: None)
    session = _session(agent)
    session.update(running=True, _active_turn_authorization=holder)
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})

    assert srv._run_prompt_submit("r", "sid", session, "hello", turn_authorization=holder)

    assert current_fizko_authorization_header() == ""
    assert "_active_turn_authorization" not in session
    assert [event for event in events if event[0] == "person.admission"] == [
        (
            "person.admission", "sid",
            {"admission_id": "1" * 32, "status": "started"},
        ),
        (
            "person.admission", "sid", {
                "admission_id": "1" * 32,
                "status": "terminal",
                "reason": "run_failed",
            },
        ),
    ]


def test_old_turn_finalizer_preserves_new_turn_owned_state(monkeypatch, tmp_path):
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="f" * 32)
    finish_entered = threading.Event()
    finish_release = threading.Event()
    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=lambda prompt, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
            ],
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=old_authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "_record_turn_marker", lambda *_args, **_kwargs: "same-key")
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *_args, **_kwargs: None)
    real_finish = srv._finish_turn

    def blocking_finish(*args, **kwargs):
        finish_entered.set()
        assert finish_release.wait(timeout=5.0)
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(srv, "_finish_turn", blocking_finish)

    assert srv._run_prompt_submit(
        "r",
        "sid",
        session,
        "old prompt",
        turn_authorization=old_authorization,
        expected_turn_id="old-turn",
    )
    old_thread = session["_run_thread"]
    assert finish_entered.wait(timeout=5.0)
    with session["history_lock"]:
        session.update(
            running=True,
            _active_turn_id="new-turn",
            _active_turn_authorization=new_authorization,
            _active_turn_route="inline",
            _active_turn_marker_key="same-key",
            _hosted_room_task={"task_id": "new-task"},
            _auto_continue_scheduled=True,
        )
    finish_release.set()
    old_thread.join(timeout=5.0)

    assert not old_thread.is_alive()
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_marker_key"] == "same-key"
    assert session["_hosted_room_task"] == {"task_id": "new-task"}
    assert session["_auto_continue_scheduled"] is True


def test_inline_finalizer_keeps_admission_closed_through_settled_effects(
    monkeypatch, tmp_path
):
    authorization = TurnAuthorization.from_raw(None)
    marker_entered = threading.Event()
    marker_release = threading.Event()
    settled_turn_ids = []
    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=lambda prompt, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
            ],
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        srv, "_record_turn_marker", lambda *_args, **_kwargs: "old-marker"
    )

    retire_calls = []

    def blocking_retire(*_args, **_kwargs):
        retire_calls.append(True)
        if len(retire_calls) == 2:
            marker_entered.set()
            assert marker_release.wait(5.0)

    monkeypatch.setattr(srv, "_retire_turn_marker", blocking_retire)
    monkeypatch.setattr(
        srv,
        "_emit_settled_session_info",
        lambda _sid, current, _agent, *_args, **_kwargs: settled_turn_ids.append(
            current.get("_active_turn_id")
        ),
    )

    assert srv._run_prompt_submit(
        "rid",
        "sid",
        session,
        "prompt",
        turn_authorization=authorization,
        expected_turn_id="old-turn",
    )
    worker = session["_run_thread"]
    assert marker_entered.wait(5.0)

    assert srv._notif_claim_turn(session) is None
    assert session["_active_turn_id"] == "old-turn"
    marker_release.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    # The snapshot is published after idle (so a retiring backend sees it promptly),
    # but only for this turn: a newer generation would suppress it entirely.
    assert settled_turn_ids == [None]
    assert session["running"] is False
    assert "_active_turn_id" not in session


def test_settled_session_info_is_skipped_once_a_newer_turn_owns_the_session(tmp_path):
    session = _session(types.SimpleNamespace(session_id="a"))
    session["_active_turn_id"] = "new-turn"
    emitted = []
    rebound = srv._emit_settled_session_info

    assert rebound("sid", session, session["agent"], "old-turn") is False
    assert emitted == []


def test_inline_completion_interrupt_effects_run_before_idle(monkeypatch, tmp_path):
    authorization = TurnAuthorization.from_raw(None)
    finish_entered = threading.Event()
    finish_release = threading.Event()
    interrupt_results = []
    tts_turn_ids = []
    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=lambda prompt, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
            ],
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "_record_turn_marker", lambda *_args, **_kwargs: "marker")
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv,
        "_tts_stream_stop",
        lambda: tts_turn_ids.append(session.get("_active_turn_id")),
    )
    real_finish = srv._finish_turn

    def blocking_finish(*args, **kwargs):
        finish_entered.set()
        assert finish_release.wait(5.0)
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(srv, "_finish_turn", blocking_finish)
    assert srv._run_prompt_submit(
        "rid",
        "sid",
        session,
        "prompt",
        turn_authorization=authorization,
        expected_turn_id="old-turn",
    )
    worker = session["_run_thread"]
    assert finish_entered.wait(5.0)
    interrupter = threading.Thread(
        target=lambda: interrupt_results.append(
            srv._interrupt_session_turn("sid", session, stop_tts=True)
        )
    )
    interrupter.start()
    for _ in range(100):
        with session["history_lock"]:
            if session["_turn_completion_claim"].get("stop_tts"):
                break
        time.sleep(0.01)
    else:
        pytest.fail("interrupt did not join inline completion claim")
    finish_release.set()
    worker.join(timeout=5.0)
    interrupter.join(timeout=5.0)

    assert not worker.is_alive()
    assert not interrupter.is_alive()
    assert interrupt_results == [False]
    assert tts_turn_ids == ["old-turn"]
    assert session["running"] is False
    assert "_turn_completion_claim" not in session


def test_inline_completion_claim_settles_when_hosted_slot_release_fails(
    monkeypatch, tmp_path
):
    authorization = TurnAuthorization.from_raw(None)
    captured_claims = []
    agent = types.SimpleNamespace(
        session_id="a",
        run_conversation=lambda prompt, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
            ],
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="release-failure-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    for name in (
        "_wire_callbacks",
        "_sync_agent_model_with_config",
        "_register_session_cwd",
        "_tts_stream_begin",
        "_sync_session_key_after_compress",
    ):
        monkeypatch.setattr(srv, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "_record_turn_marker", lambda *_args, **_kwargs: "marker")
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *_args, **_kwargs: None)

    def fail_release(current):
        captured_claims.append(current["_turn_completion_claim"])
        raise RuntimeError("hosted slot release failed")

    monkeypatch.setattr(srv, "_release_hosted_room_turn_slot", fail_release)
    assert srv._run_prompt_submit(
        "rid",
        "sid",
        session,
        "prompt",
        turn_authorization=authorization,
        expected_turn_id="release-failure-turn",
    )
    with session["history_lock"]:
        worker = session.get("_run_thread")
    if worker is not None:
        worker.join(timeout=5.0)

    assert captured_claims
    assert captured_claims[0]["event"].is_set()
    assert "_turn_completion_claim" not in session


def test_queued_personal_admission_is_atomic_against_concurrent_submit(monkeypatch):
    queued = _authorization("queued-person")
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "queued work",
        "transport": None,
        "turn_authorization": queued,
    }
    route_entered = threading.Event()
    release_route = threading.Event()
    busy_snapshots = []

    def route(_session, *_args):
        if threading.current_thread().name == "drain-personal":
            route_entered.set()
            assert release_route.wait(2)
            return False
        return False

    def busy(*_args, **_kwargs):
        with session["history_lock"]:
            busy_snapshots.append((
                session.get("running"),
                session.get("_active_turn_route"),
                session.get("_active_turn_authorization"),
            ))
        return {"result": {"status": "queued"}}

    monkeypatch.setattr(srv, "_session_uses_compute_host", route)
    monkeypatch.setattr(srv, "_run_prompt_submit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(srv, "_legacy_group_fence_error", lambda *_args: None)
    monkeypatch.setattr(srv, "_handle_busy_submit", busy)
    monkeypatch.setattr(srv, "_load_dashboard_process_isolation_config", lambda: {})
    srv._sessions["sid"] = session
    try:
        drain = threading.Thread(
            target=lambda: srv._drain_queued_prompt("drain", "sid", session),
            name="drain-personal",
        )
        drain.start()
        assert route_entered.wait(1)
        submit = threading.Thread(
            target=lambda: srv._methods["prompt.submit"](
                "submit",
                {
                    "session_id": "sid",
                    "text": "same person correction",
                    "_fizko_person_access_token": "queued-person",
                    "_fizko_person_access_token_expires_at": time.time() + 3600,
                    "_fizko_person_principal_id": "a" * 64,
                    "_fizko_person_admission_id": "7" * 32,
                },
            ),
            name="concurrent-submit",
        )
        submit.start()
        time.sleep(0.05)
        assert busy_snapshots == []
        release_route.set()
        drain.join(2)
        submit.join(2)
    finally:
        release_route.set()
        srv._sessions.pop("sid", None)

    assert not drain.is_alive()
    assert not submit.is_alive()
    assert len(busy_snapshots) == 1
    running, route_name, authorization = busy_snapshots[0]
    assert running is True
    assert route_name == "inline"
    assert authorization.same_credential(queued)


def test_failed_queued_personal_dispatch_drops_retired_admission(monkeypatch):
    holder = _authorization("queued-person")
    queued = {"text": "try later", "transport": None, "turn_authorization": holder}
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = queued
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_run_prompt_submit", lambda *_args, **_kwargs: False)

    assert srv._drain_queued_prompt("drain", "sid", session) is True

    assert session["running"] is False
    assert session.get("queued_prompt") is None
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_queued_dispatch_diagnostic_failure_restores_claim_and_releases_turn(
    monkeypatch,
):
    holder = TurnAuthorization.from_raw(None)
    queued = {"text": "retry me", "transport": None, "turn_authorization": holder}
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = queued
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("dispatch failed")
        ),
    )
    monkeypatch.setattr(
        srv,
        "_notif_log_failure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("diagnostic failed")
        ),
    )

    assert srv._drain_queued_prompt("drain", "sid", session) is True

    assert session["running"] is False
    assert session["queued_prompt"] is queued
    assert "_active_turn_id" not in session
    assert "_active_turn_route" not in session


def test_generation_cancel_does_not_restore_retired_personal_queue_claim(monkeypatch):
    holder = _authorization("queued-person", admission_id="d" * 32)
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "cancel me", "transport": None, "turn_authorization": holder,
    }
    events = []

    def bump_generation(claimed_session):
        claimed_session["_queued_prompt_generation"] = 1
        return False

    monkeypatch.setattr(srv, "_session_uses_compute_host", bump_generation)
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))

    assert srv._drain_queued_prompt("drain", "sid", session) is True

    assert session.get("queued_prompt") is None
    assert session["running"] is False
    assert events == [(
        "person.admission", "sid", {
            "admission_id": "d" * 32,
            "status": "terminal",
            "reason": "cancelled_before_dispatch",
        },
    )]


def test_generation_bump_after_claim_check_cannot_drop_personal_dispatch(monkeypatch):
    holder = _authorization("queued-person", admission_id="e" * 32)
    session = _session(types.SimpleNamespace())
    session["queued_prompt"] = {
        "text": "dispatch me", "transport": None, "turn_authorization": holder,
    }
    underlying_lock = threading.RLock()

    class BumpAfterSecondExit:
        exits = 0

        def __enter__(self):
            underlying_lock.acquire()
            return self

        def __exit__(self, *_args):
            underlying_lock.release()
            self.exits += 1
            if self.exits == 2:
                session["_queued_prompt_generation"] = 1

    session["history_lock"] = BumpAfterSecondExit()
    dispatched = []
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: dispatched.append("dispatch") or True,
    )

    assert srv._drain_queued_prompt("drain", "sid", session) is True

    assert dispatched == ["dispatch"]


def test_failed_run_prompt_admission_clears_authorization_and_route(monkeypatch):
    holder = _authorization("refused-person")
    session = _session(types.SimpleNamespace())
    session.update(running=True, _active_turn_authorization=holder, _active_turn_route="inline")
    events = []
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: RuntimeError("owned elsewhere"))
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))

    assert srv._run_prompt_submit(
        "r", "sid", session, "hello", turn_authorization=holder
    ) is False

    assert session["running"] is False
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session
    assert events[-1:] == [
        (
            "person.admission",
            "sid",
            {
                "admission_id": "1" * 32,
                "status": "terminal",
                "reason": "admission_rejected",
            },
        )
    ]


def test_interrupt_clears_active_authorization_and_uses_forced_inline_route(monkeypatch):
    holder = _authorization("active-person")
    interrupted = []
    compute_interrupts = []
    agent = types.SimpleNamespace(interrupt=lambda: interrupted.append(True))
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_authorization=holder,
        _active_turn_route="inline",
        _run_thread=None,
    )
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: True)
    monkeypatch.setattr(
        srv,
        "_get_compute_host_supervisor",
        lambda: types.SimpleNamespace(interrupt=lambda *_a, **_k: compute_interrupts.append(True)),
    )
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    assert srv._interrupt_session_turn("sid", session) is False

    assert interrupted == [True]
    assert compute_interrupts == []
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


@pytest.mark.parametrize("queued_count", [1, 3])
def test_interrupt_terminally_accounts_each_queued_personal_admission_once(
    monkeypatch, queued_count,
):
    active = _authorization("active-person", admission_id="a" * 32)
    queued = [
        _authorization(f"queued-{index}", admission_id=f"{index + 1:x}" * 32)
        for index in range(queued_count)
    ]
    session = _session(types.SimpleNamespace(interrupt=lambda: None))
    session.update(
        running=True,
        _active_turn_id="active-person-turn",
        _active_turn_authorization=active,
        _active_turn_route="inline",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
        _run_thread_turn_id="active-person-turn",
        queued_prompt={
            "text": "queued-0", "transport": None,
            "turn_authorization": queued[0],
        },
        queued_prompts=[
            {
                "text": f"queued-{index}", "transport": None,
                "turn_authorization": authorization,
            }
            for index, authorization in enumerate(queued[1:], start=1)
        ],
    )
    events = []
    monkeypatch.setattr(srv, "_emit", lambda *args, **kwargs: events.append(args))
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    assert srv._interrupt_session_turn("sid", session) is False
    assert srv._interrupt_session_turn("sid", session) is False

    terminal = [event for event in events if event[0] == "person.admission"]
    assert terminal == [
        (
            "person.admission", "sid", {
                "admission_id": authorization._fizko_admission_id(),
                "status": "terminal",
                "reason": "interrupted_while_queued",
            },
        )
        for authorization in queued
    ]
    assert session.get("queued_prompt") is None
    assert not session.get("queued_prompts")
    assert session["_active_turn_authorization"] is active
    assert session["running"] is True
    assert all(event[2]["admission_id"] != "a" * 32 for event in terminal)


def test_interrupt_terminally_accounts_abandoned_active_admission(monkeypatch):
    active = _authorization("active-person", admission_id="a" * 32)
    session = _session(types.SimpleNamespace(interrupt=lambda: None))
    session.update(
        running=False,
        _active_turn_authorization=active,
        _active_turn_route="inline",
    )
    events = []
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args) or True)
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    assert srv._interrupt_session_turn("sid", session) is False

    assert events == [(
        "person.admission", "sid", {
            "admission_id": "a" * 32,
            "status": "terminal",
            "reason": "interrupted",
        },
    )]
    assert "_active_turn_authorization" not in session


def test_reset_terminally_accounts_active_and_queued_person_admissions(monkeypatch):
    active = _authorization("active-person", admission_id="a" * 32)
    queued = _authorization("queued-person", admission_id="b" * 32)
    new_agent = types.SimpleNamespace()
    session = _session(types.SimpleNamespace())
    session.update(
        _active_turn_authorization=active,
        _active_turn_route="inline",
        queued_prompt={
            "text": "queued", "transport": None, "turn_authorization": queued,
        },
    )
    events = []
    monkeypatch.setattr(srv, "_set_session_context", lambda *_args: [])
    monkeypatch.setattr(srv, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(srv, "_rebuild_session_agent", lambda *_args, **_kwargs: new_agent)
    monkeypatch.setattr(srv, "_session_source", lambda *_args: "tui")
    monkeypatch.setattr(srv, "_context_cwd_is_launch_artifact", lambda *_args: False)
    monkeypatch.setattr(srv, "_session_info", lambda *_args: {})
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(srv, "_restart_slash_worker", lambda *_args: None)

    srv._reset_session_agent("sid", session)

    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session
    assert session.get("queued_prompt") is None
    assert events[:2] == [
        (
            "person.admission", "sid", {
                "admission_id": "a" * 32,
                "status": "terminal",
                "reason": "session_reset",
            },
        ),
        (
            "person.admission", "sid", {
                "admission_id": "b" * 32,
                "status": "terminal",
                "reason": "session_reset",
            },
        ),
    ]


def test_finalize_terminally_accounts_active_and_queued_person_admissions(monkeypatch):
    active = _authorization("active-person", admission_id="c" * 32)
    queued = _authorization("queued-person", admission_id="d" * 32)
    session = _session(types.SimpleNamespace(session_id=""))
    session.update(
        _sid="sid",
        _active_turn_authorization=active,
        _active_turn_route="inline",
        queued_prompt={
            "text": "queued", "transport": None, "turn_authorization": queued,
        },
    )
    events = []
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(srv, "_notify_session_boundary", lambda *_args: None)
    monkeypatch.setattr(srv, "_release_active_session_slot", lambda *_args: True)

    srv._finalize_session(session)
    srv._finalize_session(session)

    assert events == [
        (
            "person.admission", "sid", {
                "admission_id": "c" * 32,
                "status": "terminal",
                "reason": "session_finalized",
            },
        ),
        (
            "person.admission", "sid", {
                "admission_id": "d" * 32,
                "status": "terminal",
                "reason": "session_finalized",
            },
        ),
    ]
    assert "_active_turn_authorization" not in session
    assert session.get("queued_prompt") is None


def test_reset_does_not_drop_personal_admission_queued_during_rebuild(monkeypatch):
    before_reset = _authorization("before-reset", admission_id="a" * 32)
    after_boundary = _authorization("after-boundary", admission_id="b" * 32)
    session = _session(types.SimpleNamespace())
    session.update(
        running=True,
        queued_prompt={
            "text": "old", "transport": None, "turn_authorization": before_reset,
        },
    )
    events = []

    def rebuild(*_args, **_kwargs):
        session["queued_prompt"] = {
            "text": "new", "transport": None, "turn_authorization": after_boundary,
        }
        return types.SimpleNamespace()

    monkeypatch.setattr(srv, "_set_session_context", lambda *_args: [])
    monkeypatch.setattr(srv, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(srv, "_rebuild_session_agent", rebuild)
    monkeypatch.setattr(srv, "_session_source", lambda *_args: "tui")
    monkeypatch.setattr(srv, "_context_cwd_is_launch_artifact", lambda *_args: False)
    monkeypatch.setattr(srv, "_session_info", lambda *_args: {})
    monkeypatch.setattr(srv, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(srv, "_restart_slash_worker", lambda *_args: None)

    srv._reset_session_agent("sid", session)

    assert session["queued_prompt"]["turn_authorization"] is after_boundary
    terminal_ids = {
        payload["admission_id"]
        for event, _sid, payload in events
        if event == "person.admission"
    }
    assert terminal_ids == {"a" * 32}


def _install_token_bearing_direct_rpc_session(monkeypatch):
    calls = {"steer": [], "redirect": [], "interrupt": []}
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        steer=lambda text: calls["steer"].append(text) or True,
        redirect=lambda text: calls["redirect"].append(text) or True,
        interrupt=lambda: calls["interrupt"].append(True),
    )
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_authorization=_authorization("active-person"),
        _active_turn_route="inline",
        _run_thread=None,
    )
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)
    srv._sessions["sid"] = session
    return session, calls


def test_direct_session_steer_fails_closed_for_personal_turn(monkeypatch):
    _session_record, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    try:
        response = srv._methods["session.steer"]("r", {"session_id": "sid", "text": "change"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["steer"] == []


def test_direct_session_redirect_fails_closed_for_personal_turn(monkeypatch):
    _session_record, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    try:
        response = srv._methods["session.redirect"]("r", {"session_id": "sid", "text": "change"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["redirect"] == []


def test_direct_session_interrupt_fails_closed_for_personal_turn(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    tts_stops = []
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: tts_stops.append(True))
    try:
        response = srv._methods["session.interrupt"]("r", {"session_id": "sid"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert tts_stops == []
    assert session["running"] is True
    assert "_active_turn_authorization" in session


def test_rejected_personal_interrupt_does_not_start_pending_agent_build(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["agent"] = None
    session["agent_ready"] = threading.Event()
    starts = []
    monkeypatch.setattr(
        srv, "_start_agent_build", lambda *args, **kwargs: starts.append((args, kwargs))
    )
    try:
        response = srv._methods["session.interrupt"]("r", {"session_id": "sid"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert starts == []
    assert calls["interrupt"] == []
    assert "agent_build_started" not in session
    assert "_agent_build_thread" not in session
    assert session["running"] is True


def test_direct_interrupt_never_enters_agent_build_resolver(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["agent"] = None
    session["agent_ready"] = threading.Event()
    session["_active_turn_authorization"] = TurnAuthorization.from_raw(None)
    resolved = []

    def forbidden_resolver(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("session.interrupt must not warm an agent build")

    monkeypatch.setattr(srv, "_sess", forbidden_resolver)
    try:
        response = srv._methods["session.interrupt"]("r", {"session_id": "sid"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"]["status"] == "interrupted"
    assert resolved == []
    assert calls["interrupt"] == []


def test_interrupt_claims_tts_and_marker_before_new_personal_generation(monkeypatch):
    session, _calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["_active_turn_authorization"] = TurnAuthorization.from_raw(None)
    session["_active_turn_marker_key"] = "old-turn-key"
    tts_stops = []
    retired = []
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: tts_stops.append(True))
    monkeypatch.setattr(
        srv,
        "_retire_turn_marker",
        lambda _session, *keys, **_kwargs: retired.extend(keys),
    )
    original_interrupt = srv._interrupt_session_turn

    def interrupt_then_admit_personal(*args, **kwargs):
        result = original_interrupt(*args, **kwargs)
        with session["history_lock"]:
            session["running"] = True
            session["_active_turn_authorization"] = _authorization("new-person")
            session["_active_turn_marker_key"] = "new-person-key"
        return result

    monkeypatch.setattr(srv, "_interrupt_session_turn", interrupt_then_admit_personal)
    try:
        response = srv._methods["session.interrupt"]("r", {"session_id": "sid"})
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"]["status"] == "interrupted"
    assert tts_stops == [True]
    assert retired == ["old-turn-key", "agent-session-key"]
    assert session["_active_turn_marker_key"] == "new-person-key"
    assert session["_active_turn_authorization"].is_personal


def test_expected_hosted_task_is_checked_at_interrupt_claim(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["_active_turn_authorization"] = TurnAuthorization.from_raw(None)
    session["_hosted_room_task"] = {"task_id": "expected-old"}
    original_interrupt = srv._interrupt_session_turn

    def replace_task_before_claim(*args, **kwargs):
        with session["history_lock"]:
            session["_hosted_room_task"] = {"task_id": "new-task"}
        return original_interrupt(*args, **kwargs)

    monkeypatch.setattr(srv, "_interrupt_session_turn", replace_task_before_claim)
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {"session_id": "sid", "expected_hosted_task_id": "expected-old"},
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"] == {
        "status": "not_interrupted",
        "interrupted": False,
    }
    assert calls["interrupt"] == []
    assert session["running"] is True


def test_compute_submit_aborts_if_interrupt_won_after_admission(monkeypatch):
    submitted = []
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="admitted-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="compute",
        _turn_cancel_requested=True,
    )
    monkeypatch.setattr(
        srv, "_load_dashboard_process_isolation_config", lambda: {}
    )
    monkeypatch.setattr(
        srv,
        "_get_compute_host_supervisor",
        lambda _cfg=None: types.SimpleNamespace(
            submit_turn=lambda *args, **kwargs: submitted.append((args, kwargs))
        ),
    )

    response = srv._submit_prompt_to_compute_host(
        "rid", "sid", session, "must not dispatch"
    )

    assert response["result"]["status"] == "streaming"
    assert submitted == []
    assert session["running"] is False
    assert "_active_turn_id" not in session


def test_compute_submit_allows_synchronous_success_callback(monkeypatch):
    completed = []
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="sync-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="compute",
    )

    class Supervisor:
        def submit_turn(self, _frame, *, on_complete=None):
            assert on_complete is not None
            on_complete({"type": "turn.end", "reason": "complete"})

    monkeypatch.setattr(srv, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(srv, "_get_compute_host_supervisor", lambda _cfg=None: Supervisor())
    monkeypatch.setattr(
        srv,
        "_on_compute_host_turn_done",
        lambda *_args, **_kwargs: completed.append(True),
    )
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            srv._submit_prompt_to_compute_host("rid", "sid", session, "prompt")
        )
    )
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert completed == [True]
    assert result[0]["result"]["status"] == "streaming"


def test_compute_completion_keeps_admission_closed_through_terminal_effects(monkeypatch):
    authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _compute_host_turn_id="old-turn",
        _active_turn_id="old-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="compute",
    )
    admission_attempts = []

    def emit(event, *_args):
        if event == "message.complete":
            admission_attempts.append(srv._notif_claim_turn(session))

    monkeypatch.setattr(srv, "_emit", emit)
    monkeypatch.setattr(
        srv, "_apply_compute_host_metadata_mirror", lambda *_args: None
    )
    monkeypatch.setattr(srv, "_compute_host_session_info", lambda *_args: {})
    monkeypatch.setattr(srv, "_drain_queued_prompt", lambda *_args: False)

    srv._on_compute_host_turn_done(
        "rid", "sid", session,
        {"type": "turn.error", "message": "old failure"},
        expected_turn_id="old-turn",
    )

    assert admission_attempts == [None]
    assert session["running"] is False
    assert "_turn_completion_claim" not in session
    assert srv._notif_claim_turn(session)


def test_compute_completion_applies_interrupt_tts_before_publishing_idle(monkeypatch):
    authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _compute_host_turn_id="old-turn",
        _active_turn_id="old-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="compute",
    )
    real_threading = srv.threading
    completion_started = threading.Event()
    allow_completion = threading.Event()
    effect_turn_ids = []

    class AdmissionEvent:
        def __init__(self):
            self._event = threading.Event()

        def wait(self, timeout=None):
            return self._event.wait(timeout)

        def set(self):
            with session["history_lock"]:
                session.update(
                    running=True,
                    _active_turn_id="new-turn",
                    _active_turn_authorization=TurnAuthorization.from_raw(None),
                    _active_turn_route="inline",
                )
            self._event.set()

    monkeypatch.setattr(
        srv,
        "threading",
        types.SimpleNamespace(
            Event=AdmissionEvent,
            get_ident=real_threading.get_ident,
        ),
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)

    def block_metadata(*_args):
        completion_started.set()
        assert allow_completion.wait(2.0)

    monkeypatch.setattr(srv, "_apply_compute_host_metadata_mirror", block_metadata)
    monkeypatch.setattr(srv, "_compute_host_session_info", lambda *_args: {})
    monkeypatch.setattr(srv, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(
        srv,
        "_tts_stream_stop",
        lambda: effect_turn_ids.append(session.get("_active_turn_id")),
    )

    owner = threading.Thread(
        target=lambda: srv._on_compute_host_turn_done(
            "rid",
            "sid",
            session,
            {"type": "turn.end"},
            expected_turn_id="old-turn",
        )
    )
    results = []
    owner.start()
    assert completion_started.wait(2.0)
    waiter = threading.Thread(
        target=lambda: results.append(
            srv._interrupt_session_turn("sid", session, stop_tts=True)
        )
    )
    waiter.start()
    deadline = time.monotonic() + 2.0
    while not session["_turn_completion_claim"].get("stop_tts"):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    allow_completion.set()
    owner.join(timeout=2.0)
    waiter.join(timeout=2.0)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert results == [True]
    assert effect_turn_ids == ["old-turn"]
    assert session["_active_turn_id"] == "new-turn"


def test_completion_waiter_cannot_retarget_new_turn(monkeypatch):
    event = threading.Event()
    completion_claim = {
        "turn_id": "old-turn",
        "event": event,
        "owner_thread_id": -1,
    }
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="9" * 32)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=old_authorization,
        _active_turn_route="compute",
        _turn_completion_claim=completion_claim,
        queued_prompt={"text": "next", "transport": None},
        session_key="shared-key",
    )
    effects = []
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: effects.append("tts"))
    monkeypatch.setattr(
        srv, "_retire_turn_marker",
        lambda *_args, **_kwargs: effects.append("marker"),
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            srv._interrupt_session_turn(
                "sid", session, stop_tts=True, retire_turn_marker=True
            )
        )
    )
    worker.start()
    deadline = time.monotonic() + 2.0
    while not completion_claim.get("stop_tts") and time.monotonic() < deadline:
        time.sleep(0.01)
    with session["history_lock"]:
        session.pop("_turn_completion_claim", None)
        session.update(
            running=True,
            _active_turn_id="new-turn",
            _active_turn_authorization=new_authorization,
            _active_turn_route="inline",
        )
    event.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert results == [True]
    assert effects == []
    assert completion_claim["stop_tts"] is True
    assert completion_claim["retire_turn_marker"] is True
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization
    assert session["queued_prompt"]["text"] == "next"


def test_completion_interrupt_retires_marker_with_supported_arguments(monkeypatch):
    session = _session(None)
    session.update(
        running=True,
        session_key="old-key",
        _compute_host_turn_id="old-turn",
        _active_turn_id="old-turn",
        _active_turn_route="compute",
        _active_turn_marker_key="old-marker-key",
    )
    retired = []
    interrupt_results = []

    def retire(_session, *keys, include_current=True, expected_turn_id=None):
        retired.append((keys, include_current, expected_turn_id))

    def interrupt_during_completion(*_args):
        interrupt_results.append(
            srv._interrupt_session_turn(
                "sid",
                session,
                retire_turn_marker=True,
                expected_turn_id="old-turn",
            )
        )

    monkeypatch.setattr(srv, "_retire_turn_marker", retire)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv, "_apply_compute_host_metadata_mirror", interrupt_during_completion
    )
    monkeypatch.setattr(srv, "_compute_host_session_info", lambda *_args: {})
    monkeypatch.setattr(srv, "_drain_queued_prompt", lambda *_args: False)

    srv._on_compute_host_turn_done(
        "rid",
        "sid",
        session,
        {"type": "turn.end"},
        expected_turn_id="old-turn",
    )

    assert interrupt_results == [True]
    assert retired == [
        (("old-marker-key", "old-key"), False, "old-turn")
    ]


def test_compute_completion_exception_still_drains_queued_work(monkeypatch):
    authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _compute_host_turn_id="old-turn",
        _active_turn_id="old-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="compute",
        queued_prompt={"text": "next", "transport": None},
    )
    drained = []

    monkeypatch.setattr(
        srv, "_emit", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("terminal emit failed")
        )
    )
    monkeypatch.setattr(
        srv, "_drain_queued_prompt", lambda *_args: drained.append(True)
    )

    with pytest.raises(RuntimeError, match="terminal emit failed"):
        srv._on_compute_host_turn_done(
            "rid", "sid", session,
            {"type": "turn.error", "message": "failed"},
            expected_turn_id="old-turn",
        )

    assert drained == [True]
    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_turn_completion_claim" not in session


def test_queue_rejection_emit_failure_restores_valid_claim(monkeypatch):
    expired = _authorization("expired", expires_at=time.time() - 1)
    ordinary = TurnAuthorization.from_raw(None)
    session = _session(types.SimpleNamespace())
    session.update(
        queued_prompt={
            "text": "expired",
            "transport": None,
            "turn_authorization": expired,
        },
        queued_prompts=[{
            "text": "valid",
            "transport": None,
            "turn_authorization": ordinary,
        }],
    )
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(srv, "_emit_person_admission", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv, "_emit", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("error emit failed")
        )
    )
    monkeypatch.setattr(srv, "_notif_log_failure", lambda *_args: None)

    assert srv._drain_queued_prompt("rid", "sid", session)

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert session["queued_prompt"]["text"] == "valid"
    assert session["queued_prompt"]["turn_authorization"] is ordinary


def test_stale_queued_compute_failure_preserves_new_turn(monkeypatch):
    queued_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="7" * 32)
    session = _session(None)
    session.update(
        queued_prompt={
            "text": "queued",
            "transport": None,
            "turn_authorization": queued_authorization,
        }
    )
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_args: True)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)

    def failed_old_dispatch(*_args, **_kwargs):
        with session["history_lock"]:
            session.update(
                running=True,
                _active_turn_id="new-turn",
                _active_turn_authorization=new_authorization,
                _active_turn_route="inline",
            )
            srv._start_inflight_turn(session, "new prompt")
        return {"error": {"message": "old dispatch failed"}}

    monkeypatch.setattr(srv, "_submit_prompt_to_compute_host", failed_old_dispatch)

    assert srv._drain_queued_prompt("rid", "sid", session)

    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization
    assert session["inflight_turn"]["user"] == "new prompt"


def test_prompt_admission_revalidates_after_blocking_owner_claim(monkeypatch):
    cleared = []
    agent = types.SimpleNamespace(clear_interrupt=lambda: cleared.append(True))
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="e" * 32)
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=old_authorization,
        _active_turn_route="inline",
        attached_images=["old.png"],
    )

    def ownership_claim(_sid, _session):
        with session["history_lock"]:
            session.update(
                running=True,
                _active_turn_id="new-turn",
                _active_turn_authorization=new_authorization,
                _active_turn_route="inline",
                attached_images=["new-turn.png"],
                inflight_turn={"user": "new prompt"},
            )
        return None

    monkeypatch.setattr(srv, "_ensure_active_session_slot", ownership_claim)

    admitted = srv._admit_prompt_turn(
        "sid", session, "old prompt", None, None, None, None,
        old_authorization, expected_turn_id="old-turn",
    )

    assert admitted is None
    assert cleared == []
    assert session["attached_images"] == ["new-turn.png"]
    assert session["inflight_turn"] == {"user": "new prompt"}
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization


def test_missing_agent_admission_releases_running_for_its_own_turn(monkeypatch):
    """#111531: a turn refused for a missing agent must not leave the session busy."""
    authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        agent_error="build abandoned",
        _active_turn_id="lonely-turn",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    frames = []
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(
        srv,
        "_emit",
        lambda event, _sid, payload=None: frames.append((event, payload)),
    )

    assert srv._admit_prompt_turn(
        "sid", session, "hola", None, None, None, None, authorization,
        expected_turn_id="lonely-turn",
    ) is None

    assert [e for e, _p in frames] == ["message.complete"]
    assert frames[0][1]["status"] == "error"
    assert frames[0][1]["recoverable"] is True
    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_missing_agent_terminal_error_cannot_fail_newer_inflight(monkeypatch):
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="8" * 32)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=old_authorization,
        _active_turn_route="inline",
    )
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *_args: None)
    real_terminal_error = srv._emit_terminal_turn_error
    running_at_terminal = []

    def replace_before_terminal(*args, **kwargs):
        running_at_terminal.append(session["running"])
        with session["history_lock"]:
            session.update(
                running=True,
                _active_turn_id="new-turn",
                _active_turn_authorization=new_authorization,
                _active_turn_route="inline",
            )
            srv._start_inflight_turn(session, "new prompt")
        return real_terminal_error(*args, **kwargs)

    monkeypatch.setattr(srv, "_emit_terminal_turn_error", replace_before_terminal)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)

    assert srv._admit_prompt_turn(
        "sid",
        session,
        "old prompt",
        None,
        None,
        None,
        None,
        old_authorization,
        expected_turn_id="old-turn",
    ) is None

    assert running_at_terminal == [True]
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization
    assert session["inflight_turn"]["user"] == "new prompt"
    assert session["inflight_turn"].get("status") != "error"


def test_stale_compute_submit_cannot_relabel_payload_as_new_turn(monkeypatch):
    sent = []
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="new-turn",
        _active_turn_authorization=new_authorization,
        _active_turn_route="compute",
        _turn_cancel_requested=False,
    )
    monkeypatch.setattr(srv, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(
        srv,
        "_get_compute_host_supervisor",
        lambda _cfg=None: types.SimpleNamespace(
            submit_turn=lambda frame, **_kwargs: sent.append(frame)
        ),
    )

    response = srv._submit_prompt_to_compute_host(
        "rid", "sid", session, "OLD QUEUED TEXT",
        expected_turn_id="old-turn",
        expected_turn_authorization=old_authorization,
    )

    assert response["result"]["status"] == "streaming"
    assert sent == []
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization


def test_stale_compute_completion_cannot_clear_new_turn(monkeypatch):
    new_authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _compute_host_turn_id="new-turn",
        _active_turn_id="new-turn",
        _active_turn_authorization=new_authorization,
        _active_turn_route="compute",
    )
    emitted = []
    monkeypatch.setattr(srv, "_emit", lambda *args: emitted.append(args))

    srv._on_compute_host_turn_done(
        "rid", "sid", session, {"type": "turn.end"},
        expected_turn_id="old-turn",
    )

    assert emitted == []
    assert session["running"] is True
    assert session["_compute_host_turn_id"] == "new-turn"
    assert session["_active_turn_id"] == "new-turn"


def test_stale_persist_failure_cannot_clear_new_person_turn(monkeypatch):
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="d" * 32)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="new-turn",
        _active_turn_authorization=new_authorization,
        _active_turn_route="inline",
        _hosted_room_task={"id": "new-task"},
        inflight_turn={"user": "new prompt"},
    )
    monkeypatch.setattr(
        srv, "_ensure_session_db_row",
        lambda _session: (_ for _ in ()).throw(OSError("storage failed")),
    )

    error = srv._persist_session_row_for_submit(
        "rid", session, expected_turn_id="old-turn",
        expected_authorization=old_authorization,
    )

    assert error["error"]["code"] == 5071
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization
    assert session["_hosted_room_task"] == {"id": "new-task"}
    assert session["inflight_turn"] == {"user": "new prompt"}


def test_stale_notification_release_cannot_clear_new_turn():
    new_authorization = TurnAuthorization.from_raw(None)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="new-turn",
        _active_turn_authorization=new_authorization,
        _active_turn_route="inline",
    )

    srv._notif_release_turn(session, "old-turn")

    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization


def test_notification_delivery_claim_loss_releases_exact_turn(monkeypatch):
    from tools import async_delegation

    session = _session(None)
    turn_id = srv._notif_claim_turn(session)
    monkeypatch.setattr(
        async_delegation, "claim_event_delivery", lambda *_args: None
    )

    srv._notif_dispatch_event(
        "sid", session, {"type": "async_delegation"}, "done", turn_id
    )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_notification_delivery_claim_exception_releases_exact_turn(monkeypatch):
    from tools import async_delegation

    session = _session(None)
    turn_id = srv._notif_claim_turn(session)
    monkeypatch.setattr(
        async_delegation,
        "claim_event_delivery",
        lambda *_args: (_ for _ in ()).throw(OSError("claim unavailable")),
    )

    srv._notif_dispatch_event(
        "sid", session, {"type": "async_delegation"}, "done", turn_id
    )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_slash_loop_send_exception_releases_new_identity(monkeypatch):
    session = _session(None)
    original_turn_id = srv._notif_claim_turn(session)
    mgr = types.SimpleNamespace()
    monkeypatch.setitem(
        srv._methods,
        "command.dispatch",
        lambda *_args: {"result": {"type": "send", "message": "follow-up"}},
    )
    monkeypatch.setattr(
        srv, "_emit", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("message.start failed")
        )
    )

    assert not srv._notif_slash_loop_tick(
        "rid", "sid", session, mgr, "/status", original_turn_id
    )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_notification_ack_failure_keeps_dispatched_turn_owned(monkeypatch):
    from tools import async_delegation

    session = _session(None)
    turn_id = srv._notif_claim_turn(session)
    released = []
    claim = object()
    monkeypatch.setattr(
        async_delegation, "claim_event_delivery", lambda *_args: claim
    )
    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        lambda *_args: (_ for _ in ()).throw(OSError("ack unavailable")),
    )
    monkeypatch.setattr(
        async_delegation,
        "release_event_delivery",
        lambda *_args: released.append(True),
    )
    monkeypatch.setattr(srv, "_notif_submit", lambda *_args, **_kwargs: None)

    srv._notif_dispatch_event(
        "sid", session, {"type": "async_delegation"}, "done", turn_id
    )

    assert released == []
    assert session["running"] is True
    assert session["_active_turn_id"] == turn_id


def test_completion_batch_ack_failure_keeps_dispatched_turn_owned(monkeypatch):
    from tools import async_delegation
    from tools import process_registry_notifications as notifications_module

    session = _session(None)
    released = []
    claim = object()

    class Batch:
        def __init__(self, _items):
            pass

        def render(self, _registry):
            return "batch"

        def display_text(self, _registry):
            return "batch display"

    monkeypatch.setattr(notifications_module, "ProcessNotificationBatch", Batch)
    monkeypatch.setattr(
        async_delegation, "claim_event_delivery", lambda *_args: claim
    )
    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        lambda *_args: (_ for _ in ()).throw(OSError("ack unavailable")),
    )
    monkeypatch.setattr(
        async_delegation,
        "release_event_delivery",
        lambda *_args: released.append(True),
    )
    monkeypatch.setattr(srv, "_notif_submit", lambda *_args, **_kwargs: None)

    srv._notif_dispatch_completions(
        "sid",
        session,
        [({"type": "completion"}, "done")],
        types.SimpleNamespace(),
        None,
    )

    assert released == []
    assert session["running"] is True
    assert session.get("_active_turn_id")


def test_notification_logging_failure_still_releases_failed_dispatch(monkeypatch):
    session = _session(None)
    turn_id = srv._notif_claim_turn(session)
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("dispatch failed")
        ),
    )
    monkeypatch.setattr(
        srv,
        "_notif_log_failure",
        lambda *_args: (_ for _ in ()).throw(OSError("diagnostic failed")),
    )

    with pytest.raises(RuntimeError, match="dispatch failed"):
        srv._notif_submit(
            "rid",
            "sid",
            session,
            "notification",
            "notification failed",
            expected_turn_id=turn_id,
        )

    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_route" not in session


@pytest.mark.parametrize("kind", ["heartbeat", "loop"])
def test_autonomous_tick_storage_failure_releases_claim(monkeypatch, kind):
    session = _session(None)
    session["session_key"] = "tick-session"
    abandoned = []

    if kind == "heartbeat":
        import hermes_cli.heartbeat as module

        class State:
            fire_count = 1

            @staticmethod
            def is_due():
                return True

        class Manager:
            state = State()

            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def is_active():
                return True

            @staticmethod
            def due_prompt():
                raise OSError("heartbeat state unavailable")

            @staticmethod
            def abandon_fire():
                abandoned.append("heartbeat")

        monkeypatch.setattr(module, "HeartbeatManager", Manager)
        monkeypatch.setattr(
            srv, "_notif_gateway_owns_heartbeat", lambda *_args: False
        )
        srv._maybe_fire_tui_heartbeat_tick("sid", session)
    else:
        import hermes_cli.loops as module

        class State:
            route = None

        class Manager:
            state = State()

            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def is_due():
                return True

            @staticmethod
            def fire_tick():
                raise OSError("loop state unavailable")

            @staticmethod
            def abandon_tick():
                abandoned.append("loop")

        monkeypatch.setattr(module, "LoopManager", Manager)
        monkeypatch.setattr(module, "goal_blocks_loop_tick", lambda *_args: False)
        srv._maybe_fire_tui_loop_tick("sid", session)

    assert abandoned == [kind]
    assert session["running"] is False
    assert "_active_turn_id" not in session
    assert "_active_turn_authorization" not in session
    assert "_active_turn_route" not in session


def test_delayed_ready_worker_cannot_dispatch_inside_new_turn(monkeypatch):
    dispatched = []
    old_authorization = TurnAuthorization.from_raw(None)
    new_authorization = _authorization("new-person", admission_id="d" * 32)
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="new-turn",
        _active_turn_authorization=new_authorization,
        _active_turn_route="inline",
        _turn_cancel_requested=False,
    )
    monkeypatch.setattr(srv, "_wait_agent_for_prompt", lambda *_args: None)
    monkeypatch.setattr(
        srv,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: dispatched.append(True) or True,
    )
    monkeypatch.setattr(srv, "_emit", lambda *_args, **_kwargs: None)

    srv._run_after_agent_ready(
        "rid",
        "sid",
        session,
        "old prompt",
        None,
        None,
        None,
        old_authorization,
        "old-turn",
    )

    assert dispatched == []
    assert session["running"] is True
    assert session["_active_turn_id"] == "new-turn"
    assert session["_active_turn_authorization"] is new_authorization


def test_busy_submit_waits_for_interrupt_claim_before_queueing(monkeypatch):
    claim_event = threading.Event()
    result = []
    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _turn_interrupt_claim={"event": claim_event},
    )
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "queue")
    worker = threading.Thread(
        target=lambda: result.append(
            srv._handle_busy_submit("rid", "sid", session, "new prompt", None)
        )
    )
    worker.start()
    worker.join(timeout=0.1)

    assert worker.is_alive()
    assert session.get("queued_prompt") is None
    with session["history_lock"]:
        session.pop("_turn_interrupt_claim", None)
        session["running"] = False
    claim_event.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert result == [None]
    assert session.get("queued_prompt") is None


def test_interrupt_claim_blocks_concurrent_personal_admission(monkeypatch):
    entered_interrupt = threading.Event()
    release_interrupt = threading.Event()

    def blocking_interrupt():
        entered_interrupt.set()
        assert release_interrupt.wait(timeout=5.0)

    session = _session(types.SimpleNamespace(interrupt=blocking_interrupt))
    session.update(
        running=True,
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)
    worker = threading.Thread(
        target=srv._interrupt_session_turn,
        args=("sid", session),
        kwargs={"reject_person_authorized": True},
    )
    worker.start()
    assert entered_interrupt.wait(timeout=5.0)

    with session["history_lock"]:
        session["running"] = False
        session.pop("_active_turn_authorization", None)
    err, _fields = srv._lock_in_submit_turn(
        "r",
        "sid",
        session,
        "new personal turn",
        {},
        False,
        None,
        None,
        None,
        _authorization("new-person"),
        False,
    )

    assert err is srv._SUBMIT_TURN_INTERRUPT_PENDING
    assert session["running"] is False
    assert "_active_turn_authorization" not in session

    release_interrupt.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert "_turn_interrupt_claim" not in session


def test_notification_turn_claim_mints_immutable_identity():
    session = _session(types.SimpleNamespace())

    assert srv._notif_claim_turn(session)

    turn_id = session.get("_active_turn_id")
    assert isinstance(turn_id, str)
    assert len(turn_id) == 32


def test_agent_interrupt_identity_is_bound_for_exact_gateway_turn(monkeypatch):
    observed = []

    class Agent:
        def bind_gateway_turn(self, turn_id):
            self.active_turn_id = turn_id

        def clear_gateway_turn(self, turn_id):
            if self.active_turn_id == turn_id:
                self.active_turn_id = None

        def run_conversation(self, *_args, **_kwargs):
            observed.append(self.active_turn_id)
            return {"final_response": "done", "messages": []}

    agent = Agent()
    session = _session(agent)
    session["_active_turn_id"] = "gateway-turn"
    monkeypatch.setattr(srv, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(
        srv,
        "_start_usage_ticker",
        lambda *_args: (
            types.SimpleNamespace(set=lambda: None),
            types.SimpleNamespace(join=lambda: None),
        ),
    )
    state = srv._TurnRun(
        agent=agent,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=True,
    )

    srv._invoke_agent(
        "sid", session, state, "hello", "hello", None, [], None, None
    )

    assert observed == ["gateway-turn"]
    assert agent.active_turn_id is None


def test_tts_stop_failure_does_not_drop_queued_person_admission(monkeypatch):
    queued = _authorization("queued-person", admission_id="b" * 32)
    session = _session(types.SimpleNamespace(interrupt=lambda: None))
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
        queued_prompt={
            "text": "queued",
            "transport": None,
            "turn_authorization": queued,
        },
    )
    events = []
    monkeypatch.setattr(
        srv, "_tts_stream_stop", lambda: (_ for _ in ()).throw(RuntimeError("tts failed"))
    )
    monkeypatch.setattr(srv, "_emit", lambda *args, **_kwargs: events.append(args) or True)

    assert srv._interrupt_session_turn("sid", session, stop_tts=True) is False

    assert "_turn_interrupt_claim" not in session
    assert session.get("queued_prompt") is None
    assert (
        "person.admission",
        "sid",
        {
            "admission_id": "b" * 32,
            "status": "terminal",
            "reason": "interrupted_while_queued",
        },
    ) in events


def test_turn_progress_does_not_invalidate_turn_bound_interrupt(monkeypatch):
    calls = []

    class Agent:
        _turn_liveness_activity_generation = 7
        _active_gateway_turn_id = "old-turn"

        def bind_gateway_turn(self, turn_id):
            self._active_gateway_turn_id = turn_id

        def hard_interrupt(
            self,
            _message=None,
            *,
            require_generation=None,
            require_turn_id=None,
        ):
            calls.append((require_generation, require_turn_id))
            return require_generation is None and require_turn_id == "old-turn"

    agent = Agent()
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    srv._interrupt_session_turn("sid", session)

    assert calls == [(None, "old-turn")]


def test_internal_turn_admitted_after_claim_is_not_hard_interrupted(monkeypatch):
    marker_entered = threading.Event()
    marker_release = threading.Event()
    hard_interrupts = []

    class Agent:
        _active_gateway_turn_id = "old-turn"

        def bind_gateway_turn(self, turn_id):
            self._active_gateway_turn_id = turn_id

        def hard_interrupt(self, _message=None, *, require_turn_id=None):
            if self._active_gateway_turn_id != require_turn_id:
                return False
            hard_interrupts.append(require_turn_id)
            return True

    agent = Agent()
    session = _session(agent)
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _active_turn_marker_key="same-key",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )

    def blocking_retire(*_args, **_kwargs):
        marker_entered.set()
        assert marker_release.wait(timeout=5.0)

    monkeypatch.setattr(srv, "_retire_turn_marker", blocking_retire)
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)
    worker = threading.Thread(
        target=srv._interrupt_session_turn,
        args=("sid", session),
        kwargs={"retire_turn_marker": True},
    )
    worker.start()
    assert marker_entered.wait(timeout=5.0)

    with session["history_lock"]:
        session["running"] = False
        srv._clear_active_turn_state(session)
    assert srv._notif_claim_turn(session) is None
    assert "_active_turn_id" not in session
    marker_release.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert hard_interrupts == ["old-turn"]
    assert srv._notif_claim_turn(session)
    new_turn_id = session["_active_turn_id"]
    assert new_turn_id != "old-turn"


def test_public_follower_upgrades_internal_interrupt_effects(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    tts_stops = []
    retired = []
    results = []

    class Agent:
        def bind_gateway_turn(self, _turn_id):
            pass

        def hard_interrupt(self, _message=None, *, require_turn_id=None):
            assert require_turn_id == "shared-turn"
            entered.set()
            assert release.wait(timeout=5.0)
            return True

    session = _session(Agent())
    session.update(
        running=True,
        session_key="shared-key",
        _active_turn_id="shared-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _active_turn_marker_key="shared-key",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: tts_stops.append(True))
    monkeypatch.setattr(
        srv,
        "_retire_turn_marker",
        lambda *args, **kwargs: retired.append((args, kwargs)),
    )
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    owner = threading.Thread(
        target=lambda: results.append(srv._interrupt_session_turn("sid", session))
    )
    follower = threading.Thread(
        target=lambda: results.append(
            srv._interrupt_session_turn(
                "sid", session, stop_tts=True, retire_turn_marker=True
            )
        )
    )
    owner.start()
    assert entered.wait(timeout=5.0)
    follower.start()
    follower.join(timeout=0.1)
    assert follower.is_alive()
    release.set()
    owner.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert not owner.is_alive()
    assert not follower.is_alive()
    assert results == [False, False]
    assert tts_stops == [True]
    assert len(retired) == 1
    assert retired[0][1]["expected_turn_id"] == "shared-turn"


def test_follower_effect_upgrade_runs_after_marker_retirement_failure(monkeypatch):
    marker_entered = threading.Event()
    marker_release = threading.Event()
    tts_stops = []
    errors = []

    class Agent:
        def bind_gateway_turn(self, _turn_id):
            pass

        def hard_interrupt(self, _message=None, *, require_turn_id=None):
            return require_turn_id == "shared-turn"

    session = _session(Agent())
    session.update(
        running=True,
        session_key="shared-key",
        _active_turn_id="shared-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _active_turn_marker_key="shared-key",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )

    def failing_retire(*_args, **_kwargs):
        marker_entered.set()
        assert marker_release.wait(timeout=5.0)
        raise RuntimeError("marker failed")

    monkeypatch.setattr(srv, "_retire_turn_marker", failing_retire)
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: tts_stops.append(True))
    monkeypatch.setattr(srv, "_clear_pending", lambda *_args: None)

    def interrupt(**kwargs):
        try:
            srv._interrupt_session_turn("sid", session, **kwargs)
        except Exception as exc:
            errors.append(exc)

    owner = threading.Thread(
        target=interrupt, kwargs={"retire_turn_marker": True}
    )
    follower = threading.Thread(target=interrupt, kwargs={"stop_tts": True})
    owner.start()
    assert marker_entered.wait(timeout=5.0)
    follower.start()
    follower.join(timeout=0.1)
    assert follower.is_alive()
    marker_release.set()
    owner.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert not owner.is_alive()
    assert not follower.is_alive()
    assert tts_stops == [True]
    assert len(errors) == 2


def test_concurrent_interrupt_waits_for_owner_failure(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    errors = []

    class Supervisor:
        def interrupt(self, *_args, **_kwargs):
            entered.set()
            assert release.wait(timeout=5.0)
            raise RuntimeError("compute interrupt failed")

    session = _session(None)
    session.update(
        running=True,
        _active_turn_id="compute-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="compute",
        _compute_host_active=True,
    )
    monkeypatch.setattr(srv, "_get_compute_host_supervisor", lambda: Supervisor())

    def interrupt():
        try:
            srv._interrupt_session_turn("sid", session)
        except Exception as exc:
            errors.append(exc)

    owner = threading.Thread(target=interrupt)
    follower = threading.Thread(target=interrupt)
    owner.start()
    assert entered.wait(timeout=5.0)
    follower.start()
    follower.join(timeout=0.1)
    assert follower.is_alive()
    release.set()
    owner.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert not owner.is_alive()
    assert not follower.is_alive()
    assert len(errors) == 2
    assert all("interrupt failed" in str(error) for error in errors)


def test_concurrent_interrupt_propagates_post_claim_drain_failure(monkeypatch):
    entered_clear = threading.Event()
    release_clear = threading.Event()
    errors = []
    real_thread = threading.Thread

    session = _session(types.SimpleNamespace(interrupt=lambda: None))
    session.update(
        running=True,
        _active_turn_id="old-turn",
        _active_turn_authorization=TurnAuthorization.from_raw(None),
        _active_turn_route="inline",
        _run_thread=types.SimpleNamespace(is_alive=lambda: True),
    )

    def blocking_clear(*_args):
        entered_clear.set()
        assert release_clear.wait(timeout=5.0)

    monkeypatch.setattr(srv, "_clear_pending", blocking_clear)

    def interrupt():
        try:
            srv._interrupt_session_turn("sid", session)
        except Exception as exc:
            errors.append(exc)

    owner = real_thread(target=interrupt)
    follower = real_thread(target=interrupt)
    owner.start()
    assert entered_clear.wait(timeout=5.0)
    follower.start()
    follower.join(timeout=0.1)
    assert follower.is_alive()

    with session["history_lock"]:
        session["running"] = False
        session["queued_prompt"] = {"text": "queued", "transport": None}

    class BrokenThread:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("drain start failed")

    monkeypatch.setattr(srv.threading, "Thread", BrokenThread)
    release_clear.set()
    owner.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert not owner.is_alive()
    assert not follower.is_alive()
    assert len(errors) == 2
    assert all("failed" in str(error) for error in errors)


def test_direct_session_interrupt_rejects_copied_active_person_authorization(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {
                "session_id": "sid",
                "_fizko_person_access_token": "active-person",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "a" * 64,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert session["running"] is True
    assert "_active_turn_authorization" in session


def test_direct_session_interrupt_rejects_forged_same_principal(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["_active_turn_authorization"] = _authorization(
        "person-token-t1", principal_id="a" * 64
    )
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {
                "session_id": "sid",
                "_fizko_person_access_token": "person-token-t2",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "a" * 64,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert session["running"] is True


def test_different_person_principal_cannot_interrupt_active_turn(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["_active_turn_authorization"] = _authorization(
        "person-token-t1", principal_id="a" * 64
    )
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {
                "session_id": "sid",
                "_fizko_person_access_token": "person-token-t2",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "b" * 64,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert session["running"] is True


def test_reconnected_person_queues_rotated_bearer_behind_owned_active_turn(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    session["_active_turn_authorization"] = _authorization(
        "person-token-t1", principal_id="a" * 64
    )
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "queue")
    try:
        params = {
            "session_id": "sid",
            "text": "run after reconnect",
            "_fizko_person_access_token": "person-token-t2",
            "_fizko_person_access_token_expires_at": time.time() + 3600,
            "_fizko_person_principal_id": "a" * 64,
            "_fizko_person_admission_id": "8" * 32,
        }
        response = srv._methods["prompt.submit"]("r", params)
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"] == {"status": "queued"}
    queued = session["queued_prompt"]["turn_authorization"]
    assert queued.same_principal(session["_active_turn_authorization"])
    assert not queued.same_credential(session["_active_turn_authorization"])
    assert params == {"session_id": "sid", "text": "run after reconnect"}
    assert calls == {"steer": [], "redirect": [], "interrupt": []}


def test_interrupt_keeps_personal_fence_until_live_worker_settles(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    with session["history_lock"]:
        srv._activate_turn_identity(session)
    session["_run_thread"] = types.SimpleNamespace(is_alive=lambda: True)
    session["_run_thread_turn_id"] = session["_active_turn_id"]
    holder = session["_active_turn_authorization"]
    try:
        srv._interrupt_session_turn("sid", session)
        assert session["running"] is True
        assert session["_active_turn_authorization"] is holder
        assert srv._direct_personal_turn_mutation_error("next", session)["error"]["code"] == 4125

        with session["history_lock"]:
            session["running"] = False
            assert srv._clear_active_turn_state(session, holder)
        assert "_active_turn_authorization" not in session
    finally:
        srv._sessions.pop("sid", None)

    assert calls["interrupt"] == [True]


def test_expired_person_authorization_cannot_interrupt(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {
                "session_id": "sid",
                "_fizko_person_access_token": "active-person",
                "_fizko_person_access_token_expires_at": time.time() - 1,
                "_fizko_person_principal_id": "a" * 64,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert session["running"] is True


def test_direct_session_interrupt_rejects_different_person_authorization(monkeypatch):
    session, calls = _install_token_bearing_direct_rpc_session(monkeypatch)
    try:
        response = srv._methods["session.interrupt"](
            "r",
            {
                "session_id": "sid",
                "_fizko_person_access_token": "different-person",
                "_fizko_person_access_token_expires_at": time.time() + 3600,
                "_fizko_person_principal_id": "b" * 64,
            },
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["error"]["code"] == 4125
    assert calls["interrupt"] == []
    assert session["running"] is True
