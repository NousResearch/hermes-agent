"""Failed turns must retain a replayable ``inflight`` snapshot.

A turn that ended in error used to clear ``inflight_turn`` and emit its
terminal frame in the same breath. If the client was disconnected during that
window, the frame went to the detached drop-transport and the in-memory state
was already gone — the desktop reconnected to a session with no trace of the
failure (stuck spinner or a silently missing turn).

Contract pinned here:

* ``_fail_inflight_turn`` keeps the user prompt, partial assistant text, and
  error semantics; ``_inflight_snapshot`` exposes status/error/recoverable.
* The returned-error path (``run_conversation()`` returning ``error``) retains
  the snapshot — not just the exception path.
* The exception path closes the turn with a terminal ``message.complete``
  (``status: "error"``, same shape as the returned-error path) instead of a
  bare ``error`` event.
* ``session.resume``'s live payload carries the retained snapshot.
* A retained failure never leaks into the next turn's inflight state.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


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
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _events(captured, name):
    return [payload for event, _sid, payload in captured if event == name]


@pytest.mark.parametrize("result,kind", [
    ({"final_response": "done", "messages": [], "completed": True}, "completed"),
    ({"error": "provider unavailable", "failed": True, "failure_reason": "rate_limit"}, "provider_failed"),
    ({"interrupted": True}, "interrupted"),
    ({"failed": True, "error": "unclassified local error"}, "interrupted"),
])
def test_desktop_human_work_terminal(emits, turn_env, result, kind):
    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=lambda *a, **k: result,
        clear_interrupt=lambda: None)
    session = _session(agent=agent, source="desktop", profile="default", running=True)
    server._run_prompt_submit(
        "rid", "sid", session, "human text",
        desktop_work={"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"})
    work = _events(emits, "desktop.work")
    assert [e["kind"] for e in work] == ["started", kind]
    assert all(e["schema_version"] == 1 and e["origin"] == "desktop_user" for e in work)
    assert all(e["root_id"] == "11111111-1111-4111-8111-111111111111" for e in work)
    assert all(e["stored_session_id"] == "session-key" and e["title"] == "Chat senza titolo" for e in work)
    assert len({e["event_id"] for e in work}) == len(work)


def test_goal_continuation_keeps_one_desktop_root_until_terminal(
    emits, turn_env, monkeypatch
):
    calls = []

    def run_conversation(*args, **kwargs):
        calls.append((args, kwargs))
        return {"final_response": "done", "messages": [], "completed": True}

    followups = iter(["continue the goal", None])
    monkeypatch.setattr(
        server, "_goal_followup_after_turn", lambda *args, **kwargs: next(followups))
    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=run_conversation,
        clear_interrupt=lambda: None)
    session = _session(agent=agent, source="desktop", profile="default", running=True)

    server._run_prompt_submit(
        "rid", "sid", session, "human text",
        desktop_work={
            "origin": "desktop_user",
            "root_id": "11111111-1111-4111-8111-111111111111",
        })

    work = _events(emits, "desktop.work")
    assert len(calls) == 2
    assert [event["kind"] for event in work] == ["started", "waiting", "completed"]
    assert {event["root_id"] for event in work} == {
        "11111111-1111-4111-8111-111111111111"}


def test_ready_dispatch_preserves_explicit_human_admission(emits, turn_env, monkeypatch):
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: {"completed": True, "messages": [], "final_response": "ok"})
    session = _session(agent=agent, source="desktop", running=True)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *a: None)
    proof = {"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"}
    server._run_after_agent_ready(
        "rid", "sid", session, "human", None, None, desktop_work=proof)
    assert [p["kind"] for p in _events(emits, "desktop.work")] == ["started", "completed"]


@pytest.mark.parametrize("wait_result,cancelled", [
    ({"error": {"message": "agent initialization failed"}}, False),
    (None, True),
])
def test_desktop_human_work_closes_before_agent_ready(
    emits, turn_env, monkeypatch, wait_result, cancelled
):
    session = _session(
        source="desktop", running=True, _turn_cancel_requested=cancelled)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *a: wait_result)
    proof = {"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"}

    server._run_after_agent_ready(
        "rid", "sid", session, "human", None, None, desktop_work=proof)

    assert [p["kind"] for p in _events(emits, "desktop.work")] == ["started", "interrupted"]


def test_desktop_human_work_closes_when_turn_thread_cannot_start(emits, turn_env, monkeypatch):
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None)
    session = _session(agent=agent, source="desktop", running=True)
    proof = {"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"}

    def race_after_admission(*args):
        session["_closing"] = True
        return [], agent

    monkeypatch.setattr(server, "_admit_prompt_turn", race_after_admission)
    accepted = server._run_prompt_submit(
        "rid", "sid", session, "human", desktop_work=proof)

    assert accepted is False
    assert [p["kind"] for p in _events(emits, "desktop.work")] == ["started", "interrupted"]


def test_deferred_desktop_work_closes_when_admission_is_lost(emits, monkeypatch):
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None)
    session = _session(agent=agent, source="desktop", running=True)
    proof = {"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"}
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *args: None)
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *args: None)

    server._run_after_agent_ready(
        "rid", "sid", session, "human", None, None, desktop_work=proof)

    assert [p["kind"] for p in _events(emits, "desktop.work")] == ["started", "interrupted"]


def test_human_approval_is_correlated_to_running_work(emits, turn_env):
    def invoke(*a, **k):
        server._emit_approval_request("sid", {"request_id": "approval-1", "command": "sensitive"})
        server._emit_approval_request("sid", {"request_id": "approval-1", "command": "sensitive"})
        return {"completed": True, "messages": [], "final_response": "ok"}
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None, run_conversation=invoke)
    session = _session(agent=agent, source="desktop", running=True)
    server._run_prompt_submit("rid", "sid", session, "human", desktop_work={
        "origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"})
    work = _events(emits, "desktop.work")
    assert [p["kind"] for p in work] == ["started", "approval", "completed"]
    assert work[1]["request_id"] == "approval-1"
    server._emit_approval_request("sid", {"request_id": "after-turn"})
    assert len(_events(emits, "desktop.work")) == 3


def test_async_continuation_finishes_original_work(emits, turn_env):
    from tui_gateway.desktop_work import admit
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: {"completed": True, "messages": [], "final_response": "ok"})
    session = _session(agent=agent, source="desktop", running=True)
    work = admit("sid", session, {"origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"})
    work.emit(server._emit, "started")
    work.retain_async("batch")
    server._run_prompt_submit("rid", "sid", session, "results", display_kind="async_delegation_complete",
                              continued_desktop_work=work)
    assert "completed" not in [p["kind"] for p in _events(emits, "desktop.work")]
    work.consume_async("batch")
    assert [p["kind"] for p in _events(emits, "desktop.work")] == ["started", "waiting", "completed"]


@pytest.mark.parametrize("provider_error", [True, False])
@pytest.mark.parametrize("next_admission", [True, False])
def test_escaped_exception_has_terminal_work_classification(emits, turn_env, monkeypatch, provider_error, next_admission):
    import httpx
    from openai import AuthenticationError
    error = (AuthenticationError("auth failed", response=httpx.Response(401,
        request=httpx.Request("POST", "https://example.invalid")), body=None)
        if provider_error else RuntimeError("local dispatcher failed"))
    if next_admission:
        # A new turn can replace the retained snapshot after running is released.
        monkeypatch.setattr(server, "_emit_settled_session_info",
            lambda sid, session, agent: session.update(inflight_turn=None))
    def invoke(*a, **k):
        raise error
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None, run_conversation=invoke)
    session = _session(agent=agent, source="desktop", running=True)
    server._run_prompt_submit("rid", "sid", session, "human", desktop_work={
        "origin": "desktop_user", "root_id": "11111111-1111-4111-8111-111111111111"})
    assert [p["kind"] for p in _events(emits, "desktop.work")] == [
        "started", "provider_failed" if provider_error else "interrupted"]


# ── Unit: retention helpers ───────────────────────────────────────────


def test_fail_inflight_turn_retains_partial_and_error():
    session = _session()
    server._start_inflight_turn(session, "do the thing")
    server._append_inflight_delta(session, "partial answer")

    server._fail_inflight_turn(session, RuntimeError("provider exploded"))

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["user"] == "do the thing"
    assert snapshot["assistant"] == "partial answer"
    assert snapshot["streaming"] is False
    assert snapshot["error"] == "provider exploded"
    assert snapshot["status"] == "error"
    assert snapshot["recoverable"] is True


def test_snapshot_returned_for_error_only_turn():
    """An init failure has no user/assistant text yet — the error alone must
    survive the emptiness check, or resume shows nothing."""
    session = _session()
    server._fail_inflight_turn(session, "agent initialization failed")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["error"] == "agent initialization failed"


def test_healthy_snapshot_carries_no_error_keys():
    session = _session()
    server._start_inflight_turn(session, "hi")
    server._append_inflight_delta(session, "hello")

    snapshot = server._inflight_snapshot(session)
    assert snapshot == {"assistant": "hello", "streaming": True, "user": "hi"}


# ── Returned-error path (run_conversation returns an error result) ────


def test_returned_error_result_retains_snapshot_and_emits_terminal_frame(
    emits, turn_env
):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "provider 402: billing wall",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    payload = completes[0]
    assert payload["status"] == "error"
    assert payload["error"] == "provider 402: billing wall"
    assert payload["recoverable"] is True

    # The retained snapshot survives the finally block for resume replay.
    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["status"] == "error"
    assert snapshot["error"] == "provider 402: billing wall"
    assert snapshot["user"] == "do the thing"
    assert session["running"] is False


def test_returned_error_result_carries_error_surface(emits, turn_env):
    """A classified failure_reason rides the terminal frame AND the retained
    snapshot as a structured {layer, code, retryable} descriptor, so the
    desktop names the failing layer instead of sniffing the message."""
    agent = types.SimpleNamespace(
        session_id="session-key",
        provider="openrouter",
        model="test/model",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "Rate limit exceeded",
            "failed": True,
            "failure_reason": "rate_limit",
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    payload = _events(emits, "message.complete")[0]
    assert payload["error_surface"] == {
        "layer": "provider",
        "code": "rate_limit",
        "retryable": True,
        # The failing session's identity rides the descriptor so clients
        # report the model that actually failed, not the composer's current.
        "provider": "openrouter",
        "model": "test/model",
    }

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["error_surface"]["layer"] == "provider"


def test_returned_error_without_reason_omits_no_frame(emits, turn_env):
    """Legacy result dicts (no failure_reason) still get a best-effort
    descriptor — never a crash, never a missing terminal frame."""
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "something odd",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "go")

    server._run_prompt_submit("rid", "sid", session, "go")

    payload = _events(emits, "message.complete")[0]
    assert payload["status"] == "error"
    assert payload["error_surface"]["layer"] == "provider"
    assert payload["error_surface"]["code"] == "unknown"


def test_completed_turn_still_clears_inflight(emits, turn_env):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {"final_response": "all done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "complete"
    assert "error" not in completes[0]
    assert server._inflight_snapshot(session) is None


# ── Exception path ─────────────────────────────────────────────────────


def test_exception_closes_turn_with_terminal_complete_and_partial(emits, turn_env):
    def _boom(message, stream_callback=None, **kwargs):
        if stream_callback is not None:
            stream_callback("half an ans")
        raise RuntimeError("connection reset mid-stream")

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=_boom,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    # Terminal frame, not a bare error event.
    assert not _events(emits, "error")
    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    payload = completes[0]
    assert payload["status"] == "error"
    assert payload["error"] == "connection reset mid-stream"
    assert payload["recoverable"] is True
    assert payload["partial"] is True
    assert payload["text"] == "half an ans"

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert snapshot["assistant"] == "half an ans"
    assert snapshot["error"] == "connection reset mid-stream"
    assert session["running"] is False

    # Dispatcher-side exceptions (not API errors) classify as gateway-layer.
    assert payload["error_surface"]["layer"] == "gateway"
    assert snapshot["error_surface"]["layer"] == "gateway"


# ── Resume replay (the reason retention exists) ───────────────────────


def test_live_session_payload_exposes_retained_failure(emits, turn_env, monkeypatch):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "budget exhausted",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "long job")
    server._run_prompt_submit("rid", "sid", session, "long job")

    # What session.resume's live fast path hands a reconnecting client.
    monkeypatch.setattr(server, "_get_db", lambda: None)
    payload = server._live_session_payload("sid", session)

    assert payload["running"] is False
    inflight = payload.get("inflight")
    assert inflight is not None
    assert inflight["status"] == "error"
    assert inflight["error"] == "budget exhausted"
    assert inflight["user"] == "long job"


# ── Retained failure must not leak into the next turn ─────────────────


def test_next_turn_replaces_retained_error_snapshot(emits, turn_env):
    seen_inflight_user: list = []

    def _run_ok(message, **kwargs):
        # Capture what the inflight turn looks like while the new turn runs.
        turn = server._inflight_snapshot(_run_ok.session)
        seen_inflight_user.append(turn and turn["user"])
        return {"final_response": "fresh answer"}

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=_run_ok,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    _run_ok.session = session

    # Leftover retained failure from a previous turn.
    server._start_inflight_turn(session, "old failed prompt")
    server._fail_inflight_turn(session, "previous turn failed")

    server._run_prompt_submit("rid", "sid", session, "new prompt")

    # The new turn must have started a fresh inflight turn, not inherited the
    # failed one (the retained dict used to satisfy the isinstance guard).
    assert seen_inflight_user == ["new prompt"]
    snapshot = server._inflight_snapshot(session)
    assert snapshot is None
    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "complete"
