"""A prompt that lands mid-turn is redirected or queued, never dropped.

Before this, ``prompt.submit`` on a running session returned ``session busy``,
forcing clients into a deadline-bounded busy-retry. When turn teardown outlived
the deadline — e.g. a slow, non-interruptible tool (``web_search``) still
running when the user hit stop — the resubmitted message was silently dropped
("it just doesn't listen"). The gateway now applies the ``busy_input_mode``
policy: redirect the live turn by default, with the legacy interrupt + queue
path retained as a compatibility fallback.
"""

import threading
import time
import types

from hermes_state import SessionDB
from run_agent import AIAgent

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


# ── _enqueue_prompt ────────────────────────────────────────────────────────

def test_enqueue_pins_text_and_transport():
    session = _session()
    server._enqueue_prompt(session, "hello", "ws-1")
    assert session["queued_prompt"]["text"] == "hello"
    assert session["queued_prompt"]["transport"] == "ws-1"


def test_enqueue_preserves_distinct_messages_and_submission_metadata():
    session = _session()
    server._enqueue_prompt(
        session,
        "first",
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-1",
    )
    server._enqueue_prompt(
        session,
        "second",
        "ws-2",
        submitted_at=102.5,
        message_id="desktop-2",
    )

    assert session["queued_prompt"] == {
        "text": "first",
        "transport": "ws-1",
        "submitted_at": 101.25,
        "message_id": "desktop-1",
    }
    assert session["queued_prompts"] == [
        {
            "text": "second",
            "transport": "ws-2",
            "submitted_at": 102.5,
            "message_id": "desktop-2",
        }
    ]


def test_enqueue_keeps_one_multi_paragraph_prompt_as_one_message():
    session = _session()
    text = "first paragraph\n\nsecond paragraph"

    server._enqueue_prompt(
        session,
        text,
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-1",
    )

    assert session["queued_prompt"]["text"] == text
    assert session.get("queued_prompts", []) == []


# ── _handle_busy_submit (policy) ───────────────────────────────────────────

def test_busy_interrupt_mode_redirects_active_turn(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {"user": "original request", "assistant": "partial reply"}

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert seen == ["redirect"]
    assert session["inflight_turn"]["user"] == "redirect"
    assert session.get("queued_prompt") is None


def test_busy_interrupt_mode_falls_back_for_legacy_agent(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "queued"
    deadline = time.monotonic() + 1
    while calls["interrupt"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "redirect"


def test_busy_queue_mode_queues_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "later", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 0
    assert session["queued_prompt"]["text"] == "later"


def test_busy_steer_mode_injects_when_accepted_without_enqueueing(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    calls = {"interrupt": 0, "steer": []}
    agent = types.SimpleNamespace(
        interrupt=lambda: calls.__setitem__("interrupt", calls["interrupt"] + 1),
        steer=lambda text: (calls["steer"].append(text), True)[1],
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        "nudge",
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-steer-1",
    )

    assert resp["result"]["status"] == "steered"
    assert calls == {"interrupt": 0, "steer": ["nudge"]}
    assert session.get("queued_prompt") is None


def test_busy_steer_mode_rejection_queues_with_source_identity(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    calls = {"interrupt": 0, "steer": []}
    agent = types.SimpleNamespace(
        interrupt=lambda: calls.__setitem__("interrupt", calls["interrupt"] + 1),
        steer=lambda text: (calls["steer"].append(text), False)[1],
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        "nudge",
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-steer-1",
    )

    assert resp["result"]["status"] == "queued"
    assert calls == {"interrupt": 1, "steer": ["nudge"]}
    assert session["queued_prompt"] == {
        "text": "nudge",
        "transport": "ws-1",
        "submitted_at": 101.25,
        "message_id": "desktop-steer-1",
    }


def test_busy_steer_mode_unavailable_queues_with_source_identity(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        "nudge",
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-steer-1",
    )

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 1
    assert session["queued_prompt"] == {
        "text": "nudge",
        "transport": "ws-1",
        "submitted_at": 101.25,
        "message_id": "desktop-steer-1",
    }


def test_busy_interrupt_does_not_hold_history_lock_or_delay_queue(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupt_started = threading.Event()
    release_interrupt = threading.Event()

    def blocking_interrupt():
        interrupt_started.set()
        release_interrupt.wait(timeout=2)

    session = _session(
        agent=types.SimpleNamespace(interrupt=blocking_interrupt),
        running=True,
    )

    started = time.monotonic()
    resp = server._handle_busy_submit("r1", "sid", session, "keep this", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert time.monotonic() - started < 0.25
    assert session["queued_prompt"]["text"] == "keep this"
    assert interrupt_started.wait(timeout=1)
    assert session["history_lock"].acquire(timeout=0.25)
    session["history_lock"].release()
    release_interrupt.set()


def test_busy_helper_retries_when_turn_finished(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    session = _session(running=False)

    assert server._handle_busy_submit("r1", "sid", session, "run now", "ws-1") is None
    assert session.get("queued_prompt") is None
def test_prompt_submit_dedupes_explicit_id_already_inflight(monkeypatch):
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(
        agent=agent,
        running=True,
        inflight_turn={"message_id": "desktop-1", "user": "first"},
    )
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: "ws-2")

    response = server.handle_request(
        {
            "id": "rpc-2",
            "method": "prompt.submit",
            "params": {
                "message_id": "desktop-1",
                "session_id": "sid",
                "text": "first",
            },
        }
    )

    assert response is not None
    assert response["result"]["status"] == "duplicate"
    assert session.get("queued_prompt") is None
    assert calls["interrupt"] == 0



def test_prompt_submit_duplicate_rehomes_only_matching_queued_source(monkeypatch):
    session = _session(
        running=True,
        transport="ws-current",
        queued_prompt={
            "text": "first",
            "transport": "ws-old",
            "message_id": "desktop-1",
        },
        queued_prompts=[
            {
                "text": "second",
                "transport": "ws-still-live",
                "message_id": "desktop-2",
            }
        ],
    )
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: "ws-retry")

    response = server.handle_request(
        {
            "id": "rpc-retry",
            "method": "prompt.submit",
            "params": {
                "message_id": "desktop-1",
                "session_id": "sid",
                "text": "first",
            },
        }
    )

    assert response is not None
    assert response["result"]["status"] == "duplicate"
    assert session["transport"] == "ws-retry"
    assert session["queued_prompt"]["transport"] == "ws-retry"
    assert session["queued_prompts"][0]["transport"] == "ws-still-live"


def test_prompt_submit_does_not_dedupe_reused_rpc_id_without_explicit_id(
    monkeypatch,
):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(
        running=True,
        inflight_turn={"message_id": "rpc-1", "user": "prior connection"},
    )
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: "ws-new")

    response = server.handle_request(
        {
            "id": "rpc-1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "new connection prompt"},
        }
    )

    assert response is not None
    assert response["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "new connection prompt"
    assert "message_id" not in session["queued_prompt"]


def test_prompt_id_dedupe_uses_persisted_source_id(tmp_path):
    db = SessionDB(tmp_path / "dedupe.db")
    try:
        db.create_session("session-key", source="desktop", model="test/model")
        db.append_message(
            session_id="session-key",
            role="user",
            content="already accepted",
            platform_message_id="desktop-persisted",
        )
        session = _session(agent=types.SimpleNamespace(_session_db=db))

        assert server._has_prompt_message_id(session, "desktop-persisted") is True
        assert server._has_prompt_message_id(session, "desktop-new") is False
    finally:
        db.close()


def test_busy_interrupt_mode_normalizes_rich_text_before_redirect(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)
    rich = [{"type": "text", "text": "  redirect me  "}]

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        rich,
        "ws-1",
    )

    assert resp["result"]["status"] == "redirected"
    assert seen == ["redirect me"]
    assert session.get("queued_prompt") is None


def test_busy_queue_fallback_preserves_original_structured_text(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    rich = [{"type": "text", "text": "  keep me  "}]
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: False,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == rich


def test_busy_interrupt_mode_queues_multimodal_payload_instead_of_redirect(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    rich = [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert seen == []
    assert session["queued_prompt"]["text"] == rich


# ── _drain_queued_prompt ───────────────────────────────────────────────────

def test_drain_fires_queued_prompt_and_claims_running(monkeypatch):
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session(queued_prompt={"text": "go", "transport": "ws-9"})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "go"}
    assert session["running"] is True
    assert session["queued_prompt"] is None
    assert session["transport"] == "ws-9"


def test_drain_noop_when_nothing_queued(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session()
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["running"] is False


def test_drain_noop_when_session_already_running(monkeypatch):
    """A fresh turn that claimed the session beats a stale queued entry —
    the drain leaves it for that turn's own tail."""
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session(running=True, queued_prompt={"text": "go", "transport": None})
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["queued_prompt"]["text"] == "go"


def test_drain_failure_restores_exact_item_before_later_arrivals(monkeypatch):
    first = {
        "text": "first",
        "transport": "ws-1",
        "submitted_at": 101.25,
        "message_id": "desktop-1",
    }
    second = {
        "text": "second",
        "transport": "ws-2",
        "submitted_at": 102.5,
        "message_id": "desktop-2",
    }

    def _boom(_rid, _sid, session, _text, **_kwargs):
        server._enqueue_prompt(
            session,
            "third",
            "ws-3",
            submitted_at=103.75,
            message_id="desktop-3",
        )
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session(queued_prompt=first, queued_prompts=[second])

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert session["running"] is False
    assert session["inflight_turn"] is None
    assert session["queued_prompt"] is first
    assert session["queued_prompts"] == [
        second,
        {
            "text": "third",
            "transport": "ws-3",
            "submitted_at": 103.75,
            "message_id": "desktop-3",
        },
    ]


def test_drain_claim_dedupes_retry_before_dispatch(monkeypatch):
    retry_response = None
    queued = {
        "text": "first",
        "transport": "ws-original",
        "submitted_at": 101.25,
        "message_id": "stable-1",
    }
    session = _session(queued_prompt=queued)
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: "ws-retry")

    def _run(_rid, _sid, _session, _text, **_kwargs):
        nonlocal retry_response
        retry_response = server.handle_request(
            {
                "id": "rpc-retry",
                "method": "prompt.submit",
                "params": {
                    "message_id": "stable-1",
                    "session_id": "sid",
                    "submitted_at": 101.25,
                    "text": "first",
                },
            }
        )

    monkeypatch.setattr(server, "_run_prompt_submit", _run)

    assert server._drain_queued_prompt("rpc-original", "sid", session) is True
    assert retry_response is not None
    assert retry_response["result"]["status"] == "duplicate"
    assert session.get("queued_prompt") is None
    assert session.get("queued_prompts", []) == []
    assert session["inflight_turn"]["message_id"] == "stable-1"
    assert session["inflight_turn"]["submitted_at"] == 101.25


def test_repeated_arrivals_drain_once_in_order_to_their_own_transports(monkeypatch):
    fired = []

    def _run(rid, sid, session, text, **kwargs):
        fired.append(
            {
                "rid": rid,
                "sid": sid,
                "text": text,
                "transport": session["transport"],
                **kwargs,
            }
        )
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _run)
    session = _session()
    for index in range(3):
        server._enqueue_prompt(
            session,
            f"message-{index}",
            f"ws-{index}",
            submitted_at=100.0 + index,
            message_id=f"desktop-{index}",
        )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert server._drain_queued_prompt("r1", "sid", session) is False

    assert fired == [
        {
            "rid": "r1",
            "sid": "sid",
            "text": f"message-{index}",
            "transport": f"ws-{index}",
            "submitted_at": 100.0 + index,
            "message_id": f"desktop-{index}",
        }
        for index in range(3)
    ]
    assert session["queued_prompt"] is None
    assert session.get("queued_prompts", []) == []


class _RecordingTransport:
    def __init__(self, completed: threading.Event | None = None):
        self._closed = False
        self.completed = completed
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        event_type = ((obj.get("params") or {}).get("type"))
        if event_type == "message.complete" and self.completed is not None:
            self.completed.set()
        return not self._closed

    def close(self):
        self._closed = True


def test_session_activate_rehomes_dead_queue_item_and_preserves_live_tail(
    monkeypatch,
):
    dead_head_transport = _RecordingTransport()
    dead_head_transport.close()
    current_live_transport = _RecordingTransport()
    live_tail_transport = _RecordingTransport()
    activated_transport = _RecordingTransport()
    session = _session(
        transport=current_live_transport,
        queued_prompt={
            "text": "dead head",
            "transport": dead_head_transport,
            "message_id": "desktop-dead",
        },
        queued_prompts=[
            {
                "text": "live tail",
                "transport": live_tail_transport,
                "message_id": "desktop-live",
            }
        ],
    )
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "current_transport", lambda: activated_transport)

    response = server.handle_request(
        {
            "id": "rpc-activate",
            "method": "session.activate",
            "params": {"session_id": "sid"},
        }
    )

    assert response["result"]["session_id"] == "sid"
    assert session["transport"] is activated_transport
    assert session["queued_prompt"]["transport"] is activated_transport
    assert session["queued_prompts"][0]["transport"] is live_tail_transport


def test_disconnect_snapshot_cannot_overwrite_inflight_duplicate_retry(
    monkeypatch,
):
    snapshot_taken = threading.Event()
    finish_snapshot = threading.Event()
    old_transport = _RecordingTransport()
    new_transport = _RecordingTransport()
    old_transport.close()
    sid = "race-ui"
    session = _session(
        running=True,
        transport=old_transport,
        inflight_turn={
            "message_id": "desktop-race-1",
            "user": "survive disconnect race",
        },
    )

    class _SnapshotBarrierSessions(dict):
        def items(self):
            snapshot = list(super().items())
            snapshot_taken.set()
            finish_snapshot.wait(10)
            return snapshot

    sessions = _SnapshotBarrierSessions({sid: session})
    disconnect_result = {}
    disconnect_errors = []

    def _disconnect():
        try:
            disconnect_result["value"] = server._close_sessions_for_transport(
                old_transport
            )
        except BaseException as exc:
            disconnect_errors.append(exc)

    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: new_transport)
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda *_a, **_k: None)
    disconnect_thread = threading.Thread(target=_disconnect)
    disconnect_thread.start()

    try:
        assert snapshot_taken.wait(10), "disconnect did not snapshot the old owner"
        duplicate = server.handle_request(
            {
                "id": "rpc-retry",
                "method": "prompt.submit",
                "params": {
                    "message_id": "desktop-race-1",
                    "session_id": sid,
                    "text": "survive disconnect race",
                },
            }
        )
        assert duplicate["result"]["status"] == "duplicate"
        assert session["transport"] is new_transport
    finally:
        finish_snapshot.set()
        disconnect_thread.join(10)

    assert not disconnect_thread.is_alive()
    assert disconnect_errors == []
    assert disconnect_result["value"] == (0, 0)
    assert session["transport"] is new_transport

    server._emit("message.start", sid)
    server._emit("message.delta", sid, {"text": "delta"})
    server._emit("message.complete", sid, {"text": "complete"})
    assert old_transport.frames == []
    assert [
        (frame.get("params") or {}).get("type") for frame in new_transport.frames
    ] == ["message.start", "message.delta", "message.complete"]


def _model_response(text):
    message = types.SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_busy_steer_rejection_dedupes_and_persists_one_canonical_turn(
    monkeypatch,
    tmp_path,
):
    db = SessionDB(tmp_path / "steer-source.db")
    session_key = "steer-source"
    db.create_session(session_key, source="desktop", model="test/model")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="custom",
        model="test/model",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_key,
    )
    agent._session_db_created = True
    agent._cached_system_prompt = "You are a test assistant."
    agent._disable_streaming = True

    steer_calls = []
    interrupt_calls = []
    wire_requests = []
    completed = threading.Event()
    monkeypatch.setattr(agent, "steer", lambda text: (steer_calls.append(text), False)[1])
    monkeypatch.setattr(agent, "interrupt", lambda: interrupt_calls.append(True))
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: (
            wire_requests.append(api_kwargs["messages"]),
            _model_response("ack"),
        )[1],
    )
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_get_usage", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    session = _session(agent=agent, session_key=session_key, running=True)
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a, **_k: (session, None))
    monkeypatch.setattr(server, "current_transport", lambda: "ws-steer")
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, _sid, _payload=None: completed.set()
        if event == "message.complete"
        else None,
    )
    request = {
        "id": "rpc-steer",
        "method": "prompt.submit",
        "params": {
            "message_id": "desktop-steer-1",
            "session_id": "ui-session",
            "submitted_at": 101.25,
            "text": "canonical nudge",
        },
    }

    first = server.handle_request(request)
    duplicate = server.handle_request({**request, "id": "rpc-steer-retry"})

    assert first["result"]["status"] == "queued"
    assert duplicate["result"]["status"] == "duplicate"
    assert steer_calls == ["canonical nudge"]
    assert interrupt_calls == [True]
    assert session["queued_prompt"]["message_id"] == "desktop-steer-1"
    assert session.get("queued_prompts", []) == []

    session["running"] = False
    assert server._drain_queued_prompt("rpc-steer", "ui-session", session) is True
    assert completed.wait(10), "steer-fallback queued turn did not complete"

    canonical_users = [
        message for message in session["history"] if message.get("role") == "user"
    ]
    user_rows = [row for row in db.get_messages(session_key) if row["role"] == "user"]
    assert [message["content"] for message in canonical_users] == ["canonical nudge"]
    assert [(row["content"], row["platform_message_id"]) for row in user_rows] == [
        ("canonical nudge", "desktop-steer-1")
    ]
    assert len(wire_requests) == 1


def test_reconnect_rehomes_queued_turn_and_routes_all_events_to_live_transport(
    monkeypatch,
    tmp_path,
):
    db = SessionDB(tmp_path / "reconnect-source.db")
    session_key = "reconnect-source"
    sid = "reconnect-ui"
    db.create_session(session_key, source="desktop", model="test/model")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="custom",
        model="test/model",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_key,
    )
    agent._session_db_created = True
    agent._cached_system_prompt = "You are a test assistant."
    agent._disable_streaming = True

    wire_requests = []
    completed = threading.Event()
    old_transport = _RecordingTransport()
    new_transport = _RecordingTransport(completed)
    active_transport = {"value": old_transport}
    original_run = agent.run_conversation

    def _run_with_delta(user_message, **kwargs):
        result = original_run(user_message, **kwargs)
        kwargs["stream_callback"]("ack-delta")
        return result

    monkeypatch.setattr(agent, "run_conversation", _run_with_delta)
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: (
            wire_requests.append(api_kwargs["messages"]),
            _model_response("ack"),
        )[1],
    )
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_get_usage", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "current_transport", lambda: active_transport["value"])
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda *_a, **_k: None)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    session = _session(
        agent=agent,
        session_key=session_key,
        running=True,
        transport=old_transport,
    )
    missing = object()
    previous_session = server._sessions.get(sid, missing)
    server._sessions[sid] = session
    request = {
        "id": "rpc-original",
        "method": "prompt.submit",
        "params": {
            "message_id": "desktop-reconnect-1",
            "session_id": sid,
            "submitted_at": 101.25,
            "text": "survive reconnect",
        },
    }

    try:
        first = server.handle_request(request)
        assert first["result"]["status"] == "queued"
        assert session["queued_prompt"]["transport"] is old_transport

        old_transport.close()
        server._close_sessions_for_transport(old_transport)
        assert session["transport"] is server._detached_ws_transport

        active_transport["value"] = new_transport
        resumed = server.handle_request(
            {
                "id": "rpc-resume",
                "method": "session.resume",
                "params": {"session_id": session_key},
            }
        )
        assert resumed["result"]["session_id"] == sid
        assert session["transport"] is new_transport
        assert session["queued_prompt"]["transport"] is new_transport

        duplicate = server.handle_request({**request, "id": "rpc-retry"})
        assert duplicate["result"]["status"] == "duplicate"
        assert session["queued_prompt"]["transport"] is new_transport

        with session["history_lock"]:
            session["running"] = False
        assert server._drain_queued_prompt("rpc-drain", sid, session) is True
        assert completed.wait(10), "reconnected client did not receive completion"
        run_thread = session.get("_run_thread")
        assert run_thread is not None
        run_thread.join(10)
        assert not run_thread.is_alive()

        old_event_types = [
            (frame.get("params") or {}).get("type") for frame in old_transport.frames
        ]
        new_event_types = [
            (frame.get("params") or {}).get("type") for frame in new_transport.frames
        ]
        assert old_event_types == []
        assert {
            "message.start",
            "message.delta",
            "message.complete",
        }.issubset(new_event_types)

        canonical_users = [
            message for message in session["history"] if message.get("role") == "user"
        ]
        assert len(canonical_users) == 1
        assert canonical_users[0]["content"] == "survive reconnect"
        assert canonical_users[0]["timestamp"] == 101.25
        assert canonical_users[0]["_source_message_id"] == "desktop-reconnect-1"

        user_rows = [row for row in db.get_messages(session_key) if row["role"] == "user"]
        assert [(row["content"], row["platform_message_id"]) for row in user_rows] == [
            ("survive reconnect", "desktop-reconnect-1")
        ]
        assert len(wire_requests) == 1
        assert session["queued_prompt"] is None
        assert session.get("queued_prompts", []) == []
        assert session["inflight_turn"] is None
    finally:
        run_thread = session.get("_run_thread")
        if run_thread is not None:
            run_thread.join(10)
        if previous_session is missing:
            server._sessions.pop(sid, None)
        else:
            server._sessions[sid] = previous_session
        db.close()


def test_drain_persists_distinct_users_and_sends_valid_ordered_wire_history(
    monkeypatch,
    tmp_path,
):
    """Exercise the real gateway drain, AIAgent loop, SessionDB, and wire copy."""
    db = SessionDB(tmp_path / "state.db")
    session_key = "queued-boundaries"
    db.create_session(session_key, source="desktop", model="test/model")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="custom",
        model="test/model",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_key,
    )
    agent._session_db_created = True
    agent._cached_system_prompt = "You are a test assistant."
    agent._disable_streaming = True

    wire_requests = []
    replies = iter(("ack-first", "ack-second"))

    def _api_call(api_kwargs):
        wire_requests.append(api_kwargs["messages"])
        return _model_response(next(replies))

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_get_usage", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    completed = threading.Event()
    completion_count = 0

    def _emit(event, _sid, _payload=None):
        nonlocal completion_count
        if event == "message.complete":
            completion_count += 1
            if completion_count == 2:
                completed.set()

    monkeypatch.setattr(server, "_emit", _emit)

    session = _session(agent=agent, session_key=session_key)
    server._enqueue_prompt(
        session,
        "first queued prompt",
        "ws-1",
        submitted_at=101.25,
        message_id="desktop-1",
    )
    server._enqueue_prompt(
        session,
        "second queued prompt",
        "ws-2",
        submitted_at=102.5,
        message_id="desktop-2",
    )

    assert server._drain_queued_prompt("r1", "ui-session", session) is True
    assert completed.wait(10), "queued turns did not both complete"

    user_rows = [row for row in db.get_messages(session_key) if row["role"] == "user"]
    assert [row["content"] for row in user_rows] == [
        "first queued prompt",
        "second queued prompt",
    ]
    assert [row["timestamp"] for row in user_rows] == [101.25, 102.5]
    assert [row["platform_message_id"] for row in user_rows] == [
        "desktop-1",
        "desktop-2",
    ]

    assert len(wire_requests) == 2
    assert [
        message["content"]
        for message in wire_requests[1]
        if message.get("role") == "user"
    ] == ["first queued prompt", "second queued prompt"]
    for request in wire_requests:
        non_system_roles = [
            message["role"] for message in request if message.get("role") != "system"
        ]
        assert all(
            left != right
            for left, right in zip(non_system_roles, non_system_roles[1:])
        )
        assert all("timestamp" not in message for message in request)
        assert all("_source_message_id" not in message for message in request)
        assert all("message_id" not in message for message in request)
        assert all("platform_message_id" not in message for message in request)
