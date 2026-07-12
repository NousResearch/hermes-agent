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


def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "steered"
    assert session.get("queued_prompt") is None


def test_busy_steer_mode_falls_back_to_queue_when_rejected(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: False, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "nudge"


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


def test_drain_requeues_prompt_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    queued = {"text": "go", "transport": "ws-1"}
    session = _session(queued_prompt=queued)

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must neither wedge the session nor drop the claimed prompt.
    assert session["running"] is False
    assert session["queued_prompt"] == queued


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


def _model_response(text):
    message = types.SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], model="test/model", usage=None)


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
