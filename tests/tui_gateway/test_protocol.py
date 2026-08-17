"""Tests for tui_gateway JSON-RPC protocol plumbing."""

import io
import json
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    # The sys.modules mocks only need to cover the *initial* import — once
    # tui_gateway.server is cached, they are inert. Keeping them active for
    # the whole test poisons any module first imported inside a test body:
    # e.g. hermes_cli.active_sessions would bind the mocked get_hermes_home
    # (a fixed shared path) forever, leaking active-session registry entries
    # across every later test in the process. Scope the patch to the import.
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")

    # Snapshot the RPC registry: several tests below stub handlers
    # ("slash.exec", "fast.ping", ...) directly in the module-level dict,
    # which is shared with every other test file in the process.
    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    # Reset module-level state without re-importing. importlib.reload
    # would re-register the module's atexit hooks (ThreadPoolExecutor
    # shutdown, _shutdown_sessions); the duplicates race the stderr
    # buffer at interpreter shutdown and surface as Fatal Python error:
    # _enter_buffered_busy. Restoring the dicts in place gives the next
    # test a clean slate.
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    mod._pending.clear()
    mod._answers.clear()
    mod._live_transports.clear()


def test_shared_fixture_cleanup_uses_full_session_teardown(server, monkeypatch):
    """The cross-file autouse cleanup must close every retained resource."""
    from tests import conftest

    closed = {"worker": 0, "agent": 0, "lease": 0}

    class _Closable:
        def __init__(self, key):
            self.key = key

        def close(self):
            closed[self.key] += 1

    class _Lease:
        def release(self):
            closed["lease"] += 1

    monkeypatch.setattr(server, "_get_db", lambda: None)
    server._sessions["leaked"] = {
        "session_key": "leaked",
        "agent": _Closable("agent"),
        "slash_worker": _Closable("worker"),
        "active_session_lease": _Lease(),
        "history": [],
    }

    conftest._teardown_tui_server_sessions(server)

    assert server._sessions == {}
    assert closed == {"worker": 1, "agent": 1, "lease": 1}


@pytest.fixture()
def capture(server):
    """Redirect server's real stdout to a StringIO and return (server, buf)."""
    buf = io.StringIO()
    server._real_stdout = buf
    return server, buf


# ── JSON-RPC envelope ────────────────────────────────────────────────


def test_unknown_method(server):
    resp = server.handle_request({"id": "1", "method": "bogus"})
    assert resp["error"]["code"] == -32601


def test_ok_envelope(server):
    assert server._ok("r1", {"x": 1}) == {
        "jsonrpc": "2.0", "id": "r1", "result": {"x": 1},
    }


def test_err_envelope(server):
    assert server._err("r2", 4001, "nope") == {
        "jsonrpc": "2.0", "id": "r2", "error": {"code": 4001, "message": "nope"},
    }


@pytest.mark.parametrize("kind", ["legacy", "hard-only", "dynamic-getattr"])
def test_session_interrupt_uses_explicit_stop_compatibility(server, monkeypatch, kind):
    calls = []

    class _Legacy:
        def interrupt(self):
            calls.append("legacy")

    class _HardOnly:
        def hard_interrupt(self):
            calls.append("hard")

    class _Dynamic:
        def interrupt(self):
            calls.append("legacy")

        def __getattr__(self, name):
            if name == "hard_interrupt":
                return lambda: calls.append("fabricated-hard")
            raise AttributeError(name)

    agent = {
        "legacy": _Legacy(),
        "hard-only": _HardOnly(),
        "dynamic-getattr": _Dynamic(),
    }[kind]
    session = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": True,
        "queued_prompt": "later",
        "session_key": "session-key",
        "_run_thread": None,
    }
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    response = server._methods["session.interrupt"](
        "stop", {"session_id": "ui-session"}
    )

    assert response["result"]["status"] == "interrupted"
    assert calls == ["hard" if kind == "hard-only" else "legacy"]


# ── write_json ────────────────────────────────────────────────


def test_write_json(capture):
    server, buf = capture
    assert server.write_json({"test": True})
    assert json.loads(buf.getvalue()) == {"test": True}


def test_live_session_payload_replays_pending_approval(server, monkeypatch):
    """A reattached client receives the approval that was emitted while detached."""
    from tools import approval

    session = {
        "agent": types.SimpleNamespace(),
        "cols": 80,
        "created_at": 1.0,
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-session",
    }
    first = {
        "choices": ["once", "deny"],
        "command": "rm -rf /tmp/example",
        "description": "recursive delete",
    }
    second = {"command": "rm -rf /tmp/later", "description": "later"}
    saved_queue = approval._gateway_queues.pop("stored-session", None)
    approval._gateway_queues["stored-session"] = [
        approval._ApprovalEntry(first),
        approval._ApprovalEntry(second),
    ]
    monkeypatch.setattr(server, "_approval_request_payload", lambda data: dict(data or {}))

    try:
        payload = server._live_session_payload("runtime-session", session)
    finally:
        approval._gateway_queues.pop("stored-session", None)
        if saved_queue is not None:
            approval._gateway_queues["stored-session"] = saved_queue

    assert payload["pending_approval"] is not first
    replayed = payload["pending_approval"]
    # request_id is injected by _ApprovalEntry so reconnecting clients can
    # correlate their approval.respond with the exact queued request.
    assert replayed.pop("request_id")
    assert replayed == first


def test_live_session_payload_replays_pending_clarify(server):
    """A reattached client also receives a clarify question emitted while detached."""
    session = {
        "agent": types.SimpleNamespace(),
        "cols": 80,
        "created_at": 1.0,
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-session",
    }
    clarify_payload = {
        "choices": ["staging", "production"],
        "question": "Which deployment target?",
        "request_id": "rid-clarify",
    }
    with server._prompt_lock:
        server._pending["rid-clarify"] = ("runtime-session", threading.Event())
        server._pending_prompt_payloads["rid-clarify"] = (
            "clarify.request",
            dict(clarify_payload),
        )

    try:
        payload = server._live_session_payload("runtime-session", session)
        other = server._live_session_payload("other-session", session)
    finally:
        with server._prompt_lock:
            server._pending.pop("rid-clarify", None)
            server._pending_prompt_payloads.pop("rid-clarify", None)

    assert payload["pending_clarify"] == clarify_payload
    # Snapshot, not a live reference into the registry.
    assert payload["pending_clarify"] is not clarify_payload
    # Scoped to the owning runtime session only.
    assert "pending_clarify" not in other


def test_disable_flush_env_var_actually_wires_to_module_constant(monkeypatch):
    """End-to-end: setting `HERMES_TUI_GATEWAY_NO_FLUSH=1` and importing
    `tui_gateway.transport` fresh actually flips `_DISABLE_FLUSH` true.

    Reloads only the transport module — server.py is untouched so its
    atexit hooks/worker pool stay intact."""
    import importlib

    monkeypatch.setenv("HERMES_TUI_GATEWAY_NO_FLUSH", "1")
    transport_mod = importlib.reload(importlib.import_module("tui_gateway.transport"))

    try:
        assert transport_mod._DISABLE_FLUSH is True
    finally:
        # Restore the env-disabled state so other tests see the default.
        monkeypatch.delenv("HERMES_TUI_GATEWAY_NO_FLUSH", raising=False)
        importlib.reload(transport_mod)


# ── _emit ────────────────────────────────────────────────────────────


def test_emit_with_payload(capture):
    server, buf = capture
    server._emit("test.event", "s1", {"key": "val"})
    msg = json.loads(buf.getvalue())

    assert msg["method"] == "event"
    assert msg["params"]["type"] == "test.event"
    assert msg["params"]["session_id"] == "s1"
    assert msg["params"]["payload"]["key"] == "val"


# ── Blocking prompt round-trip ───────────────────────────────────────


def test_block_and_respond(capture):
    server, _ = capture
    result = [None]

    threading.Thread(
        target=lambda: result.__setitem__(0, server._block("test.prompt", "s1", {"q": "?"}, timeout=5)),
    ).start()

    for _ in range(100):
        if server._pending:
            break
        threading.Event().wait(0.01)

    rid = next(iter(server._pending))
    server._answers[rid] = "my_answer"
    # _pending values are (sid, Event) tuples — unpack to set the Event
    _, ev = server._pending[rid]
    ev.set()

    threading.Event().wait(0.1)
    assert result[0] == "my_answer"


@pytest.mark.parametrize(
    "event",
    ["secret.request", "sudo.request", "clarify.request", "terminal.read.request"],
)
def test_sensitive_prompt_timeout_emits_expiry(capture, event):
    server, buf = capture

    assert server._block(event, "s1", {}, timeout=0) == ""

    messages = [json.loads(line) for line in buf.getvalue().splitlines()]
    request, expiry = [message["params"] for message in messages]
    assert request["type"] == event
    assert expiry["type"] == event.removesuffix(".request") + ".expire"
    assert expiry["session_id"] == "s1"
    assert expiry["payload"]["request_id"] == request["payload"]["request_id"]


@pytest.mark.parametrize(
    ("method", "value_key"),
    [
        ("secret.respond", "value"),
        ("sudo.respond", "password"),
        ("clarify.respond", "answer"),
        ("terminal.read.respond", "text"),
    ],
)
def test_late_prompt_response_is_idempotent(server, method, value_key):
    """All four blocking bridges tolerate a late reply after their request has
    expired — the `*.respond` returns a graceful `{"status": "expired"}` instead
    of the raw 4009 protocol error a client would otherwise surface verbatim."""
    response = server.handle_request(
        {
            "id": "late-response",
            "method": method,
            "params": {"request_id": "expired-request", value_key: ""},
        }
    )

    assert response["result"] == {"status": "expired"}


def test_approval_pending_replays_unresolved_requests(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    pending = [{"request_id": "req-1", "command": "danger"}]
    monkeypatch.setattr(approval, "list_gateway_approvals", lambda key: pending if key == "agent-1" else [])

    response = server.handle_request(
        {"id": "r1", "method": "approval.pending", "params": {"session_id": "ui-1"}}
    )

    assert response["result"] == {"approvals": pending}


def test_approval_received_acknowledges_exact_request(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    calls = []
    monkeypatch.setattr(
        approval,
        "ack_gateway_approval",
        lambda key, request_id: calls.append((key, request_id)) or True,
    )

    response = server.handle_request(
        {
            "id": "r2",
            "method": "approval.received",
            "params": {"session_id": "ui-1", "request_id": "req-1"},
        }
    )

    assert response["result"] == {"acknowledged": True}
    assert calls == [("agent-1", "req-1")]


def test_approval_response_correlates_request_id(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    calls = []
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda key, choice, **kwargs: calls.append((key, choice, kwargs)) or 1,
    )

    response = server.handle_request(
        {
            "id": "r3",
            "method": "approval.respond",
            "params": {"session_id": "ui-1", "request_id": "req-1", "choice": "once"},
        }
    )

    assert response["result"] == {"resolved": 1}
    assert calls == [("agent-1", "once", {"resolve_all": False, "request_id": "req-1"})]


def test_clear_pending(server):
    ev = threading.Event()
    # _pending values are (sid, Event) tuples
    server._pending["r1"] = ("sid-x", ev)
    server._clear_pending()

    assert ev.is_set()
    assert server._answers["r1"] == ""


# ── Session lookup ───────────────────────────────────────────────────


def test_sess_missing(server):
    _, err = server._sess({"session_id": "nope"}, "r1")
    assert err["error"]["code"] == 4001


# ── session.resume payload ────────────────────────────────────────────


def test_session_resume_returns_hydrated_messages(server, monkeypatch):
    class _DB:
        def get_session(self, _sid):
            return {"id": "20260409_010101_abc123"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, _sid, include_ancestors=False, repair_alternation=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo", "reasoning": "thoughts"},
                {"role": "tool", "content": "searched"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": None},
                {"role": "narrator", "content": "skip"},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None, **_kwargs: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            # eager_build: exercise the synchronous build path (this test
            # monkeypatches _make_agent/_init_session/_session_info).
            "params": {"session_id": "20260409_010101_abc123", "cols": 100, "eager_build": True},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 3
    assert resp["result"]["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "yo", "reasoning": "thoughts"},
        {"role": "tool", "name": "tool", "context": ""},
    ]


def test_session_resume_rejects_runaway_transcript_before_history_load(
    server, monkeypatch
):
    class _DB:
        def get_session(self, sid):
            return {"id": sid, "message_count": 20_001}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def reopen_session(self, _sid):
            raise AssertionError("oversized session must be rejected before reopen")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    response = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {
                "session_id": "runaway-session",
                "omit_messages": True,
            },
        }
    )

    assert response["error"]["code"] == 4130
    assert "safe resume limit is 20000" in response["error"]["message"]


def test_session_resume_guard_failure_fails_open(server, monkeypatch):
    """A transient guard error must not block resume (fail open, log only)."""
    reopened = []

    class _DB:
        def get_session(self, sid):
            return {"id": sid}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def assert_resume_safe(self, _sid):
            raise RuntimeError("database is locked")

        def reopen_session(self, sid):
            reopened.append(sid)
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    response = server.handle_request(
        {
            "id": "r-open",
            "method": "session.resume",
            "params": {
                "session_id": "transient-guard-session",
                "omit_messages": True,
            },
        }
    )

    # The guard must not block: no 4130, and any downstream failure must not
    # be the guard's own "resume safety check failed" error. Reopen being
    # attempted proves execution moved past the guard.
    err = response.get("error") or {}
    assert err.get("code") != 4130
    assert "resume safety check failed" not in str(err.get("message", ""))
    assert reopened == ["transient-guard-session"]


def test_session_resume_active_turn_payload_matches_desktop_fixture(server, monkeypatch):
    """A live resume serializes the exact timer payload consumed by Desktop."""
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "session-resume-active-turn.json").read_text(
            encoding="utf-8"
        )
    )

    class _DB:
        def get_session(self, session_id):
            return {"id": session_id}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

    active_turn = {
        "assistant": "partial answer",
        "started_at": fixture["turn_started_at"],
        "streaming": True,
        "user": "current prompt",
    }
    server._sessions[fixture["session_id"]] = {
        "agent": types.SimpleNamespace(session_id=fixture["session_key"]),
        "created_at": fixture["started_at"],
        "history": [{"content": "earlier prompt", "role": "user"}],
        "history_lock": threading.Lock(),
        "inflight_turn": active_turn,
        "running": True,
        "session_key": fixture["session_key"],
    }
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_session_info", lambda _agent: fixture["info"])

    # JSON round-trip the real RPC envelope: the desktop fixture must stay
    # faithful to what the gateway actually serializes, not a copied shape.
    response = json.loads(
        json.dumps(
            server.handle_request(
                {
                    "id": "resume-running",
                    "method": "session.resume",
                    "params": {"session_id": fixture["session_key"]},
                }
            )
        )
    )
    result = response["result"]

    assert result["running"] is True
    assert result["turn_started_at"] == active_turn["started_at"]
    assert result == fixture


def test_enforce_session_cap_evicts_oldest_detached_only(server, monkeypatch):
    """The LRU cap frees the least-recently-active DETACHED sessions when over
    the limit, and never a live-transport / running / mid-build one."""

    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_live_sessions": 2})
    evicted: list[str] = []
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, end_reason=None, predicate=None: evicted.append(sid),
    )

    def _ready() -> threading.Event:
        ev = threading.Event()
        ev.set()
        return ev

    detached = server._detached_ws_transport
    live = object()  # no _closed attr -> live transport, never evictable

    server._sessions.clear()
    server._sessions.update(
        {
            "old_detached": {"transport": detached, "last_active": 100.0, "agent_ready": _ready()},
            "new_detached": {"transport": detached, "last_active": 300.0, "agent_ready": _ready()},
            "running_detached": {
                "transport": detached,
                "last_active": 50.0,
                "running": True,
                "agent_ready": _ready(),
            },
            "focused_live": {"transport": live, "last_active": 200.0, "agent_ready": _ready()},
        }
    )

    server._enforce_session_cap()

    # 4 sessions, cap 2 -> evict 2. Only detached+idle+built are eligible, oldest
    # first; the running one and the live-transport one are exempt.
    assert evicted == ["old_detached", "new_detached"]


def test_sync_session_key_after_compress_reanchors_active_session_lease(
    server, monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.active_sessions import (
        active_session_registry_snapshot,
        try_acquire_active_session,
    )

    lease, message = try_acquire_active_session(
        session_id="session-old",
        surface="tui",
        config={"max_concurrent_sessions": 1},
        metadata={"live_session_id": "ui-1"},
    )
    assert message is None
    assert lease is not None

    session = {
        "active_session_lease": lease,
        "agent": types.SimpleNamespace(session_id="session-new"),
        "session_key": "session-old",
    }
    fake_approval = types.SimpleNamespace(
        disable_session_yolo=lambda *_args, **_kwargs: None,
        enable_session_yolo=lambda *_args, **_kwargs: None,
        is_session_yolo_enabled=lambda *_args, **_kwargs: False,
        register_gateway_notify=lambda *_args, **_kwargs: None,
        unregister_gateway_notify=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args, **_kwargs: None)

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        server._sync_session_key_after_compress("ui-1", session)

    snapshot = active_session_registry_snapshot()
    assert session["session_key"] == "session-new"
    assert lease.session_id == "session-new"
    assert [entry["session_id"] for entry in snapshot] == ["session-new"]
    lease.release()


def test_make_agent_accepts_list_system_prompt(server, monkeypatch):
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model", "")

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_Agent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "test",
                "base_url": None,
                "api_key": None,
                "api_mode": None,
            }
        ),
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"system_prompt": ["one", "two"]}})
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("test/model", "test"))
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "session-key", session_id="session-key")

    assert captured["ephemeral_system_prompt"] == "one\ntwo"


# ── Config I/O ───────────────────────────────────────────────────────


def test_config_roundtrip(server, tmp_path):
    server._hermes_home = tmp_path
    server._save_cfg({"model": "test/model"})
    assert server._load_cfg()["model"] == "test/model"


# ── _cli_exec_blocked ────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [
    [],
    ["setup"],
    ["gateway"],
    ["sessions", "browse"],
    ["config", "edit"],
])
def test_cli_exec_blocked(server, argv):
    assert server._cli_exec_blocked(argv) is not None


# ── slash.exec skill command interception ────────────────────────────


def test_slash_exec_rejects_skill_commands(server):
    """slash.exec must reject skill commands so the TUI falls through to command.dispatch."""
    # Register a mock session
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    # Mock scan_skill_commands to return a known skill
    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}

    with patch("agent.skill_commands.get_skill_commands", return_value=fake_skills):
        resp = server.handle_request({
            "id": "r1",
            "method": "slash.exec",
            "params": {"command": "hermes-agent-dev", "session_id": sid},
        })

    # Should return an error so the TUI's .catch() fires command.dispatch
    assert "error" in resp
    assert resp["error"]["code"] == 4018
    assert "skill command" in resp["error"]["message"]


def test_command_dispatch_queue_sends_message(server):
    """command.dispatch /queue returns {type: 'send', message: ...} for the TUI."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    resp = server.handle_request({
        "id": "r1",
        "method": "command.dispatch",
        "params": {"name": "queue", "arg": "tell me about quantum computing", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "tell me about quantum computing"


def test_skills_manage_search_uses_tools_hub_sources(server):
    result = type("Result", (), {
        "description": "Build better terminal demos",
        "name": "showroom",
    })()
    auth = MagicMock(return_value="auth")
    router = MagicMock(return_value=["source"])
    search = MagicMock(return_value=[result])
    fake_hub = types.SimpleNamespace(
        GitHubAuth=auth,
        create_source_router=router,
        unified_search=search,
    )

    with patch.dict(sys.modules, {"tools.skills_hub": fake_hub}):
        resp = server.handle_request({
            "id": "skills-search",
            "method": "skills.manage",
            "params": {"action": "search", "query": "showroom"},
        })

    assert "error" not in resp
    assert resp["result"] == {
        "results": [{"description": "Build better terminal demos", "name": "showroom"}]
    }
    auth.assert_called_once_with()
    router.assert_called_once_with("auth")
    search.assert_called_once_with("showroom", ["source"], source_filter="all", limit=20)


# ── dispatch(): pool routing for long handlers (#12546) ──────────────


def test_dispatch_runs_short_handlers_inline(server):
    """Non-long handlers return their response synchronously from dispatch()."""
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    resp = server.dispatch({"id": "r1", "method": "fast.ping", "params": {}})

    assert resp == {"jsonrpc": "2.0", "id": "r1", "result": {"pong": True}}


@pytest.mark.parametrize("completion_method", ["complete.path", "complete.slash"])
def test_completion_handlers_are_pool_routed(completion_method, server):
    """complete.path/complete.slash must run on the pool, never the reader thread.

    Regression for #21123: completion ran inline, so a slow git ls-files /
    skill-scan blocked prompt.submit and froze the TUI for the 120s RPC timeout.
    """
    assert completion_method in server._LONG_HANDLERS


@pytest.mark.parametrize(
    "voice_or_wake_method",
    ["voice.toggle", "voice.record", "voice.tts", "wake.start", "wake.status"],
)
def test_voice_and_wake_handlers_are_pool_routed(voice_or_wake_method, server):
    """Voice and wake RPCs must run on the pool, never the WS reader thread.

    Regression: voice.toggle (status) triggers check_voice_requirements() →
    STT provider auto-detect → a SYNCHRONOUS faster-whisper lazy install (uv/pip
    subprocess, up to a 300s timeout). Inline on the WS reader loop it blocked
    prompt.submit / session.list frames queued behind it — the desktop showed
    sent messages that never reached the agent. Same bug class as #21123 /
    #50005: anything that can stall for seconds must stay off the reader thread.

    wake.start and wake.status share the same STT lazy-install path via
    check_wake_word_requirements() → _stt_ready() → _get_provider(), and
    wake.start additionally calls lazy_deps.ensure() for wake-word engine deps.
    The desktop polls wake.status on every gateway-ready.
    """
    assert voice_or_wake_method in server._LONG_HANDLERS


def test_skin_live_switch_end_to_end(server, tmp_path, monkeypatch):
    """Real config + skin files: activating a skin (as `hermes config set` does)
    makes the per-tool reconcile broadcast skin.changed with the resolved palette.
    Exercises _load_cfg → _skin_sig → resolve_skin → _emit with no mocks in between."""
    import hermes_cli.skin_engine as skin_engine

    (tmp_path / "skins").mkdir()
    (tmp_path / "skins" / "midnight.yaml").write_text(
        "name: midnight\ndescription: t\ncolors:\n  banner_title: '#00ffcc'\n  background: '#001010'\n"
    )
    monkeypatch.setattr(skin_engine, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_last_skin_sig", None, raising=False)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, payload)))

    # Baseline (default) — seeds the signature.
    (tmp_path / "config.yaml").write_text("display:\n  skin: default\n", encoding="utf-8")
    server._broadcast_skin_if_changed()
    emitted.clear()

    # Activate midnight, as `hermes config set display.skin midnight` would.
    time.sleep(0.01)  # ensure the config mtime moves
    (tmp_path / "config.yaml").write_text("display:\n  skin: midnight\n", encoding="utf-8")
    server._broadcast_skin_if_changed()

    assert [ev for ev, _ in emitted] == ["skin.changed"]
    assert emitted[0][1]["name"] == "midnight"
    assert emitted[0][1]["colors"]["banner_title"] == "#00ffcc"


def test_broadcast_skin_if_changed_on_any_signature_move(server, monkeypatch):
    """A skin the agent changes mid-turn goes live once per real move: a name
    switch (incl. switch-then-revert) OR an in-place color edit to the active skin
    (same name, new file mtime). An unchanged signature never re-broadcasts."""
    emitted = []
    # switch, no-op, switch, then a color edit (same name, bumped mtime).
    sigs = iter([("neon", 1.0), ("neon", 1.0), ("forest", 1.0), ("forest", 2.0)])
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, payload)))
    monkeypatch.setattr(server, "_last_skin_sig", None, raising=False)
    monkeypatch.setattr(server, "_skin_sig", lambda: next(sigs))
    monkeypatch.setattr(server, "resolve_skin", lambda: {"name": "x", "colors": {}})

    for _ in range(4):
        server._broadcast_skin_if_changed()

    assert [ev for ev, _ in emitted] == ["skin.changed"] * 3


# ── global-event broadcast (session-less events reach every WS client) ──


class _RecordingTransport:
    """Minimal Transport stand-in that records the frames written to it."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return True

    def close(self) -> None:
        pass


def test_unregister_live_transport_stops_delivery(capture):
    """A disconnected peer (unregistered in the ws finally block) receives nothing
    — and a stale write is never attempted against its closed socket."""
    server, buf = capture
    a = _RecordingTransport()
    server.register_live_transport(a)
    server.unregister_live_transport(a)

    server._broadcast_global_event("skin.changed", {"name": "x"})

    assert a.frames == []
    # No live transports left → fell back to stdio.
    assert json.loads(buf.getvalue())["params"]["type"] == "skin.changed"
