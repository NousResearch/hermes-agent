"""Tests for tui_gateway JSON-RPC protocol plumbing."""

import io
import json
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway.session_state import SessionState

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        # Snapshot module globals that tests in this file assign directly
        # (handler-registry entries, stdout redirection, config paths) so
        # each test — and any later test file sharing this already-imported
        # module — starts from a clean slate.
        saved_methods = dict(mod._methods)
        saved_attrs = {
            name: getattr(mod, name)
            for name in ("_real_stdout", "_hermes_home", "_cfg_cache", "_cfg_mtime", "_cfg_path")
        }
        yield mod
        # Reset module-level session state without re-importing. importlib.reload
        # would re-register the module's atexit hooks (ThreadPoolExecutor
        # shutdown, _shutdown_sessions); the duplicates race the stderr
        # buffer at interpreter shutdown and surface as Fatal Python error:
        # _enter_buffered_busy. Clearing the per-session dicts gives the
        # next test a clean slate; _methods is restored in place (same dict
        # object) from the snapshot taken above because dispatch tests
        # overwrite entries like "slash.exec" and "session.compress", and
        # re-registration only happens via reload (which we don't do).
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        mod._methods.update(saved_methods)
        for name, value in saved_attrs.items():
            setattr(mod, name, value)


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


# ── write_json ───────────────────────────────────────────────────────


def test_write_json(capture):
    server, buf = capture
    assert server.write_json({"test": True})
    assert json.loads(buf.getvalue()) == {"test": True}


def test_write_json_broken_pipe(server):
    class _Broken:
        def write(self, _): raise BrokenPipeError
        def flush(self): raise BrokenPipeError

    server._real_stdout = _Broken()
    assert server.write_json({"x": 1}) is False


def test_write_json_closed_stream_returns_false(server):
    """ValueError ('I/O on closed file') used to bubble up; treat as gone."""

    class _Closed:
        def write(self, _): raise ValueError("I/O operation on closed file")
        def flush(self): raise ValueError("I/O operation on closed file")

    server._real_stdout = _Closed()
    assert server.write_json({"x": 1}) is False


def test_write_json_unicode_encode_error_re_raises(server):
    """A non-UTF-8 stdout encoding raises UnicodeEncodeError (a ValueError
    subclass).  It must NOT be swallowed as 'peer gone' — that would let
    `entry.py` exit cleanly via the False path and hide the real config
    bug.  We re-raise so the existing crash-log infrastructure records it."""

    class _AsciiOnly:
        def write(self, line):
            line.encode("ascii")  # raises UnicodeEncodeError on non-ascii
        def flush(self): pass

    server._real_stdout = _AsciiOnly()
    with pytest.raises(UnicodeEncodeError):
        server.write_json({"msg": "héllo"})


def test_write_json_unrelated_value_error_re_raises(server):
    """Only ValueError('...closed file...') means peer gone.  Other
    ValueErrors are programming errors and must surface."""

    class _BadValue:
        def write(self, _): raise ValueError("something else entirely")
        def flush(self): pass

    server._real_stdout = _BadValue()
    with pytest.raises(ValueError, match="something else entirely"):
        server.write_json({"x": 1})


def test_write_json_non_serializable_payload_re_raises(server):
    """Non-JSON-safe payloads are programming errors — they must NOT be
    silently dropped via the False path (which would trigger a clean exit
    in entry.py and mask the real bug)."""
    import io

    server._real_stdout = io.StringIO()
    with pytest.raises(TypeError):
        server.write_json({"obj": object()})


def test_write_json_peer_gone_oserror_on_flush_returns_false(server):
    """A flush that raises a peer-gone OSError (EPIPE) must not strand
    the lock or crash; it returns False so the dispatcher exits cleanly."""
    import errno

    written = []

    class _FlushPeerGone:
        def write(self, line): written.append(line)
        def flush(self): raise OSError(errno.EPIPE, "broken pipe")

    server._real_stdout = _FlushPeerGone()
    assert server.write_json({"x": 1}) is False
    assert written and json.loads(written[0]) == {"x": 1}


def test_write_json_non_peer_gone_oserror_re_raises(server):
    """Host I/O failures (ENOSPC, EACCES, EIO …) are NOT peer-gone — they
    must re-raise so the crash log records them instead of looking like
    a clean disconnect via the False path."""
    import errno

    class _DiskFull:
        def write(self, _): raise OSError(errno.ENOSPC, "no space left")
        def flush(self): pass

    server._real_stdout = _DiskFull()
    with pytest.raises(OSError, match="no space"):
        server.write_json({"x": 1})


def test_write_json_skips_flush_when_disable_flush_true(monkeypatch):
    """`StdioTransport` skips flush when `_DISABLE_FLUSH` is true.

    Tests the runtime *behaviour* via direct module-attr patch.  The env
    var → module constant wiring is covered by the dedicated env test
    below; reloading server.py here would re-register atexit hooks and
    recreate the worker pool.
    """
    import importlib

    transport_mod = importlib.import_module("tui_gateway.transport")
    monkeypatch.setattr(transport_mod, "_DISABLE_FLUSH", True)

    flushed = {"count": 0}
    written = []

    class _Stream:
        def write(self, line): written.append(line)
        def flush(self): flushed["count"] += 1

    stream = _Stream()
    transport = transport_mod.StdioTransport(lambda: stream, threading.Lock())

    assert transport.write({"x": 1}) is True
    assert flushed["count"] == 0


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


def test_emit_without_payload(capture):
    server, buf = capture
    server._emit("ping", "s2")

    assert "payload" not in json.loads(buf.getvalue())["params"]


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


def test_sess_found(server):
    server._sessions["abc"] = {"agent": MagicMock()}
    s, err = server._sess({"session_id": "abc"}, "r1")

    assert s is not None
    assert err is None


# ── session.resume payload ────────────────────────────────────────────


def test_session_resume_returns_hydrated_messages(server, monkeypatch):
    class _DB:
        def get_session(self, _sid):
            return {"id": "20260409_010101_abc123"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo", "reasoning": "thoughts"},
                {"role": "tool", "content": "searched"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": None},
                {"role": "narrator", "content": "skip"},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": "20260409_010101_abc123", "cols": 100},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 3
    assert resp["result"]["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "yo", "reasoning": "thoughts"},
        {"role": "tool", "name": "tool", "context": ""},
    ]


def test_session_resume_handles_multimodal_list_content(server, monkeypatch):
    """A user message persisted with list-shaped multimodal content used to
    crash session resume with ``'list' object has no attribute 'strip'``."""

    multimodal_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ],
    }
    text_only_assistant = {"role": "assistant", "content": "ok"}

    class _DB:
        def get_session(self, _sid):
            return {"id": "20260502_000000_listcontent"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [multimodal_user, text_only_assistant]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {"session_id": "20260502_000000_listcontent", "cols": 100},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 2
    # The image_url part is preserved as a raw data URL inside the text so
    # the desktop renderer (which extracts embedded images) sees the same
    # content the optimistic local cache returns. Otherwise the inline
    # image flashes during initial cache hydration and then vanishes when
    # the resume payload overwrites it with cleaned text.
    assert resp["result"]["messages"] == [
        {
            "role": "user",
            "text": "describe this\ndata:image/png;base64,AAAA",
        },
        {"role": "assistant", "text": "ok"},
    ]


def test_session_resume_reuses_existing_live_session(server, monkeypatch):
    """Repeated resume must not allocate duplicate live agents."""

    target = "20260409_010101_abc123"
    created_sids: list[str] = []
    closed_sids: list[str] = []
    first_agent_started = threading.Event()
    agent_can_finish = threading.Event()

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo"},
            ]

    class _Worker:
        def close(self):
            pass

    class _Agent:
        def __init__(self, sid, session_id):
            self.sid = sid
            self.model = "test/model"
            self.session_id = session_id

        def close(self):
            closed_sids.append(self.sid)

    def make_agent(sid, key, session_id=None, session_db=None):
        created_sids.append(sid)
        first_agent_started.set()
        assert agent_can_finish.wait(timeout=1)
        return _Agent(sid, session_id or key)

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_SlashWorker", lambda _key, _model: _Worker())
    monkeypatch.setattr(
        server,
        "_start_notification_poller",
        lambda _sid, _session: threading.Event(),
    )
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    fake_approval = types.SimpleNamespace(
        load_permanent_allowlist=lambda: None,
        register_gateway_notify=lambda *_args, **_kwargs: None,
    )

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        first_holder = {}

        def resume_first():
            first_holder["resp"] = server.handle_request(
                {
                    "id": "first",
                    "method": "session.resume",
                    "params": {"session_id": target, "cols": 100},
                }
            )

        first_thread = threading.Thread(target=resume_first)
        first_thread.start()
        assert first_agent_started.wait(timeout=1)

        second_holder = {}

        def resume_second():
            second_holder["resp"] = server.handle_request(
                {
                    "id": "second",
                    "method": "session.resume",
                    "params": {"session_id": target, "cols": 120},
                }
            )

        second_thread = threading.Thread(target=resume_second)
        second_thread.start()
        agent_can_finish.set()

        first_thread.join(timeout=1)
        second_thread.join(timeout=1)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        first = first_holder["resp"]
        second = second_holder["resp"]

    assert "error" not in first
    assert "error" not in second
    # Both resumes resolve to the SAME single live session — the core invariant.
    assert second["result"]["session_id"] == first["result"]["session_id"]
    assert len(server._sessions) == 1
    assert [s.get("session_key") for s in server._sessions.values()].count(target) == 1
    winner = first["result"]["session_id"]
    # The agent build happens outside the resume lock, so a racing resume may
    # build a redundant agent; double-checked locking keeps only one live
    # session and closes any loser's agent (no worker/poller is wired for it).
    assert winner in created_sids
    survivors = [sid for sid in created_sids if sid not in closed_sids]
    assert survivors == [winner]
    assert all(sid == winner for sid in server._sessions)


def test_session_resume_live_payload_uses_current_history_with_ancestors(server, monkeypatch):
    """Live resume should not reuse a stale ancestor-inclusive snapshot."""

    target = "20260409_010101_child"
    ancestor_history = [{"role": "user", "content": "ancestor"}]
    current_history = [
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "current reply"},
    ]

    class _DB:
        def get_session(self, _sid):
            return {"id": target}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_messages_as_conversation(self, _sid, include_ancestors=False):
            if include_ancestors:
                return ancestor_history + current_history
            return list(current_history)

    class _Worker:
        def close(self):
            pass

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda _sid, key, session_id=None, session_db=None: types.SimpleNamespace(
            model="test/model", session_id=session_id or key
        ),
    )
    monkeypatch.setattr(server, "_SlashWorker", lambda _key, _model: _Worker())
    monkeypatch.setattr(
        server,
        "_start_notification_poller",
        lambda _sid, _session: threading.Event(),
    )
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    fake_approval = types.SimpleNamespace(
        load_permanent_allowlist=lambda: None,
        register_gateway_notify=lambda *_args, **_kwargs: None,
    )

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        first = server.handle_request(
            {
                "id": "first",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 100},
            }
        )

        assert "error" not in first
        sid = first["result"]["session_id"]
        assert first["result"]["messages"] == [
            {"role": "user", "text": "ancestor"},
            {"role": "user", "text": "current"},
            {"role": "assistant", "text": "current reply"},
        ]

        with server._sessions[sid]["history_lock"]:
            server._sessions[sid]["history"] = current_history + [
                {"role": "user", "content": "new live turn"},
                {"role": "assistant", "content": "new live reply"},
            ]

        second = server.handle_request(
            {
                "id": "second",
                "method": "session.resume",
                "params": {"session_id": target, "cols": 120},
            }
        )

    assert "error" not in second
    assert second["result"]["session_id"] == sid
    assert second["result"]["messages"] == [
        {"role": "user", "text": "ancestor"},
        {"role": "user", "text": "current"},
        {"role": "assistant", "text": "current reply"},
        {"role": "user", "text": "new live turn"},
        {"role": "assistant", "text": "new live reply"},
    ]


def test_session_activate_rebinds_orphaned_ws_session_to_current_transport(server, monkeypatch):
    """Reconnect + activate must reattach a parked live session before orphan reap."""

    class _Transport:
        def write(self, _obj):
            return True

    sid = "runtime01"
    old_transport = server._stdio_transport
    new_transport = _Transport()
    server._sessions[sid] = SessionState({
        "agent": types.SimpleNamespace(model="test/model"),
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "running": False,
        "session_key": "20260409_010101_abc123",
        "transport": old_transport,
    })
    monkeypatch.setattr(server, "current_transport", lambda: new_transport)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "test/model"},
    )

    resp = server.handle_request(
        {"id": "activate", "method": "session.activate", "params": {"session_id": sid}}
    )

    assert "error" not in resp
    assert resp["result"]["session_id"] == sid
    assert server._sessions[sid]["transport"] is new_transport
    assert not server._ws_session_is_orphaned(server._sessions[sid])


def test_session_branch_persists_branched_from_marker(server, monkeypatch):
    """TUI /branch must persist a _branched_from marker so the branch stays
    visible in /resume and /sessions.

    Regression for issue #20856: the TUI branch leaves the parent live (it
    never ends it with end_reason='branched'), so list_sessions_rich's legacy
    heuristic never surfaces it — the stable model_config marker is the only
    thing that keeps a TUI branch visible.
    """
    create_calls = []

    class _DB:
        def get_session_title(self, _key):
            return "parent-title"

        def get_next_title_in_lineage(self, base):
            return f"{base} 2"

        def create_session(self, new_key, **kwargs):
            create_calls.append((new_key, kwargs))
            return new_key

        def append_message(self, **_kwargs):
            return None

        def set_session_title(self, _key, _title):
            return None

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_new_session_key", lambda: "20260101_000001_child0")
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda _sid, key, session_id=None, session_db=None: types.SimpleNamespace(
            model="test/model", session_id=session_id or key
        ),
    )
    monkeypatch.setattr(server, "_init_session", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _s: "/tmp/branch-cwd")

    parent_sid = "parent01"
    parent_key = "20260101_000000_parent"
    server._sessions[parent_sid] = SessionState({
        "session_key": parent_key,
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.Lock(),
        "cols": 80,
    })

    resp = server.handle_request(
        {"id": "b1", "method": "session.branch", "params": {"session_id": parent_sid}}
    )

    assert "error" not in resp, resp
    assert len(create_calls) == 1
    new_key, kwargs = create_calls[0]
    assert new_key == "20260101_000001_child0"
    assert kwargs["parent_session_id"] == parent_key
    # The marker — without it the branch is invisible in /resume and /sessions.
    assert kwargs["model_config"] == {"_branched_from": parent_key}


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


def test_config_load_missing(server, tmp_path):
    server._hermes_home = tmp_path
    assert server._load_cfg() == {}


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


@pytest.mark.parametrize("argv", [
    ["version"],
    ["sessions", "list"],
])
def test_cli_exec_allowed(server, argv):
    assert server._cli_exec_blocked(argv) is None


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


def test_slash_exec_handles_plugin_commands_in_live_gateway(server):
    """Plugin slash commands return normal slash.exec output without using the worker."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: (lambda arg: f"plugin:{arg}") if name == "plugin-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin-slash",
            "method": "slash.exec",
            "params": {"command": "plugin-cmd hello", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"] == {"output": "plugin:hello"}
    assert worker.calls == []


def test_slash_exec_plugin_lookup_failure_falls_back_to_worker(server):
    """Plugin discovery failures must not break ordinary slash-worker commands."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        side_effect=RuntimeError("discovery boom"),
    ):
        resp = server.handle_request({
            "id": "r-plugin-lookup-failure",
            "method": "slash.exec",
            "params": {"command": "help", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"] == {"output": "worker:help"}
    assert worker.calls == ["help"]


def test_slash_exec_plugin_handler_error_returns_output(server):
    """Plugin handler failures keep the _ok envelope + output (so the TUI does
    not redispatch) AND additively carry result.error so programmatic clients
    can detect the failure without string-matching the output prose."""
    sid = "test-session"

    class Worker:
        def __init__(self):
            self.calls = []

        def run(self, cmd):
            self.calls.append(cmd)
            return f"worker:{cmd}"

    def handler(arg):
        raise RuntimeError(f"handler boom: {arg}")

    worker = Worker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: handler if name == "plugin-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin-handler-error",
            "method": "slash.exec",
            "params": {"command": "plugin-cmd hello", "session_id": sid},
        })

    # Envelope stays _ok (TUI keeps its pager path, no redispatch). The failure
    # is now ADDITIVELY carried in result.error alongside the preserved output,
    # so a programmatic client can detect it without parsing the prose.
    assert "error" not in resp
    assert resp["result"] == {
        "output": "Plugin command error: handler boom: hello",
        "error": "Plugin command error: handler boom: hello",
    }
    assert worker.calls == []


# ── shell.exec / cli.exec honest-status regression coverage ──────────
# These paths are already honest (a failed command is detectable via the real
# exit ``code``, never coerced to success); lock that in so it cannot regress.


def test_shell_exec_success_reports_zero_code_and_stdout(server):
    """A successful shell command returns code 0 with its stdout."""
    resp = server.handle_request({
        "id": "r-shell-ok",
        "method": "shell.exec",
        "params": {"command": "printf hi"},
    })
    assert "error" not in resp
    assert resp["result"]["code"] == 0
    assert resp["result"]["stdout"] == "hi"


def test_shell_exec_nonzero_exit_reports_real_code(server):
    """A failed shell command surfaces its real exit code — not coerced to 0 —
    so a client (ui-tui useSubmission reads result.code) can detect failure."""
    resp = server.handle_request({
        "id": "r-shell-fail",
        "method": "shell.exec",
        "params": {"command": "exit 3"},
    })
    assert "error" not in resp
    assert resp["result"]["code"] == 3


def test_shell_exec_empty_command_errors(server):
    """An empty shell command is a structured JSON-RPC error, not a silent ok."""
    resp = server.handle_request({
        "id": "r-shell-empty",
        "method": "shell.exec",
        "params": {"command": ""},
    })
    assert resp["error"]["code"] == 4004


def test_cli_exec_nonzero_exit_reports_real_code(server, monkeypatch):
    """cli.exec surfaces a failed hermes_cli subprocess's real exit code on
    result.code (with stderr in output), so failure stays detectable."""
    fake = types.SimpleNamespace(returncode=2, stdout="", stderr="boom")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake)
    resp = server.handle_request({
        "id": "r-cli-fail",
        "method": "cli.exec",
        "params": {"argv": ["version"]},
    })
    assert "error" not in resp
    assert resp["result"]["blocked"] is False
    assert resp["result"]["code"] == 2
    assert "boom" in resp["result"]["output"]


def test_cli_exec_blocked_argv_reports_blocked_payload(server):
    """An interactive/blocked argv never spawns a subprocess and is flagged."""
    resp = server.handle_request({
        "id": "r-cli-blocked",
        "method": "cli.exec",
        "params": {"argv": ["setup"]},
    })
    assert "error" not in resp
    assert resp["result"]["blocked"] is True
    assert resp["result"]["code"] == -1
    assert resp["result"]["hint"]


def test_slash_exec_model_switch_failure_sets_error_field(server):
    """A backend-rejected /model switch still resolves _ok (so the TUI keeps
    its pager path) but flags a structured `error` so the web model picker can
    surface the failure instead of closing as though the switch worked.
    `warning` stays populated so existing slash.exec consumers are unaffected.
    """
    sid = "test-session"

    class Worker:
        def run(self, cmd):
            return "  ✗ Model 'bad/model' not available"

    server._sessions[sid] = {
        "session_key": sid,
        "agent": types.SimpleNamespace(model="x"),
        "slash_worker": Worker(),
        "running": False,
    }

    def _raise(_sid, _session, _arg):
        raise ValueError("model switch failed: 'bad/model' unavailable")

    with patch.object(server, "_apply_model_switch", _raise):
        resp = server.handle_request({
            "id": "r-model-fail",
            "method": "slash.exec",
            "params": {
                "command": "model bad/model --provider anthropic",
                "session_id": sid,
            },
        })

    # Envelope is still _ok — the failure is carried inside `result.error`.
    assert "error" not in resp
    result = resp["result"]
    assert result["error"].startswith("live session sync failed: ")
    assert "model switch failed" in result["error"]
    # Preserved for the TUI / web slashExec, which read output/warning only.
    assert result["warning"] == result["error"]
    assert result["output"] == "  ✗ Model 'bad/model' not available"


def test_slash_exec_model_switch_benign_warning_has_no_error(server):
    """A *successful* switch may still carry an advisory (auto-correction or a
    non-recommended-model note). That must surface as `warning`, never `error`,
    so the picker closes + toasts instead of treating it as a failed switch."""
    sid = "test-session"

    class Worker:
        def run(self, cmd):
            return "  ✓ Model switched: ok/model"

    server._sessions[sid] = {
        "session_key": sid,
        "agent": types.SimpleNamespace(model="x"),
        "slash_worker": Worker(),
        "running": False,
    }

    def _benign(_sid, _session, _arg):
        return {"value": "ok/model", "warning": "model not in catalog — proceeding"}

    with patch.object(server, "_apply_model_switch", _benign):
        resp = server.handle_request({
            "id": "r-model-benign",
            "method": "slash.exec",
            "params": {
                "command": "model ok/model --provider anthropic",
                "session_id": sid,
            },
        })

    assert "error" not in resp
    result = resp["result"]
    assert result["warning"] == "model not in catalog — proceeding"
    assert "error" not in result
    assert result["output"] == "  ✓ Model switched: ok/model"


@pytest.mark.parametrize("cmd", ["retry", "queue hello", "q hello", "steer fix the test", "plan"])
def test_slash_exec_rejects_pending_input_commands(server, cmd):
    """slash.exec must reject commands that use _pending_input in the CLI."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    resp = server.handle_request({
        "id": "r1",
        "method": "slash.exec",
        "params": {"command": cmd, "session_id": sid},
    })

    assert "error" in resp
    assert resp["error"]["code"] == 4018
    assert "pending-input command" in resp["error"]["message"]


# ── slash.exec task_id echo + task.submit (Phase 4) ──────────────────


class _FakeSlashWorker:
    """Worker double matching the slash.exec test style above."""

    def __init__(self):
        self.calls = []

    def run(self, cmd):
        self.calls.append(cmd)
        return f"worker:{cmd}"


def _clear_task_results(server):
    import action_runtime.task_registry as task_registry_mod

    # Legacy module dict — retained for name-compat only; no production path
    # reads it anymore.
    with server._TASK_RESULTS_LOCK:
        server._TASK_RESULTS.clear()
    # The replay store now lives in the AgentTaskRegistry singleton (Phase 5
    # Step 5 fold) — reset it the same way the registry fixture does, so
    # idempotency tests can never replay a key cached by an earlier test.
    task_registry_mod._instance = None


def test_slash_exec_task_id_param_is_echoed_additively(server):
    """slash.exec accepts an optional task_id and echoes it on the wire —
    everything else in the payload stays exactly as today."""
    sid = "test-session"
    worker = _FakeSlashWorker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}

    resp = server.handle_request({
        "id": "r-slash-task-id",
        "method": "slash.exec",
        "params": {"command": "help", "session_id": sid, "task_id": "task-echo-1"},
    })

    assert "error" not in resp
    assert resp["result"] == {"output": "worker:help", "task_id": "task-echo-1"}
    assert worker.calls == ["help"]


def test_task_submit_slash_happy_path_returns_rich_result(server):
    """task.submit(intent=slash) runs the same execution core as slash.exec but
    returns the rich ExecutionResult wire shape, echoing the supplied task_id."""
    sid = "test-session"
    worker = _FakeSlashWorker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}
    _clear_task_results(server)

    resp = server.handle_request({
        "id": "r-task-submit-ok",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "slash",
            "inputs": {"command": "help"},
            "task_id": "task-123",
        },
    })

    assert "error" not in resp
    assert resp["result"] == {
        "task_id": "task-123",
        "status": "succeeded",
        "outputs": {"output": "worker:help"},
        "error": None,
        "side_effects": [],
    }
    assert worker.calls == ["help"]


def test_task_submit_idempotency_replay_does_not_rerun(server):
    """Re-submitting the SAME idempotency_key must NOT execute again: the
    stored rich result comes back with an additive replayed=true marker."""
    sid = "test-session"
    worker = _FakeSlashWorker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}
    _clear_task_results(server)

    params = {
        "session_id": sid,
        "intent": "slash",
        "inputs": {"command": "help"},
        "idempotency_key": "idem-key-1",
    }
    resp1 = server.handle_request({"id": "r-idem-1", "method": "task.submit", "params": params})
    resp2 = server.handle_request({"id": "r-idem-2", "method": "task.submit", "params": params})

    assert worker.calls == ["help"]  # executed exactly once
    assert "replayed" not in resp1["result"]
    # task_id was not supplied → a fresh uuid4 default, always present.
    assert resp1["result"]["task_id"]
    assert resp1["result"]["status"] == "succeeded"
    assert resp2["result"]["replayed"] is True
    replay = dict(resp2["result"])
    replay.pop("replayed")
    assert replay == resp1["result"]


def test_task_submit_rejects_unsupported_intent(server):
    """Pilot scope is intent='slash' only — anything else is a 4030 error."""
    resp = server.handle_request({
        "id": "r-task-bad-intent",
        "method": "task.submit",
        "params": {
            "session_id": "test-session",
            "intent": "shell",
            "inputs": {"command": "ls"},
        },
    })

    assert resp["error"]["code"] == 4030
    assert "unsupported intent: shell" in resp["error"]["message"]
    assert "pilot supports 'slash'" in resp["error"]["message"]


def test_task_submit_propagates_slash_core_errors(server):
    """Early-reject errors from the shared slash core surface as the same
    JSON-RPC errors slash.exec returns (here: unknown session)."""
    _clear_task_results(server)
    resp = server.handle_request({
        "id": "r-task-no-session",
        "method": "task.submit",
        "params": {
            "session_id": "missing",
            "intent": "slash",
            "inputs": {"command": "help"},
        },
    })

    assert resp["error"]["code"] == 4001


def test_task_submit_slash_echoes_trace_id(server):
    """Task 2.1 (§12 trace plumbing): an optional params.trace_id is threaded
    onto the final ExecutionResult and echoed in the rich wire. The unset case
    stays byte-identical — pinned by the exact-dict assert in
    test_task_submit_slash_happy_path_returns_rich_result above."""
    sid = "test-session"
    worker = _FakeSlashWorker()
    server._sessions[sid] = {"session_key": sid, "agent": None, "slash_worker": worker}
    _clear_task_results(server)

    resp = server.handle_request({
        "id": "r-task-trace",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "slash",
            "inputs": {"command": "help"},
            "task_id": "task-tr-1",
            "trace_id": "trace-abc",
        },
    })

    assert "error" not in resp
    assert resp["result"] == {
        "task_id": "task-tr-1",
        "status": "succeeded",
        "outputs": {"output": "worker:help"},
        "error": None,
        "side_effects": [],
        "trace_id": "trace-abc",
    }


# ── AgentTaskRegistry RPCs (Phase 5 Steps 3-4) ───────────────────────


@pytest.fixture()
def registry(monkeypatch):
    """Fresh AgentTaskRegistry singleton per test — the registry is a
    process-level global, so leaking records across tests would make
    task.list/task.status assertions order-dependent."""
    import action_runtime.task_registry as task_registry_mod

    monkeypatch.setattr(task_registry_mod, "_instance", None)
    yield task_registry_mod.get_registry()
    task_registry_mod._instance = None


class _FakeInterruptibleAgent:
    def __init__(self):
        self.interrupted = False
        self.interrupt_message = None
        self._interrupt_requested = False

    def interrupt(self, message=None):
        self.interrupted = True
        self.interrupt_message = message
        self._interrupt_requested = True

    def clear_interrupt(self):
        self._interrupt_requested = False
        self.interrupt_message = None


def test_task_status_round_trip_returns_snapshot(server, registry):
    """task.status on a registered record returns its snapshot + found=true."""
    from action_runtime.task_registry import AgentTaskRecord

    registry.register(AgentTaskRecord(
        task_id="task-rt-1",
        goal="round trip",
        intent="test",
        model="test/model",
        started_at=123.0,
    ))

    resp = server.handle_request({
        "id": "r-task-status",
        "method": "task.status",
        "params": {"task_id": "task-rt-1"},
    })

    assert "error" not in resp
    assert resp["result"] == {
        "found": True,
        "task_id": "task-rt-1",
        "parent_task_id": None,
        "depth": 0,
        "goal": "round trip",
        "intent": "test",
        "model": "test/model",
        "started_at": 123.0,
        "finished_at": None,
        "status": "running",
        "tool_count": 0,
        "last_tool": None,
        "result": None,
    }


def test_task_status_unknown_id_is_found_false_not_error(server, registry):
    """A query miss is data, not a protocol error."""
    resp = server.handle_request({
        "id": "r-task-status-miss",
        "method": "task.status",
        "params": {"task_id": "no-such-task"},
    })

    assert "error" not in resp
    assert resp["result"] == {"found": False, "task_id": "no-such-task"}


def test_task_cancel_unknown_id_is_found_false(server, registry):
    resp = server.handle_request({
        "id": "r-task-cancel-miss",
        "method": "task.cancel",
        "params": {"task_id": "no-such-task"},
    })

    assert "error" not in resp
    assert resp["result"] == {"found": False, "task_id": "no-such-task"}


def test_task_cancel_interrupts_registered_agent(server, registry):
    """task.cancel resolves the record's agent_ref and calls interrupt()."""
    from action_runtime.task_registry import AgentTaskRecord

    fake = _FakeInterruptibleAgent()
    registry.register(AgentTaskRecord(task_id="task-c-1", agent_ref=fake))

    resp = server.handle_request({
        "id": "r-task-cancel",
        "method": "task.cancel",
        "params": {"task_id": "task-c-1"},
    })

    assert "error" not in resp
    assert resp["result"] == {"found": True, "task_id": "task-c-1"}
    assert fake.interrupted is True


def test_task_list_returns_active_only(server, registry):
    """Completed records drop out of task.list; running ones remain."""
    from action_runtime.task_registry import AgentTaskRecord

    registry.register(AgentTaskRecord(task_id="task-l-running", goal="live"))
    registry.register(AgentTaskRecord(task_id="task-l-done", goal="done"))
    registry.complete("task-l-done", None)

    resp = server.handle_request({
        "id": "r-task-list",
        "method": "task.list",
        "params": {},
    })

    assert "error" not in resp
    tasks = resp["result"]["tasks"]
    assert [t["task_id"] for t in tasks] == ["task-l-running"]
    assert tasks[0]["status"] == "running"


def test_subagent_interrupt_falls_back_to_legacy_dict_on_registry_miss(
    server, registry, monkeypatch
):
    """Dual-write era: when the registry doesn't know the id, the handler
    falls back to delegate_tool.interrupt_subagent (Step 7 removes this).
    The module is stubbed in sys.modules (run_agent-stub pattern above):
    importing the real delegate_tool here would execute its import chain
    under the fixture's mocked hermes_constants."""
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tools.delegate_tool",
        types.SimpleNamespace(
            interrupt_subagent=lambda sid: calls.append(sid) or True
        ),
    )

    resp = server.handle_request({
        "id": "r-sub-int-fallback",
        "method": "subagent.interrupt",
        "params": {"subagent_id": "sa-0-deadbeef"},
    })

    assert "error" not in resp
    assert resp["result"] == {"found": True, "subagent_id": "sa-0-deadbeef"}
    assert calls == ["sa-0-deadbeef"]


def test_subagent_interrupt_prefers_registry_hit(server, registry, monkeypatch):
    """A registry hit must NOT reach the legacy path."""
    from action_runtime.task_registry import AgentTaskRecord

    fake = _FakeInterruptibleAgent()
    registry.register(AgentTaskRecord(task_id="sa-1-cafebabe", agent_ref=fake))
    legacy = MagicMock(return_value=False)
    monkeypatch.setitem(
        sys.modules,
        "tools.delegate_tool",
        types.SimpleNamespace(interrupt_subagent=legacy),
    )

    resp = server.handle_request({
        "id": "r-sub-int-registry",
        "method": "subagent.interrupt",
        "params": {"subagent_id": "sa-1-cafebabe"},
    })

    assert resp["result"] == {"found": True, "subagent_id": "sa-1-cafebabe"}
    assert fake.interrupted is True
    # Reason parity with the legacy interrupt_subagent path: the agent sees
    # the same human-readable interrupt message either way.
    assert fake.interrupt_message == "Interrupted via TUI (sa-1-cafebabe)"
    legacy.assert_not_called()


def test_delegation_pause_writes_both_stores(server, registry, monkeypatch):
    """Dual-write era: delegation.pause sets the registry flag AND the old
    delegate_tool global (still authoritative until the Step 7 cutover)."""
    legacy = MagicMock(side_effect=lambda paused: paused)
    monkeypatch.setitem(
        sys.modules,
        "tools.delegate_tool",
        types.SimpleNamespace(set_spawn_paused=legacy),
    )

    resp = server.handle_request({
        "id": "r-del-pause",
        "method": "delegation.pause",
        "params": {"paused": True},
    })

    assert resp["result"] == {"paused": True}
    legacy.assert_called_once_with(True)
    assert registry.spawns_paused() is True

    resp = server.handle_request({
        "id": "r-del-unpause",
        "method": "delegation.pause",
        "params": {"paused": False},
    })

    assert resp["result"] == {"paused": False}
    legacy.assert_called_with(False)
    assert registry.spawns_paused() is False


# ── task.started / task.completed push events (Task 2.2) ─────────────


def _capture_task_events(server, monkeypatch):
    """Stub _emit and (re)install the observer: the module-init install
    targeted the import-time singleton, which the registry fixture replaced
    with a fresh one. _registry_task_event resolves _emit at call time, so
    the monkeypatched stub captures what would hit the wire."""
    events = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: events.append((event, sid, payload)),
    )
    server._install_task_observer()
    return events


def test_register_emits_task_started_event(server, registry, monkeypatch):
    """register() on a session-owned record pushes task.started to the parent
    session — wire payload is the registry snapshot itself."""
    from action_runtime.task_registry import AgentTaskRecord

    events = _capture_task_events(server, monkeypatch)
    registry.register(AgentTaskRecord(
        task_id="task-ev-1", session_id="sess-ev", goal="watch", intent="test",
    ))

    assert len(events) == 1
    event, sid, payload = events[0]
    assert event == "task.started"
    assert sid == "sess-ev"
    assert payload["task_id"] == "task-ev-1"
    assert payload["status"] == "running"
    assert payload["session_id"] == "sess-ev"


def test_complete_emits_task_completed_event(server, registry, monkeypatch):
    """A successful complete() pushes task.completed with the terminal
    snapshot; a rejected late duplicate must not re-emit."""
    from action_runtime.contract import ExecutionResult, Status
    from action_runtime.task_registry import AgentTaskRecord

    events = _capture_task_events(server, monkeypatch)
    registry.register(AgentTaskRecord(task_id="task-ev-2", session_id="sess-ev"))
    events.clear()

    result = ExecutionResult(
        task_id="task-ev-2", status=Status.SUCCEEDED, outputs={"output": "done"},
    )
    assert registry.complete("task-ev-2", result) is True
    assert registry.complete("task-ev-2", result) is False  # late duplicate

    assert len(events) == 1
    event, sid, payload = events[0]
    assert event == "task.completed"
    assert sid == "sess-ev"
    assert payload["task_id"] == "task-ev-2"
    assert payload["status"] == "succeeded"
    assert payload["result"]["outputs"] == {"output": "done"}


def test_task_event_emit_failure_never_breaks_register_or_complete(
    server, registry, monkeypatch
):
    """The observer call is guarded inside the registry: a transport bug in
    _emit must never break the ledger."""
    from action_runtime.task_registry import AgentTaskRecord, TaskStatus

    def _boom(*_a, **_k):
        raise RuntimeError("transport down")

    monkeypatch.setattr(server, "_emit", _boom)
    server._install_task_observer()

    rec = AgentTaskRecord(task_id="task-ev-3", session_id="sess-ev")
    registry.register(rec)  # must not raise
    assert registry.get("task-ev-3") is rec
    assert registry.complete("task-ev-3", None) is True  # must not raise
    assert registry.get("task-ev-3").status is TaskStatus.FAILED


def test_task_events_skip_records_without_session_id(server, registry, monkeypatch):
    """No session_id → no owning client → no emit (same skip rule the
    snapshot wire-compat pin encodes)."""
    from action_runtime.task_registry import AgentTaskRecord

    events = _capture_task_events(server, monkeypatch)
    registry.register(AgentTaskRecord(task_id="task-ev-4"))
    registry.complete("task-ev-4", None)

    assert events == []


# ── task.submit intent="delegate" (Phase 5 Step 5) ───────────────────
# The delegate engine is stubbed in sys.modules (run_agent-stub pattern
# above): importing the real delegate_tool here would execute its import
# chain under the fixture's mocked hermes_constants.


def _delegate_session(server, sid="test-session"):
    """Session with a live (weakref-able) agent for delegate submits.

    A real SessionState with the turn-lifecycle keys: _task_submit_delegate
    claims the session via begin_turn()/end_turn() (same gate prompt.submit
    uses), so the busy machinery must be present.
    """
    agent = _FakeInterruptibleAgent()
    server._sessions[sid] = SessionState({
        "session_key": sid,
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": False,
        "inflight_turn": None,
        "last_active": 0.0,
    })
    return agent


def _stub_delegate_engine(monkeypatch, fake):
    monkeypatch.setitem(
        sys.modules,
        "tools.delegate_tool",
        types.SimpleNamespace(delegate_task=fake),
    )


def test_task_submit_delegate_all_success_returns_succeeded(
    server, registry, monkeypatch
):
    """intent=delegate happy path: every child completed -> status=succeeded
    with the engine's per-child breakdown carried verbatim in outputs."""
    sid = "test-session"
    agent = _delegate_session(server, sid)
    calls = []

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        calls.append((tasks, parent_agent))
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "a done",
                 "error": None, "api_calls": 2, "duration_seconds": 1},
                {"task_index": 1, "status": "completed", "summary": "b done",
                 "error": None, "api_calls": 3, "duration_seconds": 2},
            ],
            "total_duration_seconds": 2.5,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-ok",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["task a", {"goal": "task b"}]},
            "task_id": "task-del-1",
        },
    })

    assert "error" not in resp
    r = resp["result"]
    assert r["task_id"] == "task-del-1"
    assert r["status"] == "succeeded"
    assert r["error"] is None
    assert [e["status"] for e in r["outputs"]["results"]] == [
        "completed", "completed",
    ]
    assert r["outputs"]["total_duration_seconds"] == 2.5
    # Goal strings were coerced to the engine's own {goal: ...} task shape
    # and the session's live agent was passed as the parent.
    assert calls == [([{"goal": "task a"}, {"goal": "task b"}], agent)]
    # The parent task is a real registry record, completed terminally, tied
    # to the gateway session (Step 6 persistence identity).
    record = registry.get("task-del-1")
    assert record is not None
    assert record.intent == "delegate"
    assert record.status.value == "succeeded"
    assert record.session_id == sid
    # Pre-trace caller: no trace_id key anywhere (Task 2.1 byte-compat pin).
    assert "trace_id" not in resp["result"]
    assert "trace_id" not in record.snapshot()
    # The turn claim is released once the batch finishes.
    assert server._sessions[sid]["running"] is False


def test_task_submit_delegate_links_children_to_submitted_task_id(
    server, registry, monkeypatch
):
    """Task 2.3 (doc Step 5 ruling): the gateway hands its parent record's
    task_id to the engine as registry_parent_task_id, so children registered
    during the run link to the submitted task_id — not the engine-internal
    _parent_subagent_id."""
    from action_runtime.contract import ExecutionResult, Status
    from action_runtime.task_registry import AgentTaskRecord

    sid = "test-session"
    _delegate_session(server, sid)
    seen = {}

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        # Register a child mid-run through the REAL registry register path,
        # with parent_task_id from the caller-supplied kwarg — the same
        # record shape _run_single_child produces (the engine side of this
        # contract is pinned in tests/tools/test_delegate_registry_dualwrite.py).
        seen["registry_parent_task_id"] = kwargs.get("registry_parent_task_id")
        registry.register(AgentTaskRecord(
            task_id="sa-0-child001",
            parent_task_id=kwargs.get("registry_parent_task_id"),
            goal="task a",
            label="task a",
            intent="delegate",
        ))
        registry.complete(
            "sa-0-child001",
            ExecutionResult(
                task_id="sa-0-child001",
                status=Status.SUCCEEDED,
                outputs={"summary": "a done"},
            ),
        )
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "a done",
                 "error": None},
            ],
            "total_duration_seconds": 0.5,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-link",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["task a"]},
            "task_id": "task-del-link",
        },
    })

    assert "error" not in resp
    assert resp["result"]["status"] == "succeeded"
    # The engine received the parent record's id...
    assert seen["registry_parent_task_id"] == "task-del-link"
    # ...so the child record links to the submitted parent task_id.
    child = registry.get("sa-0-child001")
    assert child is not None
    assert child.parent_task_id == "task-del-link"
    assert registry.get("task-del-link").status.value == "succeeded"


def test_task_submit_delegate_echoes_trace_id(server, registry, monkeypatch):
    """Task 2.1 (§12): intent=delegate threads trace_id onto the parent
    AgentTaskRecord AND the final ExecutionResult before complete(), so the
    rich wire, the registry record, and its snapshot all correlate."""
    sid = "test-session"
    _delegate_session(server, sid)

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done",
                 "error": None},
            ],
            "total_duration_seconds": 1.0,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-trace",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-tr",
            "trace_id": "trace-del-1",
        },
    })

    assert "error" not in resp
    assert resp["result"]["trace_id"] == "trace-del-1"
    record = registry.get("task-del-tr")
    assert record.trace_id == "trace-del-1"
    assert record.result.trace_id == "trace-del-1"
    snap = record.snapshot()
    assert snap["trace_id"] == "trace-del-1"
    assert snap["result"]["trace_id"] == "trace-del-1"


def test_task_submit_delegate_mixed_batch_is_partial(server, registry, monkeypatch):
    """Q4 ruling pin: some children finished, some did not -> PARTIAL (never
    FAILED — the Core must be able to re-plan only the unfinished children)."""
    sid = "test-session"
    _delegate_session(server, sid)

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done",
                 "error": None},
                {"task_index": 1, "status": "interrupted", "summary": None,
                 "error": "Parent agent interrupted — child did not finish in time"},
            ],
            "total_duration_seconds": 4.0,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-partial",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a", "b"]},
            "task_id": "task-del-2",
        },
    })

    assert "error" not in resp
    r = resp["result"]
    assert r["status"] == "partial"
    assert r["error"] is None
    assert [e["status"] for e in r["outputs"]["results"]] == [
        "completed", "interrupted",
    ]
    assert registry.get("task-del-2").status.value == "partial"


def test_task_submit_delegate_engine_exception_is_failed(
    server, registry, monkeypatch
):
    """An exception escaping the engine maps to FAILED + internal ExecError —
    never a protocol error (the submit itself round-tripped honestly)."""
    sid = "test-session"
    _delegate_session(server, sid)

    def boom(tasks=None, parent_agent=None, **kwargs):
        raise RuntimeError("kaboom")

    _stub_delegate_engine(monkeypatch, boom)

    resp = server.handle_request({
        "id": "r-del-boom",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-3",
        },
    })

    assert "error" not in resp
    r = resp["result"]
    assert r["status"] == "failed"
    assert r["error"]["type"] == "internal"
    assert r["error"]["retryable"] is False
    assert "kaboom" in r["error"]["message"]
    assert registry.get("task-del-3").status.value == "failed"


def test_task_submit_delegate_missing_tasks_is_param_error(server, registry):
    """Missing, non-list, or empty inputs.tasks -> 4012 param error before
    any session/agent work happens."""
    sid = "test-session"
    _delegate_session(server, sid)

    for bad_inputs in ({}, {"tasks": []}, {"tasks": "not-a-list"}):
        resp = server.handle_request({
            "id": "r-del-bad-tasks",
            "method": "task.submit",
            "params": {
                "session_id": sid,
                "intent": "delegate",
                "inputs": bad_inputs,
            },
        })
        assert resp["error"]["code"] == 4012
        assert "inputs.tasks required" in resp["error"]["message"]

    resp = server.handle_request({
        "id": "r-del-bad-item",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["ok", 42]},
        },
    })
    assert resp["error"]["code"] == 4012
    assert "inputs.tasks[1]" in resp["error"]["message"]


def test_task_submit_delegate_without_live_agent_errors(server, registry):
    """A session without a live agent cannot delegate."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    resp = server.handle_request({
        "id": "r-del-no-agent",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
        },
    })

    assert resp["error"]["code"] == 4010


def test_task_submit_delegate_idempotency_replay_does_not_rerun(
    server, registry, monkeypatch
):
    """Re-submitting the SAME idempotency_key replays the cached rich dict
    + replayed=true without invoking the delegate engine again."""
    sid = "test-session"
    _delegate_session(server, sid)
    calls = []

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        calls.append(tasks)
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done",
                 "error": None},
            ],
            "total_duration_seconds": 0.1,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    params = {
        "session_id": sid,
        "intent": "delegate",
        "inputs": {"tasks": ["a"]},
        "idempotency_key": "idem-del-1",
    }
    resp1 = server.handle_request(
        {"id": "r-del-idem-1", "method": "task.submit", "params": params}
    )
    resp2 = server.handle_request(
        {"id": "r-del-idem-2", "method": "task.submit", "params": params}
    )

    assert len(calls) == 1  # executed exactly once
    assert "replayed" not in resp1["result"]
    assert resp1["result"]["status"] == "succeeded"
    assert resp2["result"]["replayed"] is True
    replay = dict(resp2["result"])
    replay.pop("replayed")
    assert replay == resp1["result"]


def test_task_submit_delegate_rejects_busy_session(server, registry, monkeypatch):
    """A delegate batch must never run concurrently with a user turn on the
    SAME agent (cross-coupled interrupt state) — busy sessions get the same
    4009 rejection prompt.submit gives, and the engine never runs."""
    sid = "test-session"
    _delegate_session(server, sid)
    server._sessions[sid]["running"] = True
    calls = []

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        calls.append(tasks)
        return json.dumps({"results": [], "total_duration_seconds": 0})

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-busy",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-busy",
        },
    })

    assert resp["error"]["code"] == 4009
    assert resp["error"]["message"] == "session busy"
    assert calls == []
    # The rejection happened before the parent record was registered.
    assert registry.get("task-del-busy") is None
    # The pre-existing claim is untouched (we never began a turn).
    assert server._sessions[sid]["running"] is True


def test_task_submit_delegate_inflight_duplicate_task_id_is_rejected(
    server, registry, monkeypatch
):
    """An idempotent retry while the FIRST run is still in flight must not
    re-execute the batch: replay only covers keys AFTER completion, so a
    RUNNING task_id is rejected with 4034 instead of re-running."""
    from action_runtime.task_registry import AgentTaskRecord

    sid = "test-session"
    _delegate_session(server, sid)
    registry.register(AgentTaskRecord(task_id="task-dup-1", intent="delegate"))
    calls = []

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        calls.append(tasks)
        return json.dumps({"results": [], "total_duration_seconds": 0})

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-dup",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-dup-1",
        },
    })

    assert resp["error"]["code"] == 4034
    assert "already running" in resp["error"]["message"]
    assert calls == []
    # The first run's record is untouched.
    assert registry.get("task-dup-1").status.value == "running"


def test_task_submit_delegate_clears_stale_parent_interrupt(
    server, registry, monkeypatch
):
    """A task.cancel during the batch interrupts the PARENT agent; the flag
    must not survive the submit, or it would abort the user's NEXT turn."""
    sid = "test-session"
    agent = _delegate_session(server, sid)

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        # Simulate task.cancel arriving mid-batch: the registry resolves the
        # parent record's agent_ref and calls interrupt() on it.
        assert registry.interrupt("task-del-int") is True
        assert parent_agent._interrupt_requested is True
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "interrupted", "summary": None,
                 "error": "Parent agent interrupted — child did not finish in time"},
            ],
            "total_duration_seconds": 0.2,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-int",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-int",
        },
    })

    assert "error" not in resp
    assert resp["result"]["status"] == "failed"
    # The stale interrupt was cleared and the session released.
    assert agent._interrupt_requested is False
    assert server._sessions[sid]["running"] is False


def test_task_submit_delegate_denied_only_for_spawn_pause(
    server, registry, monkeypatch
):
    """Only the operator spawn-pause guard maps to DENIED/retryable=True;
    any other no-results {'error': ...} aggregate (credential resolution,
    task-shape validation, ...) is INTERNAL/retryable=False — not an
    authorization denial."""
    sid = "test-session"
    _delegate_session(server, sid)
    aggregates = iter([
        json.dumps({
            "error": (
                "Delegation spawning is paused. Clear the pause via the TUI "
                "(`p` in /agents) or the `delegation.pause` RPC before retrying."
            )
        }),
        json.dumps({"error": "Failed to resolve delegation credentials: no key"}),
    ])
    _stub_delegate_engine(
        monkeypatch, lambda tasks=None, parent_agent=None, **kw: next(aggregates)
    )

    resp = server.handle_request({
        "id": "r-del-paused",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-paused",
        },
    })
    assert resp["result"]["status"] == "failed"
    assert resp["result"]["error"]["type"] == "denied"
    assert resp["result"]["error"]["retryable"] is True

    resp = server.handle_request({
        "id": "r-del-creds",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-del-creds",
        },
    })
    assert resp["result"]["status"] == "failed"
    assert resp["result"]["error"]["type"] == "internal"
    assert resp["result"]["error"]["retryable"] is False


def test_task_submit_delegate_persists_tasks_jsonl_line(
    server, registry, spawn_home, monkeypatch
):
    """Step 6 end-to-end on a production-shaped path: the parent delegate
    record carries the gateway session_id, so complete() appends one
    snapshot line to that session's _tasks.jsonl under the (tmp) home."""
    sid = "sess-e2e"
    _delegate_session(server, sid)

    def fake_delegate_task(tasks=None, parent_agent=None, **kwargs):
        return json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done",
                 "error": None},
            ],
            "total_duration_seconds": 0.1,
        })

    _stub_delegate_engine(monkeypatch, fake_delegate_task)

    resp = server.handle_request({
        "id": "r-del-persist",
        "method": "task.submit",
        "params": {
            "session_id": sid,
            "intent": "delegate",
            "inputs": {"tasks": ["a"]},
            "task_id": "task-e2e-1",
        },
    })

    assert resp["result"]["status"] == "succeeded"
    lines = (
        (spawn_home / "spawn-trees" / sid / "_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["task_id"] == "task-e2e-1"
    assert entry["session_id"] == sid
    assert entry["status"] == "succeeded"
    # Q5 parity: the batch label (single task -> its goal) rides the line.
    assert entry["label"] == "a"


def test_registry_register_bg_session_id_persists_on_complete(
    server, registry, spawn_home
):
    """The bg/preview register helper threads session_id onto the record, so
    the completion path appends to that session's _tasks.jsonl."""
    server._registry_register_bg(
        "bg_abc123", goal="poke", intent="tui-background", session_id="sess-bg"
    )
    assert registry.get("bg_abc123").session_id == "sess-bg"

    server._registry_complete_bg("bg_abc123", text="done")

    lines = (
        (spawn_home / "spawn-trees" / "sess-bg" / "_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["task_id"] == "bg_abc123"
    assert entry["session_id"] == "sess-bg"
    assert entry["status"] == "succeeded"
    # Q5 parity: bg records alias the goal as their label.
    assert entry["label"] == "poke"


# ── spawn_tree.list: legacy _index.jsonl + registry _tasks.jsonl (Step 6b) ──


@pytest.fixture()
def spawn_home(server, tmp_path, monkeypatch):
    """Point the lazily-imported get_hermes_home at a tmp dir so the
    spawn-tree handlers read/write under tmp_path/spawn-trees."""
    monkeypatch.setattr(
        sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path
    )
    return tmp_path


def _spawn_session_dir(spawn_home, session_id="sess-1"):
    d = spawn_home / "spawn-trees" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_legacy_snapshot(session_dir, fname, label="legacy run", **index_extra):
    """One saved snapshot file + its _index.jsonl line, as spawn_tree.save
    writes them."""
    path = session_dir / fname
    path.write_text(
        json.dumps({"session_id": session_dir.name, "subagents": [{}, {}]}),
        encoding="utf-8",
    )
    entry = {
        "path": str(path),
        "session_id": session_dir.name,
        "started_at": 100.0,
        "finished_at": 200.0,
        "label": label,
        "count": 2,
        **index_extra,
    }
    with (session_dir / "_index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _write_task_snapshot(
    session_dir, task_id, goal="child goal", finished_at=300.0, result=None, **extra
):
    """One registry-completed line, as AgentTaskRegistry.complete() appends.
    ``extra`` carries the conditional snapshot keys (label/tools/error/...)
    newer records emit."""
    snap = {
        "task_id": task_id,
        "parent_task_id": None,
        "depth": 1,
        "goal": goal,
        "intent": "delegate",
        "model": "test/model",
        "started_at": 250.0,
        "finished_at": finished_at,
        "status": "succeeded",
        "tool_count": 3,
        "last_tool": "bash",
        "result": result,
        "session_id": session_dir.name,
        **extra,
    }
    with (session_dir / "_tasks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")
    return snap


def _spawn_tree_list(server, session_id="sess-1"):
    resp = server.handle_request({
        "id": "r-st-list",
        "method": "spawn_tree.list",
        "params": {"session_id": session_id},
    })
    assert "error" not in resp
    return resp["result"]["entries"]


def test_spawn_tree_list_legacy_only_unchanged(server, spawn_home):
    """No _tasks.jsonl: the wire shape is byte-identical to the pre-6b list
    (same keys, no additive ones)."""
    d = _spawn_session_dir(spawn_home)
    expected = _write_legacy_snapshot(d, "20260610T000000.json")

    entries = _spawn_tree_list(server)

    assert entries == [expected]
    assert set(entries[0]) == {
        "path", "session_id", "started_at", "finished_at", "label", "count",
    }


def test_spawn_tree_list_tasks_only_entries_appear(server, spawn_home):
    """A session with only the registry ledger still lists its tasks, mapped
    onto the legacy entry schema."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(d, "sa-0-aaaa", goal="first child", finished_at=300.0)
    _write_task_snapshot(d, "sa-1-bbbb", goal="second child", finished_at=400.0)

    entries = _spawn_tree_list(server)

    assert [e["task_id"] for e in entries] == ["sa-1-bbbb", "sa-0-aaaa"]  # newest first
    e = entries[1]
    assert e["path"] == str(d / "_tasks.jsonl")
    assert e["session_id"] == "sess-1"
    assert e["started_at"] == 250.0
    assert e["finished_at"] == 300.0
    assert e["label"] == "first child"
    assert e["count"] == 1
    assert e["status"] == "succeeded"
    assert e["source"] == "registry"


def test_spawn_tree_list_merge_dedupes_by_task_id_preferring_legacy(
    server, spawn_home
):
    """Q5 ruling: same task_id in both sources -> the TUI-assembled legacy
    entry wins; registry-only ids are appended."""
    d = _spawn_session_dir(spawn_home)
    legacy = _write_legacy_snapshot(
        d, "20260610T000000.json", label="rich legacy", task_id="sa-0-aaaa"
    )
    _write_task_snapshot(d, "sa-0-aaaa", goal="poor registry copy")
    _write_task_snapshot(d, "sa-1-bbbb", goal="registry only", finished_at=400.0)

    entries = _spawn_tree_list(server)

    assert len(entries) == 2
    by_id = {e["task_id"]: e for e in entries}
    assert by_id["sa-0-aaaa"] == legacy  # legacy entry, untouched
    assert by_id["sa-1-bbbb"]["label"] == "registry only"
    assert by_id["sa-1-bbbb"]["source"] == "registry"


def test_spawn_tree_list_prefers_record_label(server, spawn_home):
    """Q5 sunset progress: when the snapshot carries the additive "label"
    key, the list entry uses it over the goal fallback."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(
        d, "sa-0-aaaa", goal="first goal", label="goal one · goal two"
    )

    (e,) = _spawn_tree_list(server)
    assert e["label"] == "goal one · goal two"
    assert e["source"] == "registry"


def test_spawn_tree_list_skips_corrupt_tasks_lines(server, spawn_home):
    """Corrupt _tasks.jsonl lines are skipped silently, like the legacy
    index reader."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(d, "sa-0-aaaa")
    with (d / "_tasks.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write("\n")
        f.write('["a", "json", "array", "not", "a", "dict"]\n')
    _write_task_snapshot(d, "sa-1-bbbb")

    entries = _spawn_tree_list(server)

    assert sorted(e["task_id"] for e in entries) == ["sa-0-aaaa", "sa-1-bbbb"]


# ── spawn_tree.load: legacy snapshots + registry _tasks.jsonl (gap 3) ──


def _spawn_tree_load(server, path, **extra):
    return server.handle_request({
        "id": "r-st-load",
        "method": "spawn_tree.load",
        "params": {"path": str(path), **extra},
    })


def test_spawn_tree_load_legacy_snapshot_unchanged(server, spawn_home):
    """Pin: a legacy snapshot path flows through the exact pre-existing code
    path — the stored payload comes back verbatim."""
    d = _spawn_session_dir(spawn_home)
    _write_legacy_snapshot(d, "20260610T000000.json")

    resp = _spawn_tree_load(server, d / "20260610T000000.json")

    assert "error" not in resp
    assert resp["result"] == {"session_id": "sess-1", "subagents": [{}, {}]}


def test_spawn_tree_load_ledger_returns_last_snapshot(server, spawn_home):
    """Loading the _tasks.jsonl path a registry-only list entry carries
    synthesizes a legacy-shaped payload from the LAST ledger line."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(d, "sa-0-aaaa", goal="first child", finished_at=300.0)
    _write_task_snapshot(
        d,
        "sa-1-bbbb",
        goal="second child",
        finished_at=400.0,
        result={
            "task_id": "sa-1-bbbb",
            "status": "succeeded",
            "outputs": {"output": "rich answer text"},
            "error": None,
            "side_effects": [],
        },
    )

    resp = _spawn_tree_load(server, d / "_tasks.jsonl")

    assert "error" not in resp
    payload = resp["result"]
    assert set(payload) == {
        "session_id", "started_at", "finished_at", "label", "subagents",
        "task_id", "source",
    }
    assert payload["session_id"] == "sess-1"
    assert payload["started_at"] == 250.0
    assert payload["finished_at"] == 400.0
    assert payload["label"] == "second child"
    assert payload["task_id"] == "sa-1-bbbb"
    assert payload["source"] == "registry"
    # One node, the task itself — no fake children.
    (sub,) = payload["subagents"]
    assert sub["id"] == "sa-1-bbbb"
    assert sub["parentId"] is None
    assert sub["depth"] == 1
    assert sub["goal"] == "second child"
    assert sub["status"] == "completed"  # "succeeded" mapped to TUI vocabulary
    assert sub["model"] == "test/model"
    assert sub["toolCount"] == 3
    assert sub["tools"] == ["bash"]
    assert sub["startedAt"] == 250_000.0  # ms epoch, like live subagents
    assert sub["durationSeconds"] == 150.0
    assert sub["summary"] == "rich answer text"  # the rendered node body


def test_spawn_tree_load_ledger_prefers_q5_parity_fields(server, spawn_home):
    """Q5 sunset progress: a snapshot carrying the additive label/tools/error
    keys loads with the richer values — full tools tail over the last_tool
    fallback, explicit label over the goal, error summary as the node body
    when the result has no text."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(
        d,
        "sa-0-aaaa",
        goal="child goal",
        label="goal one · goal two",
        tools=["bash", "web_search", "bash"],
        error="provider exploded",
        result={
            "task_id": "sa-0-aaaa",
            "status": "failed",
            "outputs": {},
            "error": None,  # no message in the rich result itself
            "side_effects": [],
        },
    )

    resp = _spawn_tree_load(server, d / "_tasks.jsonl")

    assert "error" not in resp
    payload = resp["result"]
    assert payload["label"] == "goal one · goal two"
    (sub,) = payload["subagents"]
    assert sub["goal"] == "child goal"  # the node keeps its own goal
    assert sub["tools"] == ["bash", "web_search", "bash"]  # not ["bash"] tail
    assert sub["summary"] == "provider exploded"


def test_spawn_tree_load_ledger_task_id_param_picks_record(server, spawn_home):
    """An explicit task_id param selects that task's latest ledger line, not
    the file's last line."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(d, "sa-0-aaaa", goal="stale first write")
    _write_task_snapshot(d, "sa-0-aaaa", goal="first child")  # later line wins
    _write_task_snapshot(d, "sa-1-bbbb", goal="second child", finished_at=400.0)

    resp = _spawn_tree_load(server, d / "_tasks.jsonl", task_id="sa-0-aaaa")

    assert "error" not in resp
    payload = resp["result"]
    assert payload["task_id"] == "sa-0-aaaa"
    assert payload["label"] == "first child"
    assert payload["subagents"][0]["id"] == "sa-0-aaaa"

    missing = _spawn_tree_load(server, d / "_tasks.jsonl", task_id="sa-9-zzzz")
    assert missing["error"]["code"] == 5000


def test_spawn_tree_load_ledger_skips_corrupt_lines(server, spawn_home):
    """A corrupt trailing ledger line is skipped — the last VALID snapshot
    loads, same lenient reads as spawn_tree.list."""
    d = _spawn_session_dir(spawn_home)
    _write_task_snapshot(d, "sa-0-aaaa", goal="only valid line")
    with (d / "_tasks.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json\n")

    resp = _spawn_tree_load(server, d / "_tasks.jsonl")

    assert "error" not in resp
    assert resp["result"]["task_id"] == "sa-0-aaaa"
    assert resp["result"]["label"] == "only valid line"


def test_spawn_tree_load_missing_file_errors_gracefully(server, spawn_home):
    """Missing files still produce the graceful 5000 error, never a crash —
    both for legacy snapshot paths and for ledger paths."""
    d = _spawn_session_dir(spawn_home)

    legacy = _spawn_tree_load(server, d / "20260101T000000.json")
    assert legacy["error"]["code"] == 5000

    ledger = _spawn_tree_load(server, d / "_tasks.jsonl")
    assert ledger["error"]["code"] == 5000


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


def test_command_dispatch_queue_requires_arg(server):
    """command.dispatch /queue without an argument returns an error."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    resp = server.handle_request({
        "id": "r2",
        "method": "command.dispatch",
        "params": {"name": "queue", "arg": "", "session_id": sid},
    })

    assert "error" in resp
    assert resp["error"]["code"] == 4004


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


def test_command_dispatch_steer_fallback_sends_message(server):
    """command.dispatch /steer with no active agent falls back to send."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    resp = server.handle_request({
        "id": "r3",
        "method": "command.dispatch",
        "params": {"name": "steer", "arg": "focus on testing", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "focus on testing"


def test_command_dispatch_retry_finds_last_user_message(server):
    """command.dispatch /retry walks session['history'] to find the last user message."""
    sid = "test-session"
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    server._sessions[sid] = SessionState({
        "session_key": sid,
        "agent": None,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
    })

    resp = server.handle_request({
        "id": "r4",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "second question"
    # Verify history was truncated: everything from last user message onward removed
    assert len(server._sessions[sid]["history"]) == 2
    assert server._sessions[sid]["history"][-1]["role"] == "assistant"
    assert server._sessions[sid]["history_version"] == 1


def test_command_dispatch_retry_empty_history(server):
    """command.dispatch /retry with empty history returns error."""
    sid = "test-session"
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    resp = server.handle_request({
        "id": "r5",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" in resp
    assert resp["error"]["code"] == 4018


def test_command_dispatch_retry_handles_multipart_content(server):
    """command.dispatch /retry extracts text from multipart content lists."""
    sid = "test-session"
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "analyze this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]},
        {"role": "assistant", "content": "I see the image."},
    ]
    server._sessions[sid] = SessionState({
        "session_key": sid,
        "agent": None,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
    })

    resp = server.handle_request({
        "id": "r6",
        "method": "command.dispatch",
        "params": {"name": "retry", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "analyze this"


def test_command_dispatch_returns_skill_payload(server):
    """command.dispatch returns structured skill payload for the TUI to send()."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}
    fake_msg = "Loaded skill content here"

    with patch("agent.skill_commands.scan_skill_commands", return_value=fake_skills), \
         patch("agent.skill_commands.build_skill_invocation_message", return_value=fake_msg):
        resp = server.handle_request({
            "id": "r2",
            "method": "command.dispatch",
            "params": {"name": "hermes-agent-dev", "session_id": sid},
        })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "skill"
    assert result["message"] == fake_msg
    assert result["name"] == "hermes-agent-dev"


def test_command_dispatch_awaits_async_plugin_handler(server):
    async def _handler(arg):
        return f"async:{arg}"

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: _handler if name == "async-cmd" else None,
    ):
        resp = server.handle_request({
            "id": "r-plugin",
            "method": "command.dispatch",
            "params": {"name": "async-cmd", "arg": "hello"},
        })

    assert "error" not in resp
    assert resp["result"] == {"type": "plugin", "output": "async:hello"}


# ── dispatch(): pool routing for long handlers (#12546) ──────────────


def test_dispatch_runs_short_handlers_inline(server):
    """Non-long handlers return their response synchronously from dispatch()."""
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    resp = server.dispatch({"id": "r1", "method": "fast.ping", "params": {}})

    assert resp == {"jsonrpc": "2.0", "id": "r1", "result": {"pong": True}}


def test_dispatch_offloads_long_handlers_and_emits_via_stdout(capture):
    """Long handlers run on the pool and write their response via write_json."""
    server, buf = capture
    server._methods["slash.exec"] = lambda rid, params: server._ok(rid, {"output": "hi"})

    resp = server.dispatch({"id": "r2", "method": "slash.exec", "params": {}})
    assert resp is None

    for _ in range(50):
        if buf.getvalue():
            break
        time.sleep(0.01)

    written = json.loads(buf.getvalue())
    assert written == {"jsonrpc": "2.0", "id": "r2", "result": {"output": "hi"}}


def test_dispatch_long_handler_does_not_block_fast_handler(server):
    """A slow long handler must not prevent a concurrent fast handler from completing."""
    released = threading.Event()
    server._methods["slash.exec"] = lambda rid, params: (released.wait(timeout=5), server._ok(rid, {"done": True}))[1]
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    t0 = time.monotonic()
    assert server.dispatch({"id": "slow", "method": "slash.exec", "params": {}}) is None

    fast_resp = server.dispatch({"id": "fast", "method": "fast.ping", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"pong": True}
    assert fast_elapsed < 0.5, f"fast handler blocked for {fast_elapsed:.2f}s behind slow handler"

    released.set()


def test_dispatch_session_compress_does_not_block_fast_handler(server):
    """Manual TUI compaction can take minutes, so it must not block the RPC loop."""
    released = threading.Event()

    def slow_compress(rid, params):
        released.wait(timeout=5)
        return server._ok(rid, {"done": True})

    server._methods["session.compress"] = slow_compress
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    t0 = time.monotonic()
    assert server.dispatch({"id": "slow", "method": "session.compress", "params": {}}) is None

    fast_resp = server.dispatch({"id": "fast", "method": "fast.ping", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"pong": True}
    assert fast_elapsed < 0.5, f"fast handler blocked for {fast_elapsed:.2f}s behind session.compress"

    released.set()


def test_dispatch_long_handler_exception_produces_error_response(capture):
    """An exception inside a pool-dispatched handler still yields a JSON-RPC error."""
    server, buf = capture

    def boom(rid, params):
        raise RuntimeError("kaboom")

    server._methods["slash.exec"] = boom

    server.dispatch({"id": "r3", "method": "slash.exec", "params": {}})

    for _ in range(50):
        if buf.getvalue():
            break
        time.sleep(0.01)

    written = json.loads(buf.getvalue())
    assert written["id"] == "r3"
    assert written["error"]["code"] == -32000
    assert "kaboom" in written["error"]["message"]


def test_dispatch_unknown_long_method_still_goes_inline(server):
    """Method name not in _LONG_HANDLERS takes the sync path even if handler is slow."""
    server._methods["some.method"] = lambda rid, params: server._ok(rid, {"ok": True})

    resp = server.dispatch({"id": "r4", "method": "some.method", "params": {}})

    assert resp["result"] == {"ok": True}
