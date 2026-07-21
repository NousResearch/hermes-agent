import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    MUTATOR_ROUTE_TABLE,
    HostSupervisor,
    append_log_record,
)


def _json_lines(out: io.StringIO) -> list[dict]:
    frames = []
    for line in out.getvalue().splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def _wait_for_frame(out: io.StringIO, predicate, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _json_lines(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for frame; saw={_json_lines(out)}")


def test_compute_host_workers_inherit_tui_pool_env_or_8(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_RPC_POOL_WORKERS", raising=False)
    monkeypatch.delenv("HERMES_COMPUTE_HOST_WORKERS", raising=False)
    assert _default_workers() == 8

    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "11")
    assert _default_workers() == 11

    # Dead-RC tombstone: malformed env falls back to 8, not the old except-branch 4.
    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "not-an-int")
    assert _default_workers() == 8


def test_compute_host_frame_protocol_round_trip():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=2, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-1",
                "prompt": "hello",
                "delta_count": 3,
                "delay_s": 0,
            }
        )

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-1")
        assert end["history_version"] == 1
        frames = _json_lines(out)
        assert [f["type"] for f in frames if f.get("request_id") == "turn-1"] == [
            "turn.started",
            "delta",
            "delta",
            "delta",
            "turn.end",
        ]
    finally:
        host.close()


@pytest.mark.parametrize(
    ("event", "method", "value_key", "value"),
    [
        ("clarify.request", "clarify.respond", "answer", "clarified"),
        ("input.request", "clarify.respond", "answer", "input"),
        ("sudo.request", "sudo.respond", "password", "sudo-value"),
        ("secret.request", "secret.respond", "value", "secret-value"),
        (
            "terminal.read.request",
            "terminal.read.respond",
            "text",
            '{"lines":["terminal"]}',
        ),
    ],
)
def test_compute_host_interactive_response_reaches_real_child_waiter(
    monkeypatch, event, method, value_key, value
):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    sid = f"child-{event}"
    server._sessions[sid] = {
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": f"key-{event}",
        "transport": host._transport,
    }
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    answer: dict[str, str] = {}
    waiter = threading.Thread(
        target=lambda: answer.setdefault(
            "value", server._block(event, sid, {"prompt": event}, timeout=2)
        )
    )
    try:
        waiter.start()
        request = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "interactive.request"
            and frame.get("event") == event,
        )
        assert request["source_sid"] == sid
        assert request["request_id"]

        host.handle_frame(
            {
                "type": "interactive.response",
                "sid": sid,
                "request_id": f"response-{event}",
                "interactive_request_id": request["request_id"],
                "method": method,
                "params": {value_key: value},
            }
        )
        ack = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "interactive.response.ack"
            and frame.get("request_id") == f"response-{event}",
        )
        assert ack["interactive_request_id"] == request["request_id"]
        _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "interactive.complete"
            and frame.get("request_id") == request["request_id"],
        )
        waiter.join(2)
        assert answer == {"value": value}
    finally:
        server._clear_pending(sid)
        waiter.join(2)
        server._sessions.pop(sid, None)
        with server._prompt_lock:
            server._interactive_prompt_queues.clear()
        host.close()


def test_compute_host_interactive_response_reaches_real_approval_resolver(
    monkeypatch,
):
    from tools import approval
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    sid = "child-approval"
    session_key = "child-approval-key"
    server._sessions[sid] = {
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "transport": host._transport,
    }
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 2)
    result: dict[str, dict] = {}
    waiter = threading.Thread(
        target=lambda: result.setdefault(
            "value",
            approval._await_gateway_decision(
                session_key,
                lambda data: server._emit_approval_request(sid, data),
                {
                    "command": "dangerous",
                    "description": "needs approval",
                    "pattern_key": "test",
                    "pattern_keys": ["test"],
                },
            ),
        )
    )
    try:
        waiter.start()
        request = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "interactive.request"
            and frame.get("event") == "approval.request",
        )
        host.handle_frame(
            {
                "type": "interactive.response",
                "sid": sid,
                "request_id": "approval-response",
                "interactive_request_id": request["request_id"],
                "method": "approval.respond",
                "params": {"choice": "once"},
            }
        )
        _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "interactive.response.ack"
            and frame.get("request_id") == "approval-response",
        )
        waiter.join(2)
        assert result["value"]["resolved"] is True
        assert result["value"]["choice"] == "once"
        assert not approval.has_blocking_approval(session_key)
    finally:
        approval.resolve_gateway_approval(session_key, "deny", resolve_all=True)
        server._drop_pending_approvals(sid)
        waiter.join(2)
        server._sessions.pop(sid, None)
        with server._prompt_lock:
            server._interactive_prompt_queues.clear()
        host.close()


@pytest.mark.parametrize("init_fails", [False, True])
def test_compute_host_remote_profile_is_bound_through_session_init_and_fallback(
    monkeypatch, tmp_path, init_fails
):
    """Agent, _init_session, and fallback config reads share one profile."""
    from tui_gateway import server

    host = ComputeHost(stdout=io.StringIO(), max_workers=1, heartbeat_secs=0)
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    sid = f"remote-profile-{'fallback' if init_fails else 'init'}"
    captured: dict[str, Any] = {"fallback_homes": []}

    class ProfileDB:
        def __init__(self, db_path=None):
            assert db_path is not None
            captured["db_path"] = Path(db_path)

    def make_agent(*_args, session_db=None, **_kwargs):
        captured["agent_home"] = server.get_hermes_home_override()
        captured["agent_db"] = session_db
        return object()

    def init_session(target_sid, key, agent, history, **kwargs):
        captured["init_home"] = server.get_hermes_home_override()
        captured["init_kwargs"] = kwargs
        if init_fails:
            raise RuntimeError("force minimal fallback")
        server._sessions[target_sid] = {
            "agent": agent,
            "session_key": key,
            "history": list(history),
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "profile_home": str(kwargs["profile_home"]),
        }

    def profile_config(value):
        captured["fallback_homes"].append(server.get_hermes_home_override())
        return value

    monkeypatch.setattr("hermes_state.SessionDB", ProfileDB)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_init_session", init_session)
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: profile_config(True))
    monkeypatch.setattr(
        server, "_load_tool_progress_mode", lambda: profile_config("verbose")
    )
    monkeypatch.setattr(server, "_resolve_session_source", lambda value: value or "tui")

    try:
        session = host._ensure_server_session(
            server,
            {
                "sid": sid,
                "session_key": "remote-key",
                "history": [{"role": "user", "content": "hello"}],
                "history_version": 3,
                "profile_home": str(profile_home),
                "source": "tui",
            },
        )

        assert captured["db_path"] == profile_home / "state.db"
        assert Path(captured["agent_home"]) == profile_home
        assert captured["agent_db"].__class__ is ProfileDB
        assert Path(captured["init_home"]) == profile_home
        assert captured["init_kwargs"]["session_db"] is captured["agent_db"]
        assert captured["init_kwargs"]["profile_home"] == str(profile_home)
        assert session["profile_home"] == str(profile_home)
        assert session["history_version"] == 3
        if init_fails:
            assert [Path(home) for home in captured["fallback_homes"]] == [
                profile_home,
                profile_home,
            ]
            assert session["show_reasoning"] is True
            assert session["tool_progress_mode"] == "verbose"
        else:
            assert captured["fallback_homes"] == []
        assert server.get_hermes_home_override() is None
    finally:
        server._sessions.pop(sid, None)
        host.close()


def test_compute_host_real_turn_end_serializes_authoritative_history(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    initial = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]
    session = {
        "agent": object(),
        "history": list(initial),
        "history_lock": threading.Lock(),
        "history_version": 6,
        "running": False,
        "session_key": "history-key",
    }

    def run_prompt(_rid, _sid, target, text):
        with target["history_lock"]:
            target["history"] = [
                *target["history"],
                {"role": "user", "content": text},
                {"role": "assistant", "content": "second answer"},
            ]
            target["history_version"] += 1
            target["running"] = False

    monkeypatch.setattr(host, "_ensure_server_session", lambda _server, _frame: session)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {"model": "test"})
    try:
        host._run_real_turn(
            {
                "sid": "history-sid",
                "request_id": "history-turn",
                "session_key": "history-key",
                "history": initial,
                # A persistent child can be authoritative at a newer version
                # than a stale parent frame. The CAS base must be the version
                # the child actually ran from (6), not this value.
                "history_version": 5,
                "text": "second",
            }
        )
        end = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "turn.end"
            and frame.get("request_id") == "history-turn",
        )
        # Parse the emitted line again: this is the real wire serialization,
        # not a shared in-memory list handed to the assertion.
        encoded = json.dumps(end)
        decoded = json.loads(encoded)
        assert decoded["base_history_version"] == 6
        assert decoded["history_version"] == 7
        assert [message["role"] for message in decoded["history"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert decoded["history"][-1]["content"] == "second answer"
    finally:
        host.close()


def test_compute_host_real_turn_waits_for_chained_successor_before_settling(
    monkeypatch,
):
    """turn.end must not snapshot until a chained successor turn settles.

    server._run_prompt_submit runs a turn on session["_run_thread"] and, in
    that thread's tail, can chain a successor (queued prompt / goal
    continuation / post-turn completion) by calling _run_prompt_submit again —
    installing a FRESH _run_thread and re-setting running=True before the first
    thread exits. The old compute-host code joined the first thread once and
    snapshotted immediately, racing the live successor: turn.end carried
    partial history and the source was marked idle mid-successor. This test
    installs+starts such a successor from the first dispatcher and asserts the
    host does not emit turn.end / snapshot history until the successor settles,
    and that the final history includes the successor's turn.
    """
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)

    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "session_key": "chain-key",
        "_run_thread": None,
    }

    first_dispatch_done = threading.Event()
    release_successor = threading.Event()
    first_may_proceed = threading.Event()

    class JoinObservedThread(threading.Thread):
        """Release the dispatcher only after the host starts joining it."""

        def join(self, timeout=None):
            first_may_proceed.set()
            return super().join(timeout)

    def run_prompt(_rid, _sid, target, text):
        # Emulate server._run_prompt_submit: start the turn on a background
        # thread, set session["_run_thread"], and return immediately.
        def _first_turn():
            # Hold session["_run_thread"] == first until the host has observed
            # and begun joining it, so the buggy single-join path deterministically
            # captures the FIRST thread (as it does in production, where the turn
            # runs before the tail swaps in a successor).
            first_may_proceed.wait(5)
            with target["history_lock"]:
                target["history"] = [
                    *target["history"],
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "first answer"},
                ]
                target["history_version"] += 1
                # The first turn releases running in its finally...
                target["running"] = False

            # ...then, exactly like _drain_queued_prompt -> _run_prompt_submit,
            # it claims running again and installs a SUCCESSOR thread that
            # replaces session["_run_thread"] BEFORE this first thread exits.
            def _successor_turn():
                release_successor.wait(5)
                with target["history_lock"]:
                    target["history"] = [
                        *target["history"],
                        {"role": "user", "content": "successor"},
                        {"role": "assistant", "content": "final answer"},
                    ]
                    target["history_version"] += 1
                    target["running"] = False

            with target["history_lock"]:
                target["running"] = True
            successor = threading.Thread(target=_successor_turn, daemon=True)
            target["_run_thread"] = successor
            successor.start()
            first_dispatch_done.set()

        first = JoinObservedThread(target=_first_turn, daemon=True)
        target["_run_thread"] = first
        first.start()

    monkeypatch.setattr(host, "_ensure_server_session", lambda _s, _f: session)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {"model": "test"})

    real_turn = threading.Thread(
        target=lambda: host._run_real_turn(
            {
                "sid": "chain-sid",
                "request_id": "chain-turn",
                "session_key": "chain-key",
                "text": "hello",
            }
        ),
        daemon=True,
    )
    try:
        real_turn.start()

        # Wait until the first dispatcher has installed + started the successor
        # and exited. The OLD code would already have joined the first thread
        # and emitted turn.end here.
        assert first_dispatch_done.wait(5)
        # Shutdown may stop waiting for the worker future, but it must not
        # manufacture a terminal frame from partial history while the chained
        # daemon thread still owns the turn.
        host._closed.set()
        # Give a buggy (settle-early) path a chance to wrongly emit turn.end.
        time.sleep(0.15)
        assert not any(
            frame.get("type") == "turn.end"
            and frame.get("request_id") == "chain-turn"
            for frame in _json_lines(out)
        ), "turn.end emitted before the chained successor settled"
        assert session["running"] is True

        # Let the successor finish; only now may the host settle.
        release_successor.set()
        end = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "turn.end"
            and frame.get("request_id") == "chain-turn",
        )
        # Final history includes BOTH the first turn and the successor turn,
        # and reflects the successor's version.
        decoded = json.loads(json.dumps(end))
        assert decoded["history_version"] == 2
        assert [message["role"] for message in decoded["history"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert decoded["history"][-1]["content"] == "final answer"
        assert decoded["message_count"] == 4
        assert decoded["interrupted"] is False
        real_turn.join(5)
        assert not real_turn.is_alive()
    finally:
        release_successor.set()
        real_turn.join(5)
        host.close()


def test_compute_host_terminal_barrier_spans_snapshot_through_turn_end_emit(
    monkeypatch,
):
    """No new child turn may claim the snapshot→turn.end serialization gap."""
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=2, heartbeat_secs=0)
    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "session_key": "barrier-key",
        "_run_thread": None,
    }
    emit_entered = threading.Event()
    release_emit = threading.Event()
    original_emit = host.emit

    def run_prompt(_rid, _sid, target, text):
        with target["history_lock"]:
            target["history"] = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "done"},
            ]
            target["history_version"] += 1
            target["running"] = False

    def blocking_emit(frame):
        if frame.get("type") == "turn.end" and frame.get("request_id") == "barrier-turn":
            assert session.get("_compute_host_terminal_pending") == "barrier-turn"
            emit_entered.set()
            assert release_emit.wait(5)
        original_emit(frame)

    monkeypatch.setattr(host, "_ensure_server_session", lambda _s, _f: session)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {"model": "test"})
    monkeypatch.setattr(host, "emit", blocking_emit)

    first = threading.Thread(
        target=lambda: host._run_real_turn(
            {
                "sid": "barrier-sid",
                "request_id": "barrier-turn",
                "session_key": "barrier-key",
                "text": "hello",
            }
        ),
        daemon=True,
    )
    try:
        first.start()
        assert emit_entered.wait(5)
        assert session["running"] is False
        assert session["_compute_host_terminal_pending"] == "barrier-turn"

        # A second parent request arriving before turn.end is written must see
        # the terminal barrier as busy rather than starting from the stale idle
        # snapshot.
        host._run_real_turn(
            {
                "sid": "barrier-sid",
                "request_id": "overlap-turn",
                "session_key": "barrier-key",
                "text": "overlap",
            }
        )
        overlap = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "turn.error"
            and frame.get("request_id") == "overlap-turn",
        )
        assert overlap["message"] == "session busy"
        assert session["history_version"] == 1

        release_emit.set()
        end = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "turn.end"
            and frame.get("request_id") == "barrier-turn",
        )
        assert end["history_version"] == 1
        first.join(5)
        assert not first.is_alive()
        assert "_compute_host_terminal_pending" not in session
    finally:
        release_emit.set()
        first.join(5)
        host.close()


def test_compute_host_interrupt_control_is_not_queued_behind_turn():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-slow",
                "prompt": "hello",
                "delta_count": 200,
                "delay_s": 0.01,
            }
        )
        _wait_for_frame(out, lambda f: f.get("type") == "delta" and f.get("request_id") == "turn-slow")

        host.handle_frame({"type": "interrupt", "sid": "alpha", "request_id": "stop-1"})
        ack = _wait_for_frame(out, lambda f: f.get("type") == "interrupt.ack" and f.get("request_id") == "stop-1")
        assert ack["applied"] is True

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-slow")
        assert end["interrupted"] is True
        typed = [f["type"] for f in _json_lines(out)]
        assert typed.index("interrupt.ack") < typed.index("turn.end")
    finally:
        host.close()


def test_compute_host_flushes_sessions_on_orphan_shutdown(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    session = {"session_key": "key"}
    calls: list[tuple[dict, str]] = []
    server._sessions["flush-sid"] = session
    monkeypatch.setattr(
        server,
        "_finalize_session",
        lambda sess, end_reason="tui_close": calls.append((sess, end_reason)),
    )
    try:
        host.flush_all_sessions(reason="orphan")
        assert calls == [(session, "compute_host_orphan")]
    finally:
        server._sessions.pop("flush-sid", None)
        host.close()


def test_compute_host_parent_guard_exits_when_parent_pid_changes(monkeypatch):
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    host._parent_pid = 111
    monkeypatch.setattr(os, "getppid", lambda: 222)

    def _exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _exit)

    with pytest.raises(SystemExit) as exc_info:
        host._parent_guard_loop()

    assert exc_info.value.code == 0
    orphan = next(frame for frame in _json_lines(out) if frame.get("type") == "orphan")
    assert orphan["old_ppid"] == 111
    assert orphan["ppid"] == 222
    assert isinstance(orphan["host_ns"], int)


def test_mutator_route_table_matches_prd_inventory():
    assert MUTATOR_ROUTE_TABLE == {
        "prompt.submit": "turn-path",
        "session.interrupt": "turn-path",
        "reload.mcp": "run-concurrent",
        "session.compress": "idle-gated",
        "prompt.submit.truncate": "idle-gated",
        "slash.model": "idle-gated",
        "slash.personality": "idle-gated",
        "slash.prompt": "idle-gated",
        "slash.compress": "idle-gated",
        "session.reset": "idle-gated",
        "session.history.reload": "idle-gated",
        "slash.retry": "idle-gated",
    }


def test_compute_host_compress_control_runs_identity_guard_in_host(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)

    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        context_compressor = None

    session = {
        "agent": _Agent(),
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }
    calls: dict[str, object] = {}

    def _compress(sess, focus_topic=None, **_kwargs):
        assert sess is session
        calls["compress_focus"] = focus_topic
        with sess["history_lock"]:
            sess["history"] = [{"role": "summary", "content": "compressed in host"}]
            sess["history_version"] = 3

    def _sync(sid, sess):
        assert sess is session
        calls["sync"] = sid
        sess["session_key"] = "after-key"

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _sync)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {
            "model": "host-model",
            "provider": "host-provider",
            "usage": {"total": 2},
        },
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-1",
                "route_name": "slash.compress",
                "command": "/compress focus",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == {"compress_focus": "focus", "sync": "sid"}
    assert ack["route_name"] == "slash.compress"
    assert ack["session_key"] == "after-key"
    assert ack["history_version"] == 3
    assert ack["message_count"] == 1
    assert ack["session_info"]["model"] == "host-model"


def test_append_log_record_single_write_lines(tmp_path):
    path = tmp_path / "agent.log"

    def writer(i: int) -> None:
        append_log_record(path, f"line-{i:03d}-" + ("x" * 2000))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert sorted(line.split("-", 2)[1] for line in lines) == [f"{i:03d}" for i in range(32)]
    assert all(line.endswith("x" * 2000) for line in lines)


def test_supervisor_startup_reconcile_pid_reuse_guard(tmp_path, monkeypatch):
    registry = tmp_path / "dashboard-compute-host.json"
    registry.write_text(json.dumps({"host_pid": os.getpid(), "boot_id": "stale"}), encoding="utf-8")

    killed: list[int] = []
    supervisor = HostSupervisor(registry_path=registry, argv=[sys.executable, "-c", ""], autostart=False)
    monkeypatch.setattr(supervisor, "_pid_matches_compute_host", lambda _pid: False)
    monkeypatch.setattr(supervisor, "_terminate_pid", lambda pid, **_kw: killed.append(pid))

    result = supervisor.reconcile_startup_orphan()

    assert result == "pid-reuse-ignored"
    assert killed == []
    assert not registry.exists()


def test_supervisor_crash_emits_turn_error_and_respawns(tmp_path):
    script = tmp_path / "fake_host.py"
    script.write_text(
        """
import json, os, sys
print(json.dumps({'type':'hello','host_pid':os.getpid(),'boot_id':'boot-1','build_sha':'test','hermes_home':os.environ.get('HERMES_HOME','')}), flush=True)
for raw in sys.stdin:
    frame=json.loads(raw)
    if frame.get('type') == 'shutdown':
        print(json.dumps({'type':'shutdown.ack','request_id':frame.get('request_id')}), flush=True)
        break
    if frame.get('type') == 'turn.start':
        print(json.dumps({'type':'turn.started','sid':frame.get('sid'),'request_id':frame.get('request_id')}), flush=True)
        sys.stdout.flush()
        os._exit(7)
""".strip(),
        encoding="utf-8",
    )
    registry = tmp_path / "dashboard-compute-host.json"
    completions: list[dict] = []
    rpc_events: list[dict] = []
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, str(script)],
        rpc_sink=rpc_events.append,
        respawn_max=2,
        heartbeat_secs=1,
        expected_build_sha="test",
        autostart=False,
    )
    try:
        supervisor.start()
        supervisor.submit_turn(
            {"type": "turn.start", "sid": "sid-1", "request_id": "turn-1", "text": "hello"},
            on_complete=completions.append,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not completions:
            time.sleep(0.02)
        assert completions, "host crash did not complete pending turn"
        assert completions[0]["type"] == "turn.error"
        assert completions[0]["reason"] == "crash"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not supervisor.is_running():
            time.sleep(0.02)
        assert supervisor.is_running()
    finally:
        supervisor.shutdown()


def test_supervisor_pipe_proxies_detached_interaction_bidirectionally(
    tmp_path, monkeypatch
):
    """Exercise the server proxy through real supervisor stdin/stdout pipes."""
    from tui_gateway import server

    script = tmp_path / "interactive_host.py"
    script.write_text(
        """
import json, os, sys
print(json.dumps({'type':'hello','host_pid':os.getpid(),'boot_id':'interactive','build_sha':'test','hermes_home':os.environ.get('HERMES_HOME','')}), flush=True)
turn = None
for raw in sys.stdin:
    frame = json.loads(raw)
    if frame.get('type') == 'shutdown':
        print(json.dumps({'type':'shutdown.ack','request_id':frame.get('request_id')}), flush=True)
        break
    if frame.get('type') == 'turn.start':
        turn = frame
        print(json.dumps({
            'type':'interactive.request',
            'sid':frame.get('sid'),
            'source_sid':frame.get('sid'),
            'request_id':'child-clarify-1',
            'event':'clarify.request',
            'payload':{'request_id':'child-clarify-1','question':'from child','choices':[]},
        }), flush=True)
    if frame.get('type') == 'interactive.response':
        assert frame.get('sid') == turn.get('sid')
        assert frame.get('interactive_request_id') == 'child-clarify-1'
        assert frame.get('method') == 'clarify.respond'
        assert frame.get('params', {}).get('answer') == 'through pipe'
        print(json.dumps({
            'type':'interactive.response.ack',
            'sid':frame.get('sid'),
            'request_id':frame.get('request_id'),
            'interactive_request_id':'child-clarify-1',
            'response':{'jsonrpc':'2.0','id':frame.get('request_id'),'result':{'status':'ok'}},
        }), flush=True)
        print(json.dumps({
            'type':'interactive.complete',
            'sid':frame.get('sid'),
            'source_sid':frame.get('sid'),
            'request_id':'child-clarify-1',
            'event':'clarify.request',
            'reason':'answered',
        }), flush=True)
        print(json.dumps({
            'type':'turn.end',
            'sid':turn.get('sid'),
            'request_id':turn.get('request_id'),
            'history_version':1,
            'message_count':0,
        }), flush=True)
""".strip(),
        encoding="utf-8",
    )
    registry = tmp_path / "dashboard-compute-host.json"
    emitted: list[tuple[str, str, dict]] = []
    completed: list[dict] = []
    source = {
        "detached_turn_task_id": "pipe-task",
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "pipe-source-key",
    }
    owner = {
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": "pipe-owner-key",
    }
    server._sessions.update({"pipe-source": source, "pipe-owner": owner})
    server._detached_turns["pipe-task"] = {
        "owner_session_id": "pipe-owner",
        "source_session_id": "pipe-source",
        "status": "running",
        "task_id": "pipe-task",
    }
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append(
            (event, sid, dict(payload or {}))
        )
        or True,
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, str(script)],
        interactive_sink=server._compute_host_interactive_sink,
        respawn_max=0,
        heartbeat_secs=1,
        expected_build_sha="test",
        autostart=False,
    )
    old_supervisor = server._compute_host_supervisor
    server._compute_host_supervisor = supervisor
    try:
        supervisor.submit_turn(
            {
                "sid": "pipe-source",
                "request_id": "pipe-turn",
                "text": "hello",
            },
            on_complete=completed.append,
        )
        deadline = time.monotonic() + 3
        while not emitted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert emitted
        event, presentation_sid, payload = emitted[0]
        assert (event, presentation_sid) == ("clarify.request", "pipe-owner")
        assert payload["request_id"] == "host:pipe-source:child-clarify-1"

        denied = server._methods["clarify.respond"](
            "wrong-owner",
            {
                "answer": "wrong",
                "request_id": payload["request_id"],
                "session_id": "pipe-source",
            },
        )
        assert denied["error"]["code"] == 4043
        assert completed == []

        accepted = server._methods["clarify.respond"](
            "right-owner",
            {
                "answer": "through pipe",
                "request_id": payload["request_id"],
                "session_id": "pipe-owner",
            },
        )
        assert accepted["result"]["status"] == "ok"
        deadline = time.monotonic() + 3
        while not completed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert completed and completed[0]["type"] == "turn.end"
        with server._prompt_lock:
            assert payload["request_id"] not in server._compute_host_interactions
            assert server._interactive_prompt_queues == {}
    finally:
        supervisor.shutdown()
        server._compute_host_supervisor = old_supervisor
        server._sessions.pop("pipe-source", None)
        server._sessions.pop("pipe-owner", None)
        server._detached_turns.pop("pipe-task", None)
        with server._prompt_lock:
            server._compute_host_interactions.clear()
            server._pending_prompt_payloads.clear()
            server._pending_prompt_presentations.clear()
            server._interactive_prompt_queues.clear()
