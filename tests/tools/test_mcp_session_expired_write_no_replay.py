"""Tests for the write-safety carve-out in session-expired retry (gcal-mcp
stabilization, 2026-08-28).

``_handle_session_expired_and_retry`` reconnects the transport and replays
the failed call once. That is safe for read-only tools (idempotent), but a
session-expired error (broken pipe / closed stream / EOF) can also fire
*after* a write already reached the remote server, just before its response
made it back over the wire. Blindly replaying a ``create_event``-style call
in that window could duplicate the write. Tools without a discovery-time
``readOnlyHint: true`` annotation (fail-closed default: unknown metadata is
write-capable, matching ``_annotation_read_only_hint``) must therefore never
be transparently replayed — only the transport gets reconnected, and the
caller is told the outcome is unknown.
"""
import json
import threading
from unittest.mock import MagicMock

import pytest


def _install_stub_server(name: str):
    """Minimal server stub: reconnect swaps in a fresh session + fires ready."""
    from tools import mcp_tool

    mcp_tool._ensure_mcp_loop()

    server = MagicMock()
    server.name = name

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    server._ready = _ReadyAdapter()

    class _EventAdapter:
        def set(self):
            old_session = server.session
            new_session = MagicMock()
            if hasattr(old_session, "call_tool"):
                new_session.call_tool = old_session.call_tool
            server.session = new_session
            ready_flag.set()

    server._reconnect_event = _EventAdapter()
    server.session = MagicMock()
    mcp_tool._servers[name] = server
    return server


# ---------------------------------------------------------------------------
# _handle_session_expired_and_retry(retryable=...) — unit coverage
# ---------------------------------------------------------------------------


def test_non_retryable_reconnects_but_does_not_replay(monkeypatch, tmp_path):
    """retryable=False must still heal the transport, but must NOT call
    retry_call — and the returned error must flag the outcome as unknown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _handle_session_expired_and_retry

    server = _install_stub_server("gcal-write")
    call_count = {"n": 0}

    def _retry_call():
        call_count["n"] += 1
        return '{"result": "created"}'

    try:
        out = _handle_session_expired_and_retry(
            "gcal-write",
            RuntimeError("Session terminated"),
            _retry_call,
            "tools/call create_event",
            retryable=False,
        )
        assert call_count["n"] == 0, (
            "a non-read-only tool must never have its call replayed after "
            "a session-expired transport error"
        )
        assert out is not None
        parsed = json.loads(out)
        assert parsed.get("error"), parsed
        assert parsed.get("uncertain_outcome") is True, parsed
        # Transport must still have been reconnected so the *next* call works.
        assert server._ready.is_set()
    finally:
        mcp_tool._servers.pop("gcal-write", None)


def test_retryable_default_still_replays(monkeypatch, tmp_path):
    """Sanity guard: the default (retryable=True, existing read-path
    behavior) is unchanged by the new parameter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _handle_session_expired_and_retry

    server = _install_stub_server("gcal-read")
    call_count = {"n": 0}

    def _retry_call():
        call_count["n"] += 1
        return '{"result": "ok"}'

    try:
        out = _handle_session_expired_and_retry(
            "gcal-read",
            RuntimeError("Session terminated"),
            _retry_call,
            "resources/list",
        )
        assert call_count["n"] == 1
        assert json.loads(out) == {"result": "ok"}
    finally:
        mcp_tool._servers.pop("gcal-read", None)


# ---------------------------------------------------------------------------
# _make_tool_handler wiring — readOnlyHint drives retryable end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "read_only_hint, expect_replay",
    [
        (True, True),
        (False, False),
        (None, False),  # missing hint fails closed to write-capable
    ],
)
def test_tool_handler_replay_gated_by_read_only_hint(
    monkeypatch, tmp_path, read_only_hint, expect_replay,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    server_name = f"gcal-hint-{read_only_hint}"
    server = _install_stub_server(server_name)
    mcp_tool._server_error_counts.pop(server_name, None)
    mcp_tool._server_breaker_opened_at.pop(server_name, None)

    if read_only_hint is None:
        mcp_tool._tool_read_only_hints.pop(server_name, None)
    else:
        mcp_tool._tool_read_only_hints[server_name] = {
            "create_event": read_only_hint,
        }

    call_count = {"n": 0}

    async def _call_tool(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Broken pipe")
        result = MagicMock()
        result.is_error = False
        result.content = [MagicMock(type="text", text="second-call-result")]
        result.structured_content = None
        return result

    server.session.call_tool = _call_tool

    try:
        handler = _make_tool_handler(server_name, "create_event", 10.0)
        out = json.loads(handler({}))
        if expect_replay:
            assert call_count["n"] == 2, "read-only tool should be replayed once"
            assert out == {"result": "second-call-result"}
        else:
            assert call_count["n"] == 1, (
                "write-capable tool must not be replayed after a "
                "session-expired transport error"
            )
            assert out.get("uncertain_outcome") is True, out
    finally:
        mcp_tool._servers.pop(server_name, None)
        mcp_tool._tool_read_only_hints.pop(server_name, None)
        mcp_tool._server_error_counts.pop(server_name, None)
        mcp_tool._server_breaker_opened_at.pop(server_name, None)
