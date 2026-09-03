"""Behavior tests for the public MCP connection-status snapshot."""

import json
import threading
import time
from types import SimpleNamespace


def test_connection_status_snapshot_is_closed_ordered_and_non_sensitive(monkeypatch):
    import tools.mcp_tool as mcp_tool

    secret = "Bearer super-secret-token"
    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {
            "zeta": {"url": "https://secret.example/mcp", "headers": {"Authorization": secret}},
            "alpha": {"command": "secret-command", "args": [secret], "enabled": False},
            "reloading": {"url": "https://private.invalid/mcp"},
            "failed": {"transport": "sse", "url": "https://private.invalid/sse"},
            "idle": {"command": "private-command"},
        },
    )
    ready = threading.Event()
    ready.set()
    connected = SimpleNamespace(
        session=object(),
        _ready=ready,
        _reconnecting=False,
        _registered_tool_names=["zeta__one", "zeta__two"],
        _error=None,
        _was_parked=False,
    )
    reconnecting_ready = threading.Event()
    reconnecting = SimpleNamespace(
        session=None,
        _ready=reconnecting_ready,
        _reconnecting=True,
        _registered_tool_names=[],
        _error=None,
        _was_parked=False,
    )
    mcp_tool._ensure_mcp_loop()

    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_connecting = set(mcp_tool._server_connecting)
        saved_errors = dict(mcp_tool._server_connect_errors)
        mcp_tool._servers.clear()
        mcp_tool._servers.update({"zeta": connected, "reloading": reconnecting})
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()
        mcp_tool._server_connect_errors["failed"] = secret

    try:
        snapshot = mcp_tool.get_mcp_connection_status()
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._server_connecting.clear()
            mcp_tool._server_connecting.update(saved_connecting)
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connect_errors.update(saved_errors)

    assert snapshot == [
        {"name": "alpha", "state": "disabled", "transport": "stdio"},
        {"name": "failed", "state": "failed", "transport": "sse", "error_code": "connection_failed"},
        {"name": "idle", "state": "unknown", "transport": "stdio"},
        {"name": "reloading", "state": "connecting", "transport": "streamable_http"},
        {"name": "zeta", "state": "connected", "transport": "streamable_http", "tool_count": 2},
    ]
    serialized = json.dumps(snapshot)
    assert secret not in serialized
    assert "secret-command" not in serialized
    assert "secret.example" not in serialized


def test_connection_status_reads_server_lifecycle_only_on_mcp_loop(monkeypatch):
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {"guarded": {"command": "private-command"}},
    )
    mcp_tool._ensure_mcp_loop()

    async def _thread_id():
        return threading.get_ident()

    owner_thread = mcp_tool._run_on_mcp_loop(_thread_id)

    class _OwnerOnlyReady:
        def is_set(self):
            assert threading.get_ident() == owner_thread
            return True

    class _OwnerOnlyServer:
        def __getattribute__(self, name):
            if name in {
                "session",
                "_ready",
                "_error",
                "_was_parked",
                "_registered_tool_names",
            }:
                assert threading.get_ident() == owner_thread
            return object.__getattribute__(self, name)

        session = object()
        _ready = _OwnerOnlyReady()
        _error = None
        _was_parked = False
        _registered_tool_names = ["guarded__callable"]

    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        mcp_tool._servers.clear()
        mcp_tool._servers["guarded"] = _OwnerOnlyServer()

    try:
        assert mcp_tool.get_mcp_connection_status() == [
            {
                "name": "guarded",
                "state": "connected",
                "transport": "stdio",
                "tool_count": 1,
            }
        ]
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)


def test_recovered_live_server_ignores_latched_reconnecting_flag(monkeypatch):
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {"recovered": {"url": "https://private.invalid/mcp"}},
    )
    ready = threading.Event()
    ready.set()
    recovered = SimpleNamespace(
        session=object(),
        _ready=ready,
        _reconnecting=True,
        _error=None,
        _was_parked=False,
        _registered_tool_names=["recovered__one", "recovered__two"],
    )
    mcp_tool._ensure_mcp_loop()
    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_connecting = set(mcp_tool._server_connecting)
        saved_errors = dict(mcp_tool._server_connect_errors)
        mcp_tool._servers.clear()
        mcp_tool._servers["recovered"] = recovered
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()

    try:
        assert mcp_tool.get_mcp_connection_status() == [
            {
                "name": "recovered",
                "state": "connected",
                "transport": "streamable_http",
                "tool_count": 2,
            }
        ]
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._server_connecting.clear()
            mcp_tool._server_connecting.update(saved_connecting)
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connect_errors.update(saved_errors)


def test_parked_after_live_failure_is_failed_not_unknown(monkeypatch):
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {"parked": {"command": "private-command"}},
    )
    stale_ready = threading.Event()
    stale_ready.set()
    parked = SimpleNamespace(
        session=None,
        _ready=stale_ready,
        _reconnecting=False,
        _error=None,
        _was_parked=True,
        _registered_tool_names=[],
    )
    mcp_tool._ensure_mcp_loop()
    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_connecting = set(mcp_tool._server_connecting)
        saved_errors = dict(mcp_tool._server_connect_errors)
        mcp_tool._servers.clear()
        mcp_tool._servers["parked"] = parked
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()

    try:
        assert mcp_tool.get_mcp_connection_status() == [
            {
                "name": "parked",
                "state": "failed",
                "transport": "stdio",
                "error_code": "connection_failed",
            }
        ]
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._server_connecting.clear()
            mcp_tool._server_connecting.update(saved_connecting)
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connect_errors.update(saved_errors)


def test_connection_status_state_precedence_table():
    import tools.mcp_tool as mcp_tool

    cases = [
        ("live_disabled", False, {"live": True}, "disabled"),
        ("live_with_error", True, {"live": True, "error": True}, "connected"),
        (
            "connecting_with_error",
            True,
            {"connecting": True, "error": True},
            "failed",
        ),
        ("disabled_with_error", False, {"error": True}, "disabled"),
        (
            "reconnecting_with_stale_ready",
            True,
            {"present": True, "ready": True, "reconnecting": True},
            "connecting",
        ),
        (
            "parked_after_live_failure",
            True,
            {"present": True, "ready": True, "parked": True},
            "failed",
        ),
        (
            "active_post_live_retry_with_stale_ready",
            True,
            {"present": True, "ready": True},
            "connecting",
        ),
    ]

    for case, enabled, runtime, expected in cases:
        assert mcp_tool._mcp_connection_state(enabled, runtime) == expected, case


def test_connection_status_rejects_same_mcp_loop_call_without_stalling(monkeypatch):
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {"same-loop": {"command": "private-command"}},
    )
    mcp_tool._ensure_mcp_loop()

    async def _call_status_on_owner_loop():
        started = time.monotonic()
        try:
            mcp_tool.get_mcp_connection_status()
        except RuntimeError as exc:
            return time.monotonic() - started, str(exc)
        raise AssertionError("same-loop call unexpectedly succeeded")

    elapsed, message = mcp_tool._run_on_mcp_loop(
        _call_status_on_owner_loop,
        timeout=2,
    )

    assert elapsed < 0.5
    assert message == (
        "MCP connection status cannot be read synchronously from the MCP event loop"
    )
