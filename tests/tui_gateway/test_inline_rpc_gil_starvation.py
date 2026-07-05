"""Tests for tui_gateway inline-RPC pool routing under GIL pressure (#50005).

The WS read loop in ``handle_ws()`` processes requests sequentially via
``await asyncio.to_thread(server.dispatch, req, transport)``. Inline handlers
(NOT in ``_LONG_HANDLERS``) run ``handle_request()`` synchronously inside
``dispatch()``, blocking the loop from reading the next request. Under GIL
pressure from multiple concurrent agent turns, even lightweight RPCs like
``session.list`` and ``pet.info`` can take seconds, causing frontend requests
to time out (120s) and the WebSocket to disconnect — the false "needs setup"
failure mode (#50005).

The fix routes all frontend-polled RPCs through ``_LONG_HANDLERS`` so
``dispatch()`` returns immediately (``_pool.submit`` + ``return None``) and
the WS read loop is never blocked.
"""

import io
import json
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

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
        methods = dict(mod._methods)
        real_stdout = mod._real_stdout
        try:
            yield mod
        finally:
            mod._methods.clear()
            mod._methods.update(methods)
            mod._real_stdout = real_stdout
            mod._sessions.clear()
            mod._pending.clear()
            mod._answers.clear()


@pytest.fixture()
def capture(server):
    """Redirect server's real stdout to a StringIO and return (server, buf)."""
    buf = io.StringIO()
    server._real_stdout = buf
    return server, buf


# ─── RPCs that must be in _LONG_HANDLERS ────────────────────────────────

# These are polled by the Desktop frontend. Before the fix they ran inline,
# blocking the WS read loop under GIL pressure and causing false "needs setup"
# (#50005). Each one does I/O (DB query, file read, network) that can take
# seconds when the GIL is contended by concurrent agent turns.

FRONTEND_POLLED_RPCS = [
    "session.list",          # loads session list — SQLite query
    "pet.info",              # petdex poll — file/network read
    "process.list",          # background process status — process registry scan
    "setup.runtime_check",   # runtime readiness — resolve_runtime_provider() I/O
    "setup.status",          # provider configured check — config/credential scan
]


@pytest.mark.parametrize("method", FRONTEND_POLLED_RPCS)
def test_frontend_polled_rpc_is_pool_routed(server, method):
    """Every frontend-polled RPC must be in _LONG_HANDLERS so dispatch()
    returns immediately and the WS read loop is not blocked (#50005)."""
    assert method in server._LONG_HANDLERS, (
        f"{method!r} is not in _LONG_HANDLERS — it will block the WS read "
        f"loop under GIL pressure, causing false 'needs setup' (#50005)."
    )


def test_dispatch_inline_rpc_does_not_block_under_gil_pressure(server):
    """A slow inline-turned-long handler must not prevent a concurrent fast
    handler from completing. This is the core invariant: dispatch() must
    return immediately for _LONG_HANDLERS so the WS read loop stays free.

    Simulates the GIL-pressure scenario from #50005: a slow handler (mimicking
    a session.list query under GIL contention) must not block a fast handler
    (mimicking setup.runtime_check).
    """
    released = threading.Event()

    def slow_session_list(rid, params):
        released.wait(timeout=5)
        return server._ok(rid, {"sessions": []})

    server._methods["session.list"] = slow_session_list
    server._methods["fast.check"] = lambda rid, params: server._ok(rid, {"ok": True})

    t0 = time.monotonic()
    # session.list is in _LONG_HANDLERS → dispatch returns None immediately
    assert server.dispatch({"id": "slow", "method": "session.list", "params": {}}) is None

    # fast.check is inline → dispatch runs it synchronously and returns the result
    fast_resp = server.dispatch({"id": "fast", "method": "fast.check", "params": {}})
    fast_elapsed = time.monotonic() - t0

    assert fast_resp["result"] == {"ok": True}
    assert fast_elapsed < 0.5, (
        f"fast handler blocked for {fast_elapsed:.2f}s behind slow session.list — "
        f"the WS read loop would stall, causing false 'needs setup' (#50005)."
    )

    released.set()


def test_dispatch_pet_info_does_not_block_prompt_submit(server):
    """pet.info (polled every few seconds by the Desktop petdex) must not
    block prompt.submit. Before the fix, pet.info ran inline and a slow
    pet.info under GIL pressure delayed prompt.submit until the 120s RPC
    timeout fired (#50005).
    """
    released = threading.Event()

    def slow_pet_info(rid, params):
        released.wait(timeout=5)
        return server._ok(rid, {"pet": "cat"})

    server._methods["pet.info"] = slow_pet_info
    server._methods["prompt.submit"] = lambda rid, params: server._ok(rid, {"status": "streaming"})

    t0 = time.monotonic()
    assert server.dispatch({"id": "pet", "method": "pet.info", "params": {}}) is None

    # prompt.submit is inline (it spawns its own thread) — should return immediately
    resp = server.dispatch({"id": "prompt", "method": "prompt.submit", "params": {}})
    elapsed = time.monotonic() - t0

    assert resp["result"] == {"status": "streaming"}
    assert elapsed < 0.5, (
        f"prompt.submit blocked for {elapsed:.2f}s behind slow pet.info — "
        f"the user's message would appear stuck under GIL pressure (#50005)."
    )

    released.set()


def test_rpc_pool_workers_supports_concurrent_long_handlers(server):
    """The RPC thread pool must have enough workers to handle concurrent
    long handlers without queueing. With 6+ frontend-polled RPCs added to
    _LONG_HANDLERS, the default 4 workers can be exhausted when multiple
    agent turns are running. The pool must be at least 8."""
    assert server._rpc_pool_workers >= 8, (
        f"_rpc_pool_workers is {server._rpc_pool_workers}, expected >= 8. "
        f"Frontend-polled RPCs added to _LONG_HANDLERS need more workers to "
        f"avoid queueing under multi-agent load (#50005)."
    )
