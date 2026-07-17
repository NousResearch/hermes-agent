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
from types import SimpleNamespace
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
        yield mod
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
    "session.active_list",   # live-session rehydrate — in-memory registry
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
    assert fast_elapsed < 2.0, (
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
    assert elapsed < 2.0, (
        f"prompt.submit blocked for {elapsed:.2f}s behind slow pet.info — "
        f"the user's message would appear stuck under GIL pressure (#50005)."
    )

    released.set()


def test_on_demand_model_options_does_not_block_inline_requests(server):
    """Desktop model pickers invoke model.options on demand, but its provider,
    config, and model-catalog I/O must still run outside the dispatch thread.
    """
    released = threading.Event()

    def blocked_model_options(rid, params):
        released.wait(timeout=1)
        return server._ok(rid, {"providers": []})

    server._methods["model.options"] = blocked_model_options
    server._methods["fast.check"] = lambda rid, params: server._ok(rid, {"ok": True})

    try:
        t0 = time.monotonic()
        assert server.dispatch(
            {"id": "models", "method": "model.options", "params": {}}
        ) is None

        fast_resp = server.dispatch(
            {"id": "fast", "method": "fast.check", "params": {}}
        )
        fast_elapsed = time.monotonic() - t0

        assert fast_resp["result"] == {"ok": True}
        assert fast_elapsed < 0.5
    finally:
        released.set()


def test_model_options_pool_uses_request_time_runtime_snapshot(server, monkeypatch):
    """A queued model-options read must report one coherent pre-dispatch runtime."""
    from hermes_cli.inventory import ConfigContext

    agent = SimpleNamespace(
        model="request-model",
        provider="request-provider",
        base_url="https://request.invalid/v1",
    )
    server._sessions["session-1"] = {"agent": agent}

    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: ConfigContext(
            current_provider="disk-provider",
            current_model="disk-model",
            current_base_url="https://disk.invalid/v1",
            user_providers={},
            custom_providers=[],
        ),
    )

    def build_payload(ctx, **_kwargs):
        return {
            "providers": [],
            "model": ctx.current_model,
            "provider": ctx.current_provider,
            "base_url": ctx.current_base_url,
        }

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", build_payload)

    worker_entered = threading.Event()
    release_worker = threading.Event()
    response_written = threading.Event()
    original_handle_request = server.handle_request

    def blocked_handle_request(request):
        if request.get("method") == "model.options":
            worker_entered.set()
            if not release_worker.wait(timeout=2):
                raise TimeoutError("model.options worker was not released")
        return original_handle_request(request)

    monkeypatch.setattr(server, "handle_request", blocked_handle_request)

    class CaptureTransport:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)
            response_written.set()
            return True

        def close(self):
            pass

    transport = CaptureTransport()
    request = {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1", "explicit_only": True},
    }

    try:
        assert server.dispatch(request, transport) is None
        assert worker_entered.wait(timeout=1)

        agent.model = "half-switched-model"
        agent.provider = "half-switched-provider"
        agent.base_url = "https://half-switched.invalid/v1"
        release_worker.set()

        assert response_written.wait(timeout=2)
    finally:
        release_worker.set()

    assert request == {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1", "explicit_only": True},
    }
    assert transport.frames == [
        {
            "jsonrpc": "2.0",
            "id": "models",
            "result": {
                "providers": [],
                "model": "request-model",
                "provider": "request-provider",
                "base_url": "https://request.invalid/v1",
            },
        }
    ]


def test_model_options_pool_snapshot_preserves_custom_provider_identity(server, monkeypatch):
    """The worker's frozen runtime keeps the picker on its canonical custom row."""
    from hermes_cli.inventory import ConfigContext

    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: ConfigContext(
            current_provider="custom:configured-provider",
            current_model="disk-model",
            current_base_url="https://disk.invalid/v1",
            user_providers={},
            custom_providers=[],
        ),
    )
    canonical = MagicMock(return_value="custom:request-provider")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        canonical,
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.build_models_payload",
        lambda ctx, **_kwargs: {
            "providers": [],
            "model": ctx.current_model,
            "provider": ctx.current_provider,
            "base_url": ctx.current_base_url,
        },
    )

    response = server._methods["model.options"](
        "custom",
        {
            server._MODEL_OPTIONS_RUNTIME_SNAPSHOT: {
                "model": "request-model",
                "provider": "custom",
                "base_url": "https://request.invalid/v1",
            },
        },
    )

    assert response["result"] == {
        "providers": [],
        "model": "request-model",
        "provider": "custom:request-provider",
        "base_url": "https://request.invalid/v1",
    }
    canonical.assert_called_once_with(
        base_url="https://request.invalid/v1",
        config_provider="custom:configured-provider",
    )
def test_model_options_pool_snapshots_active_fallback_runtime(server, monkeypatch):
    """A queued picker read must report the live fallback, not its primary."""
    agent = SimpleNamespace(
        model="fallback-model",
        provider="fallback-provider",
        base_url="https://fallback.invalid/v1",
        _primary_runtime={
            "model": "primary-model",
            "provider": "primary-provider",
            "base_url": "https://primary.invalid/v1",
        },
    )
    server._sessions["session-1"] = {"agent": agent}

    def echo_runtime_snapshot(rid, params):
        return server._ok(rid, params[server._MODEL_OPTIONS_RUNTIME_SNAPSHOT])

    monkeypatch.setitem(server._methods, "model.options", echo_runtime_snapshot)

    worker_entered = threading.Event()
    release_worker = threading.Event()
    response_written = threading.Event()
    original_handle_request = server.handle_request

    def blocked_handle_request(request):
        if request.get("method") == "model.options":
            worker_entered.set()
            if not release_worker.wait(timeout=2):
                raise TimeoutError("model.options worker was not released")
        return original_handle_request(request)

    monkeypatch.setattr(server, "handle_request", blocked_handle_request)

    class CaptureTransport:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)
            response_written.set()
            return True

        def close(self):
            pass

    transport = CaptureTransport()
    request = {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1"},
    }

    try:
        assert server.dispatch(request, transport) is None
        assert worker_entered.wait(timeout=1)

        agent.model = "later-model"
        agent.provider = "later-provider"
        agent.base_url = "https://later.invalid/v1"
        release_worker.set()

        assert response_written.wait(timeout=2)
    finally:
        release_worker.set()

    assert request == {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1"},
    }
    assert transport.frames == [
        {
            "jsonrpc": "2.0",
            "id": "models",
            "result": {
                "model": "fallback-model",
                "provider": "fallback-provider",
                "base_url": "https://fallback.invalid/v1",
            },
        }
    ]


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
