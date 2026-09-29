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
    # Mocks are scoped to the initial import only — keeping them active for
    # the whole test would poison modules first imported inside test bodies
    # (see tests/tui_gateway/test_protocol.py for the full rationale).
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")

    # Tests below stub handlers ("session.list", "prompt.submit", ...) in
    # the module-level _methods dict shared with every other test file in
    # the process — snapshot and restore it around each test.
    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


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


class _CaptureTransport:
    def __init__(self):
        self.frames = []
        self.written = threading.Event()

    def write(self, frame):
        self.frames.append(frame)
        self.written.set()
        return True

    def close(self):
        pass


def _dispatch_behind_blocked_worker(server, monkeypatch, request, mutate):
    """Dispatch ``request`` onto the RPC pool, then run ``mutate()`` only after the pool worker
    has picked the request up — the queued-worker race the request-time runtime snapshot
    freezes against (#65388)."""
    worker_entered = threading.Event()
    release_worker = threading.Event()
    original_handle = server._handle_admitted_request

    def blocked_handle_admitted_request(req):
        if req.get("method") == "model.options":
            worker_entered.set()
            if not release_worker.wait(timeout=2):
                raise TimeoutError("model.options worker was not released")
        return original_handle(req)

    monkeypatch.setattr(server, "_handle_admitted_request", blocked_handle_admitted_request)

    transport = _CaptureTransport()
    try:
        assert server.dispatch(request, transport) is None
        assert worker_entered.wait(timeout=1)
        mutate()
        release_worker.set()
        assert transport.written.wait(timeout=2)
    finally:
        release_worker.set()
    return transport


def _stub_picker_context(monkeypatch, reported):
    """Disk config reports a third identity; ``reported`` captures the runtime the payload
    builder actually received, base_url included (it never rides the result contract)."""
    from hermes_cli.inventory import ConfigContext

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
        reported["runtime"] = (ctx.current_model, ctx.current_provider, ctx.current_base_url)
        return {"providers": [], "model": ctx.current_model, "provider": ctx.current_provider}

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", build_payload)


def test_model_options_pool_uses_request_time_runtime_snapshot(server, monkeypatch):
    """A queued model-options read must report one coherent pre-dispatch runtime (#65388)."""
    agent = SimpleNamespace(
        model="request-model",
        provider="request-provider",
        base_url="https://request.invalid/v1",
    )
    server._sessions["session-1"] = {"agent": agent}

    reported: dict = {}
    _stub_picker_context(monkeypatch, reported)

    request = {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1", "explicit_only": True},
    }

    def _switch_after_queue():
        agent.model = "half-switched-model"
        agent.provider = "half-switched-provider"
        agent.base_url = "https://half-switched.invalid/v1"

    transport = _dispatch_behind_blocked_worker(server, monkeypatch, request, _switch_after_queue)

    # The freeze copies the request: the queued switch must not rewrite what the client sent.
    assert request == {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1", "explicit_only": True},
    }
    assert transport.frames == [
        {
            "jsonrpc": "2.0",
            "id": "models",
            "result": {"providers": [], "model": "request-model", "provider": "request-provider"},
        }
    ]
    assert reported["runtime"] == ("request-model", "request-provider", "https://request.invalid/v1")


def test_model_options_pool_snapshots_active_fallback_runtime(server, monkeypatch):
    """A queued picker read must report the live fallback runtime — never the preferred primary
    parked in ``_primary_runtime`` for next-turn restore (#65388)."""
    agent = SimpleNamespace(
        model="fallback-model",
        provider="fallback-provider",
        base_url="https://fallback.invalid/v1",
        _fallback_activated=True,
        _primary_runtime={
            "model": "primary-model",
            "provider": "primary-provider",
            "base_url": "https://primary.invalid/v1",
        },
    )
    server._sessions["session-1"] = {"agent": agent}

    reported: dict = {}
    _stub_picker_context(monkeypatch, reported)

    request = {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1"},
    }

    def _switch_after_queue():
        agent.model = "later-model"
        agent.provider = "later-provider"
        agent.base_url = "https://later.invalid/v1"

    transport = _dispatch_behind_blocked_worker(server, monkeypatch, request, _switch_after_queue)

    assert request == {
        "id": "models",
        "method": "model.options",
        "params": {"session_id": "session-1"},
    }
    assert transport.frames == [
        {
            "jsonrpc": "2.0",
            "id": "models",
            "result": {"providers": [], "model": "fallback-model", "provider": "fallback-provider"},
        }
    ]
    assert reported["runtime"] == ("fallback-model", "fallback-provider", "https://fallback.invalid/v1")
