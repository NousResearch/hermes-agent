"""Test that start_server configures ws-ping keepalive.

The server now uses uvicorn.Server directly (not uvicorn.run) so we stub
Config + Server + asyncio.run to capture kwargs without starting an event loop.
"""

import asyncio
import contextlib
import signal

import uvicorn

from hermes_cli import web_server


def _stub_uvicorn(monkeypatch):
    """Replace uvicorn.Config/Server with fakes so start_server returns
    immediately.  Returns a dict with captured Config kwargs."""
    captured: dict = {}

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000
        _loop_factory = None

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def load(self):
            pass

        def get_loop_factory(self):
            return self._loop_factory

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *a, **kw):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        force_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

        def handle_exit(self, sig, frame):
            self.should_exit = True

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)

    def _server_factory(config):
        server = _FakeServer()
        captured["server"] = server
        return server

    monkeypatch.setattr(web_server, "load_config", lambda: {"dashboard": {}})
    monkeypatch.setattr(uvicorn, "Server", _server_factory)
    return captured


def test_start_server_applies_process_local_ssh_bootstrap_state(monkeypatch):
    captured = _stub_uvicorn(monkeypatch)

    web_server.start_server(
        host="127.0.0.1",
        port=0,
        open_browser=False,
        ssh_session_token="s" * 64,
        ssh_owner_nonce="0123456789abcdef",
    )

    assert web_server._SESSION_TOKEN == "s" * 64
    assert web_server._SSH_OWNER_NONCE == "0123456789abcdef"
    assert captured["port"] == 0


def _stub_hard_exit_timer(monkeypatch):
    timers = []
    exits = []

    class _FakeTimer:
        def __init__(self, delay, fn):
            self.delay = delay
            self.fn = fn
            self.daemon = False
            self.started = False
            timers.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(web_server.threading, "Timer", _FakeTimer)
    monkeypatch.setattr(web_server.os, "_exit", lambda code: exits.append(code))
    return timers, exits


def test_start_server_disables_ws_ping_on_loopback(monkeypatch):
    """Loopback binds (the Desktop case) MUST disable uvicorn's protocol-level
    keepalive ping so an event-loop stall can never trigger a false disconnect.

    uvicorn's ws ping runs on the same event loop as agent turns. A single
    synchronous GIL-holding call on a worker thread can starve that loop for
    minutes, so the loop can't process the pong and uvicorn kills an
    otherwise-healthy local connection (#53773 "event loop stalled 226.3s",
    #48445/#50005). On loopback there is no network/proxy path where a
    half-open connection can occur — a dead local client tears the socket down
    with a real FIN/RST that surfaces as WebSocketDisconnect regardless — so
    the ping provides no liveness value and only harms. Assert it is disabled.
    """
    captured = _stub_uvicorn(monkeypatch)

    # Loopback bind => no auth gate, so this reaches the Config constructor.
    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert captured["ws_ping_interval"] is None
    assert captured["ws_ping_timeout"] is None
    assert captured["timeout_graceful_shutdown"] == web_server._DASHBOARD_GRACEFUL_SHUTDOWN_TIMEOUT


def test_start_server_enables_ws_ping_for_half_open_detection(monkeypatch):
    """Non-loopback (public) binds MUST keep the ws ping enabled so half-open
    connections (reverse-proxy 524, dropped Cloudflare Tunnel) raise
    WebSocketDisconnect into the reaping path (#32377).

    The invariant asserted here is that ping stays enabled (non-None, positive)
    and the timeout is never shorter than the interval — not a frozen literal,
    which churns every time the window is retuned. Loopback disables the ping
    (see test_start_server_disables_ws_ping_on_loopback); this covers the
    public-bind half-open case, so the auth gate is active here.
    """
    captured = _stub_uvicorn(monkeypatch)

    # Non-loopback bind so the _is_loopback branch selects the enabled-ping
    # window. Neutralize the auth gate so start_server reaches uvicorn.Config
    # without requiring a registered provider (a real public bind would raise
    # SystemExit here). The ping window keys off the host, not the auth flag.
    monkeypatch.setattr(web_server, "should_require_auth", lambda *a, **k: False)
    web_server.start_server(host="0.0.0.0", port=0, open_browser=False)

    assert captured["ws_ping_interval"] and captured["ws_ping_interval"] > 0
    assert captured["ws_ping_timeout"] and captured["ws_ping_timeout"] > 0
    assert captured["ws_ping_timeout"] >= captured["ws_ping_interval"]
    assert captured["timeout_graceful_shutdown"] == web_server._DASHBOARD_GRACEFUL_SHUTDOWN_TIMEOUT


def test_start_server_can_enable_loopback_ws_ping_for_tunneled_deployments(monkeypatch):
    """Headless loopback dashboards can opt into ping for proxy/tunnel liveness."""
    captured = _stub_uvicorn(monkeypatch)
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "dashboard": {
                "ws_ping_interval": 15,
                "ws_ping_timeout": 12,
            }
        },
    )

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert captured["ws_ping_interval"] == 15.0
    assert captured["ws_ping_timeout"] == 12.0


def test_loopback_ws_ping_interval_implies_timeout(monkeypatch):
    captured = _stub_uvicorn(monkeypatch)
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"dashboard": {"ws_ping_interval": 15}},
    )

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert captured["ws_ping_interval"] == 15.0
    assert captured["ws_ping_timeout"] == 15.0


def test_start_server_bounds_graceful_shutdown(monkeypatch):
    """Dashboard shutdown must not wait forever for open WebSocket tasks.

    uvicorn treats ``timeout_graceful_shutdown=None`` as unbounded. A finite
    value gives systemd-managed dashboards a real SIGTERM exit path instead of
    waiting until the service manager escalates to SIGKILL (#58005).
    """
    captured = _stub_uvicorn(monkeypatch)

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert captured["timeout_graceful_shutdown"] == web_server._DASHBOARD_GRACEFUL_SHUTDOWN_TIMEOUT
    assert captured["timeout_graceful_shutdown"] > 0


def test_dashboard_shutdown_timeouts_enforce_hard_exit_cushion():
    graceful, hard_exit = web_server._dashboard_shutdown_timeouts({
        "dashboard": {
            "graceful_shutdown_timeout": 4,
            "hard_exit_grace": 4.5,
        },
    })

    assert graceful == 4.0
    assert hard_exit == 6.0


def test_dashboard_shutdown_timeouts_reject_non_finite_values():
    graceful, hard_exit = web_server._dashboard_shutdown_timeouts({
        "dashboard": {
            "graceful_shutdown_timeout": ".inf",
            "hard_exit_grace": "nan",
        },
    })

    assert graceful == web_server._DASHBOARD_GRACEFUL_SHUTDOWN_TIMEOUT
    assert hard_exit == web_server._DASHBOARD_HARD_EXIT_GRACE


def test_start_server_uses_custom_shutdown_config(monkeypatch):
    timers, _exits = _stub_hard_exit_timer(monkeypatch)
    captured = _stub_uvicorn(monkeypatch)
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "dashboard": {
                "graceful_shutdown_timeout": 4,
                "hard_exit_grace": 9,
            }
        },
    )

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)
    captured["server"].handle_exit(signal.SIGTERM, None)

    assert captured["timeout_graceful_shutdown"] == 4.0
    assert timers[0].delay == 9.0


def test_start_server_installs_repeated_sigterm_escalation(monkeypatch):
    """start_server wires repeated SIGTERM escalation onto uvicorn.Server."""
    timers, _exits = _stub_hard_exit_timer(monkeypatch)
    captured = _stub_uvicorn(monkeypatch)

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)
    server = captured["server"]

    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit is True
    assert server.force_exit is False

    server.handle_exit(signal.SIGTERM, None)
    assert server.force_exit is True
    assert len(timers) == 1


def test_dashboard_server_sigterm_arms_hard_exit_failsafe(monkeypatch):
    """A first SIGTERM arms a process-level failsafe; repeats still force uvicorn."""
    timers, exits = _stub_hard_exit_timer(monkeypatch)

    class _Server:
        should_exit = False
        force_exit = False

        def handle_exit(self, sig, frame):
            self.should_exit = True

    server = _Server()
    web_server._install_dashboard_sigterm_escalation(
        server,
        hard_exit_grace=web_server._DASHBOARD_HARD_EXIT_GRACE,
    )

    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit is True
    assert server.force_exit is False
    assert len(timers) == 1
    assert timers[0].delay == web_server._DASHBOARD_HARD_EXIT_GRACE
    assert timers[0].daemon is True
    assert timers[0].started is True

    timers[0].fn()
    assert exits == [0]

    server.handle_exit(signal.SIGTERM, None)
    assert server.force_exit is True
    assert len(timers) == 1


def test_dashboard_hard_exit_failsafe_exits_even_if_stderr_write_fails(monkeypatch):
    timers, exits = _stub_hard_exit_timer(monkeypatch)

    def _raise_write(*args, **kwargs):
        raise RuntimeError("stderr unavailable")

    monkeypatch.setattr(web_server.os, "write", _raise_write)
    web_server._arm_dashboard_hard_exit_failsafe(
        signal.SIGTERM,
        web_server._DASHBOARD_HARD_EXIT_GRACE,
    )

    timers[0].fn()

    assert exits == [0]


def test_dashboard_signal_wrapper_preserves_sigint_escalation(monkeypatch):
    """The SIGTERM wrapper must not break uvicorn's repeated-SIGINT path."""
    timers, _exits = _stub_hard_exit_timer(monkeypatch)

    class _Server:
        should_exit = False
        force_exit = False

        def handle_exit(self, sig, frame):
            if self.should_exit and sig == signal.SIGINT:
                self.force_exit = True
            else:
                self.should_exit = True

    server = _Server()
    web_server._install_dashboard_sigterm_escalation(
        server,
        hard_exit_grace=web_server._DASHBOARD_HARD_EXIT_GRACE,
    )

    server.handle_exit(signal.SIGINT, None)
    assert server.should_exit is True
    assert server.force_exit is False

    server.handle_exit(signal.SIGINT, None)
    assert server.force_exit is True
    assert len(timers) == 0


def test_dashboard_sigterm_wrapper_escalates_real_uvicorn_server(monkeypatch):
    timers, _exits = _stub_hard_exit_timer(monkeypatch)
    config = uvicorn.Config(web_server.app, host="127.0.0.1", port=0)
    server = uvicorn.Server(config)

    web_server._install_dashboard_sigterm_escalation(
        server,
        hard_exit_grace=web_server._DASHBOARD_HARD_EXIT_GRACE,
    )

    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit is True
    assert server.force_exit is False

    server.handle_exit(signal.SIGTERM, None)
    assert server.force_exit is True
    assert len(timers) == 1


def test_start_server_runs_on_uvicorns_loop_factory(monkeypatch):
    """The dashboard/desktop backend must serve uvicorn on the loop *uvicorn*
    selects, not the interpreter default.

    On Windows ``asyncio.run`` defaults to a ProactorEventLoop, but uvicorn's
    socket-serving stack forces a SelectorEventLoop on win32
    (``uvicorn/loops/asyncio.py``). Serving on the proactor loop binds a socket
    that never accepts — the backend prints "Skipping web UI build" and hangs
    forever with the port LISTENING but no TCP handshake (#50641). We fix that
    by routing the serve call through ``uvicorn._compat.asyncio_run`` with
    ``config.get_loop_factory()`` — exactly what ``uvicorn.Server.run`` does.

    This asserts the behavioral contract: on Windows the loop factory the runner
    receives is the one uvicorn's own Config produced, and bare ``asyncio.run``
    is never the serve path when the loop-factory runner exists.
    """
    _stub_uvicorn(monkeypatch)

    # The fix only changes behavior on win32; simulate it so the Windows branch
    # is actually exercised on a POSIX CI host.
    monkeypatch.setattr(web_server.sys, "platform", "win32")

    # The fake Config (installed by _stub_uvicorn) returns its ``_loop_factory``
    # from get_loop_factory(). Pin a sentinel so we can assert it is threaded
    # through to the runner unchanged.
    sentinel_factory = object()
    monkeypatch.setattr(uvicorn.Config, "_loop_factory", sentinel_factory, raising=False)

    seen: dict = {}

    def _fake_runner(coro, *, loop_factory=None):
        seen["loop_factory"] = loop_factory
        coro.close()  # drain without an event loop

    monkeypatch.setattr("uvicorn._compat.asyncio_run", _fake_runner, raising=False)

    # Bare asyncio.run must NOT be the serve path on Windows when the
    # loop-factory runner is importable.
    called_bare = {"hit": False}

    def _guard_asyncio_run(coro):
        called_bare["hit"] = True
        coro.close()
        return None

    monkeypatch.setattr(asyncio, "run", _guard_asyncio_run)

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert seen.get("loop_factory") is sentinel_factory, (
        "start_server must pass uvicorn's get_loop_factory() result to the "
        "runner so Windows serves on a SelectorEventLoop"
    )
    assert called_bare["hit"] is False, (
        "start_server must not fall back to bare asyncio.run when uvicorn's "
        "loop-factory runner is available"
    )


def test_start_server_keeps_bare_asyncio_run_on_posix(monkeypatch):
    """POSIX behavior must be byte-for-byte unchanged: serve via the plain
    ``asyncio.run(_serve())`` path, never the Windows loop-factory branch.

    The #50641 fix is intentionally win32-scoped to keep the blast radius
    minimal — Python's default loop on POSIX is already a SelectorEventLoop
    (or uvloop), which is what uvicorn serves on, so there is nothing to fix.
    """
    _stub_uvicorn(monkeypatch)
    monkeypatch.setattr(web_server.sys, "platform", "linux")

    # If the Windows branch were taken, the loop-factory runner would fire.
    runner_called = {"hit": False}

    def _fake_runner(coro, *, loop_factory=None):
        runner_called["hit"] = True
        coro.close()

    monkeypatch.setattr("uvicorn._compat.asyncio_run", _fake_runner, raising=False)

    bare_called = {"hit": False}

    def _fake_asyncio_run(coro):
        bare_called["hit"] = True
        coro.close()
        return None

    monkeypatch.setattr(asyncio, "run", _fake_asyncio_run)

    web_server.start_server(host="127.0.0.1", port=0, open_browser=False)

    assert bare_called["hit"] is True, "POSIX must serve via bare asyncio.run"
    assert runner_called["hit"] is False, (
        "POSIX must not take the Windows loop-factory branch"
    )
