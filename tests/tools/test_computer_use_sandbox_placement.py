"""computer_use stays where the terminal backend is, whatever the approval state.

With a sandboxed terminal (``docker``) every cua-driver process, the embedded daemon that ``--yolo`` /
``approvals.mode: off`` / ``bounded`` select included, runs through the sandbox's exec prefix on the sandbox
display. Only a gateway-hosted desktop (``terminal.backend: local``, or ``bot_desktop.placement: gateway``) spawns
the host binary. The permission mode changes what the driver may do, never which machine it runs on.
"""
from __future__ import annotations

import asyncio
import contextlib
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.bot_desktop import placement, runtime
from tools.computer_use import cua_backend, cua_backend_daemon, cua_backend_driver, cua_backend_session
from tools.environments import streams

pytestmark = pytest.mark.platforms("linux")

HOST_DRIVER = "/opt/host-cua-driver"


class _FakeDocker:
    """Stand-in for DockerEnvironment: only what ``streams.exec_prefix`` reads."""
    _docker_exe = "docker"
    _container_id = "c0ffee"


@pytest.fixture(autouse=True)
def _fresh_computer_use_state():
    from tools.computer_use.tool import reset_backend_for_tests

    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


@pytest.fixture
def machines(monkeypatch):
    """Wire a docker sandbox (screen up on :20) next to a host driver, and record where cua-driver runs.

    ``machines.use(backend)`` picks ``terminal.backend``; ``spawned`` holds the argv of every long-lived driver
    process, ``ran`` the argv of every one-shot (``status``, ``stop``).
    """
    spawned: list[list[str]] = []
    spawn_kwargs: list[dict] = []
    ran: list[list[str]] = []
    process = Mock()
    process.poll.return_value = None
    process.stderr = []
    process.wait.return_value = 0

    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: True)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":20", "XAUTHORITY": "/x"})
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: _FakeDocker())
    monkeypatch.setattr(runtime, "touch_activity", lambda: None)
    from tools.bot_desktop import sandbox_host
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: "pn")
    monkeypatch.setattr(streams, "exec_prefix", lambda env, *, user=None, interactive=True:
                        ["docker", "exec", "-i", "-u", user, env._container_id])
    monkeypatch.setattr(cua_backend_daemon, "spawn_owned", lambda argv, **kw: spawned.append(argv) or spawn_kwargs.append(kw) or process)
    monkeypatch.setattr(cua_backend_daemon, "release_process_group_once_empty", lambda *a, **kw: None)
    monkeypatch.setattr(cua_backend, "_run_quiet",
                        lambda argv, **kw: ran.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(cua_backend, "_cua_no_overlay", lambda: False)
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"]))

    def use(backend: str) -> None:
        monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)
        # A sandboxed machine has no usable host driver at all: reaching for one is the failure under test.
        monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd",
                            lambda override=None: HOST_DRIVER if backend == "local" else None)

    return SimpleNamespace(use=use, spawned=spawned, spawn_kwargs=spawn_kwargs, ran=ran, process=process)


def _bypass(monkeypatch, how: str) -> None:
    """Switch approvals off the way a user would: ``approvals.mode: off``, or a session ``--yolo`` / ``/yolo``."""
    from tools import approval, approval_context

    if how == "off":
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
    else:
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)


def _backend_for(monkeypatch, tmp_path, approvals: str):
    """The backend ``computer_use`` would build for this approval state (real mode resolution)."""
    from tools.computer_use import tool

    cfg: dict = {}
    if approvals == "bounded":
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("version: 2\nmode: bounded\n", encoding="utf-8")
        cfg = {"permission_mode": "bounded", "capability_manifest": str(manifest)}
    elif approvals in ("off", "yolo"):
        _bypass(monkeypatch, approvals)
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: cfg)
    mode = tool._cua_permission_mode("sid")
    assert mode == {"standard": "standard", "bounded": "bounded", "off": "unrestricted", "yolo": "unrestricted"}[approvals]
    return cua_backend.CuaDriverBackend(permission_mode=mode)


def _mcp_launch(session) -> tuple[str, list[str]]:
    """The command the session's MCP transport would spawn (the site that used to prefer the host daemon)."""
    import mcp.client.stdio as mcp_stdio

    captured: list = []

    class _Stop(Exception):
        pass

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(params):
        captured.append(params)
        raise _Stop
        yield  # pragma: no cover

    real, mcp_stdio.stdio_client = mcp_stdio.stdio_client, _fake_stdio_client
    try:
        with pytest.raises(_Stop):
            asyncio.run(session._lifecycle_coro())
    finally:
        mcp_stdio.stdio_client = real
    return captured[0].command, list(captured[0].args)


@pytest.mark.parametrize("approvals", ["off", "yolo"])
def test_approvals_off_keeps_the_docker_sandbox_driver_in_the_sandbox(machines, monkeypatch, tmp_path, approvals):
    machines.use("docker")
    backend = _backend_for(monkeypatch, tmp_path, approvals)
    daemon = backend._embedded_daemon
    assert daemon is not None  # unrestricted still means a private daemon...

    daemon.start()  # ...but one that lives next to the screen it drives

    (serve,) = machines.spawned
    assert serve[:6] == ["docker", "exec", "-i", "-u", "pn", "c0ffee"], serve
    script = serve[-1]
    assert "cua-driver serve --embedded" in script and "--permission-mode unrestricted" in script
    assert "--dangerously-bypass-approvals" in script
    assert "CUA_DRIVER_PERMISSION_MODE=unrestricted" in script and "DISPLAY=:20" in script
    assert daemon.socket_path.startswith("/tmp/hc-") and daemon.socket_path in script  # the sandbox's /tmp
    assert all(argv[0] == "docker" for argv in machines.ran), machines.ran  # readiness probe too
    # The exec client's stdin is held open: closing it (Hermes exiting however it does) ends the daemon in the sandbox.
    assert machines.spawn_kwargs[0]["stdin"] == subprocess.PIPE and "cat <&3" in script

    command, args = _mcp_launch(backend._session)
    assert command == "docker" and args[:5] == ["exec", "-i", "-u", "pn", "c0ffee"]
    assert "cua-driver mcp --embedded --socket " + daemon.socket_path in args[-1]

    daemon.stop()
    assert machines.ran[-1][0] == "docker" and "cua-driver stop --socket " + daemon.socket_path in machines.ran[-1][-1]
    machines.process.stdin.close.assert_called_once()
    assert HOST_DRIVER not in [part for argv in [*machines.spawned, *machines.ran] for part in argv]


@pytest.mark.parametrize("approvals", ["off", "yolo", "bounded"])
def test_local_backend_runs_the_host_driver_whatever_the_approvals(machines, monkeypatch, tmp_path, approvals):
    machines.use("local")
    daemon = _backend_for(monkeypatch, tmp_path, approvals)._embedded_daemon

    daemon.start()

    (serve,) = machines.spawned
    assert serve[:2] == [HOST_DRIVER, "serve"] and "docker" not in serve
    assert machines.spawn_kwargs[0]["stdin"] == subprocess.DEVNULL
    assert daemon.proxy_invocation()[0] == HOST_DRIVER


def test_bounded_manifest_cannot_reach_a_sandboxed_driver_so_it_fails_closed(machines, monkeypatch, tmp_path):
    machines.use("docker")
    daemon = _backend_for(monkeypatch, tmp_path, "bounded")._embedded_daemon

    with pytest.raises(RuntimeError, match="placement: gateway"):
        daemon.start()
    assert machines.spawned == [] and machines.ran == []  # never started anywhere, least of all the host


def test_standard_mode_keeps_the_sandbox_driver_and_has_no_daemon(machines, monkeypatch, tmp_path):
    machines.use("docker")
    backend = _backend_for(monkeypatch, tmp_path, "standard")
    assert backend._embedded_daemon is None

    command, args = _mcp_launch(backend._session)
    assert command == "docker" and "cua-driver mcp" in args[-1] and "export DISPLAY=:20" in args[-1]


def _cli_fallback_command(monkeypatch, backend) -> list[str]:
    """argv of the ``cua-driver call`` the MCP -> CLI fallback transport would run."""
    ran: list[list[str]] = []
    monkeypatch.setattr(cua_backend_session, "_cli_run_json", lambda cmd, env, name, timeout: ran.append(cmd) or {})
    monkeypatch.setattr(cua_backend_session, "_cli_result", lambda payload, shot_file: {})
    backend._session._call_tool_via_cli("list_windows", {}, 5.0)
    return ran[0]


@pytest.mark.parametrize("approvals", ["standard", "off"])
def test_cli_fallback_on_a_sandboxed_backend_runs_in_the_sandbox(machines, monkeypatch, tmp_path, approvals):
    machines.use("docker")
    backend = _backend_for(monkeypatch, tmp_path, approvals)
    if backend._embedded_daemon is not None:
        backend._embedded_daemon.start()

    cmd = _cli_fallback_command(monkeypatch, backend)

    assert cmd[:2] == ["docker", "exec"] and "cua-driver call list_windows" in cmd[-1]
    if approvals == "off":
        assert "--socket " + backend._embedded_daemon.socket_path in cmd[-1]


@pytest.mark.parametrize("approvals", ["standard", "off"])
def test_cli_fallback_on_a_local_backend_runs_the_host_driver(machines, monkeypatch, tmp_path, approvals):
    machines.use("local")
    backend = _backend_for(monkeypatch, tmp_path, approvals)
    if backend._embedded_daemon is not None:
        backend._embedded_daemon.start()

    cmd = _cli_fallback_command(monkeypatch, backend)

    assert cmd[:3] == [HOST_DRIVER, "call", "list_windows"]
